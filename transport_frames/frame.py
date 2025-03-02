""" Module for creating and weighting transport frame """

import warnings

import geopandas as gpd
import momepy
import networkx as nx
import osmnx as ox
import pandas as pd
from shapely import unary_union
from shapely.geometry import Point, Polygon
from shapely.ops import unary_union

from transport_frames.utils.helper_funcs import BaseSchema, _determine_ref_type


warnings.simplefilter("ignore", UserWarning)
import re

import contextily as ctx
import matplotlib.pyplot as plt
import pandera as pa
from pandera.typing import Series


class PointSchema(BaseSchema):
    """
    Schema for validating points.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the points, default is [shapely.Point]
    """
    _geom_types = [Point]


class CentersSchema(BaseSchema):
    """
    Schema for validating central cities.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the blocks, default is [shapely.Polygon]
    """
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Point]


def get_frame(graph: nx.MultiDiGraph, 
              admin_centers: gpd.GeoDataFrame, 
              area_polygon: gpd.GeoDataFrame, 
              region_polygons: gpd.GeoDataFrame, 
              country_polygon: gpd.GeoDataFrame=ox.geocode_to_gdf("RUSSIA"))-> nx.MultiDiGraph:
    """
    Creates frame from graph. Edges are filtered based on reg [1,2].
    Region point nodes are assigned with 'city_name' attribute.
    Points, connecting the roads with other regions or countries are marked with 'exit' and 'exit_country'.

    Parameters
    ----------
    graph: nx.MultiDiGraph
        City network graph with classified roads and edges.
    admin_centers : gpd.GeoDataFrame
        Administrative region centers points to assign 'city_name' attribute
    region_polygons : gpd.GeoDataFrame
        Polygons or regions of the country to assign 'border_region' attribute to exits
    country_polygon : gpd.GeoDataFrame
        Polygon of the country to assign 'exit_country' attribute

    Returns
    -------
    nx.MultiDiGraph
        Frame of the graph with added 'city_node','exit','exit_country','border_region' attributes
    """

    admin_centers = CentersSchema(admin_centers).to_crs(graph.graph["crs"]).copy()
    frame = _filter_roads(graph)
    frame = _assign_city_names_to_nodes(admin_centers, frame)
    frame = mark_exits(frame, area_polygon, region_polygons, country_polygon)

    return frame


def _filter_roads(graph) -> nx.MultiDiGraph:
    """
    Filters the graph to include only reg_1 and reg_2 roads.
    
    Parameters
    ----------
    graph : nx.MultiDiGraph
        graph with edges, containing 'reg' attribute
    
    Returns
    -------
    nx.MultiDiGraph
        Frame of filtered edges with 'reg' in [1, 2] 
    """
    edges_to_keep = [(u, v, k) for u, v, k, d in graph.edges(data=True, keys=True) if d.get("reg") in ([1, 2])]
    frame = graph.edge_subgraph(edges_to_keep).copy()
    for node, data in frame.nodes(data=True):
        data["nodeID"] = node
    return frame


def _assign_city_names_to_nodes(points, frame, max_distance=3000)-> nx.MultiDiGraph:
    """
    Assigns city names to nodes in the graph based on proximity to city centers.
    
    Parameters
    ----------
    points : gpd.GeoDataFrame
        Region centers points
    frame : nx.MultiDiGraph
        Frame of filtered edges with 'reg' in [1, 2] 
    max_distance : int
        Max distance (meters) on which a node can be assigned to a city

    Returns
    -------
    nx.MultiDiGraph
        A frame with nodes assigned with 'city_name' attribute
    """

    points = points.to_crs(frame.graph["crs"])  # Ensure same CRS
    n, e = momepy.nx_to_gdf(frame)

    # Find nearest nodes within max_distance
    projected = gpd.sjoin_nearest(points, n, how="left", distance_col="distance", max_distance=max_distance)

    assigned_cities = set()  # Track assigned cities
    for row in projected.itertuples():
        city_name = row.name
        node_id = row.nodeID

        if pd.notna(node_id):
            for _, d in frame.nodes(data=True):
                if d.get("nodeID") == node_id:
                    d["city_name"] = city_name
                    assigned_cities.add(city_name)  # Mark as assigned

    # Find cities that were not assigned
    missed_cities = set(points["name"]) - assigned_cities

    if missed_cities:
        print("Some cities weren’t assigned to nodes due to remote location:", missed_cities)

    return frame


def mark_exits(
    frame: nx.MultiDiGraph, 
    area_polygon: gpd.GeoDataFrame, 
    regions_polygons: gpd.GeoDataFrame, 
    country_polygon: gpd.GeoDataFrame
) -> nx.MultiDiGraph:
    """
    Assign the 'exit', 'exit_country' and 'boorder_region' attribute to nodes 
    in a GeoDataFrame based on their intersection with city boundaries.

    Parameters:
    -----------
    frame : nx.MultiDiGraph
        Frame graph
    area_polygon : gpd.GeoDataFrame
        GeoDataFrame with boundary of the inverstigated region
    regions_polygons : gpd.GeoDataFrame
        GeoDataFrame with polygons of the regions with 'name' column
    country_polygon : gpd.GeoDataFrame
        GeoDataFrame with boundary of the inverstigated country

    Returns
    -------
    nx.MultiDiGraph 
        Frame with assigned 'exit', 'exit_country' and 'boorder_region' node attributes
    """

    gdf_nodes = momepy.nx_to_gdf(frame)[0]
    city_boundary = unary_union(area_polygon.to_crs(gdf_nodes.crs).boundary)
    regions_polygons.to_crs(gdf_nodes.crs, inplace=True)
    country_polygon.to_crs(gdf_nodes.crs, inplace=True)
    gdf_nodes.loc[:, "exit"] = gdf_nodes["geometry"].apply(
        lambda point: True if city_boundary.intersects(point.buffer(0.1)) else False
    )
    if len(gdf_nodes[gdf_nodes["exit"] == True]) == 0:
        print("There are no region exits. Try using a larger polygon for downloading from osm")
    exits = gdf_nodes[gdf_nodes.exit == 1].copy()
    country_boundary = unary_union(country_polygon.to_crs(exits.crs).boundary)

    exits.loc[:, "exit_country"] = exits["geometry"].apply(
        lambda point: True if country_boundary.intersects(point.buffer(0.1)) else False
    )
    gdf_nodes = gdf_nodes.assign(exit_country=exits["exit_country"])
    gdf_nodes["exit_country"] = False
    gdf_nodes.loc[exits.index, "exit_country"] = exits["exit_country"].astype(bool)
    gdf_nodes["exit_country"] = gdf_nodes["exit_country"].fillna(False)
    gdf_nodes = add_region_attr(gdf_nodes, regions_polygons, area_polygon)

    for i, (node, data) in enumerate(frame.nodes(data=True)):
        data["exit"] = gdf_nodes.iloc[i]["exit"]
        data["exit_country"] = gdf_nodes.iloc[i]["exit_country"]
        data["border_region"] = gdf_nodes.iloc[i]["border_region"]

    return frame


def add_region_attr(n: gpd.GeoDataFrame, 
                    regions: gpd.GeoDataFrame, 
                    polygon_buffer: gpd.GeoDataFrame)-> gpd.GeoDataFrame:
    """
    Add a 'border_region' attribute to nodes based on their intersection with region polygons.

    Parameters
    ----------
    n : gpd.GeoDataFrame
        Nodes GeoDataFrame with 'exit' and 'exit_country' attributes
    regions : GeoDataFrame
        Regions GeoDataFrame with 'name' column
    polygon_buffer : gpd.GeoDataFrame
        GeoDataFrame of region of interest boundary

    Returns
    -------
    gpd.GeoDataFrame
        Updated nodes GeoDataFrame with 'border_region' attribute
    """
    exits = n[n["exit"] == 1]
    exits = exits.to_crs(n.crs)
    exits.loc[:, "buf"] = exits["geometry"].buffer(1000)
    filtered_regions = _filter_polygons_by_buffer(regions, polygon_buffer, n.crs)
    joined_gdf = gpd.sjoin(
        exits.set_geometry("buf"),
        filtered_regions.to_crs(exits.crs),
        how="left",
        predicate="intersects",
    )
    joined_gdf = joined_gdf.drop_duplicates(subset="geometry")
    exits.loc[:, "border_region"] = joined_gdf["name"]
    n.loc[:, "border_region"] = exits["border_region"]
    return n


def _filter_polygons_by_buffer(gdf_polygons: gpd.GeoDataFrame, 
                               polygon_buffer: Polygon, 
                               crs)-> gpd.GeoDataFrame:
    """
    Extract and filter region polygons based on a buffer around a given polygon.

    Parameters
    -----------
    gdf_polygons : gpd.GeoDataFrame
        Gdf of regions polygons
    polygon_buffer : Polygon
        Polygon of the buffer polygon.

    Returns
    -------
    - GeoDataFrame: Filtered GeoDataFrame of region polygons.
    """
    gdf_polygons = gdf_polygons.to_crs(crs)
    polygon_buffer = polygon_buffer.to_crs(crs)
    polygon_buffer = gpd.GeoDataFrame({"geometry": polygon_buffer.buffer(0.1)}, crs=polygon_buffer.crs).to_crs(
        gdf_polygons.crs
    )
    gdf_polygons = gpd.overlay(gdf_polygons, polygon_buffer, how="difference")
    buffer_polygon = polygon_buffer.buffer(5000)
    filtered_gdf = gdf_polygons[gdf_polygons.intersects(buffer_polygon.unary_union)]
    return filtered_gdf


def plot_frame(frame: nx.MultiDiGraph, 
               zoom_out_factor: float = 1.1) -> None:
    """
    Plot the transport frame with exit points, city nodes, and categorized roads.

    The function visualizes the road network using a basemap and styles edges and nodes based on their attributes:
    
    - **Edges (Road Segments)**:
      - `reg == 1` → Red (Higher priority roads)
      - `reg == 2` → Blue (Lower priority roads)
    - **Nodes (Intersections & Key Points)**:
      - `exit == 1` → Green (Exit points)
      - `exit_country == 1` → Black (Country exit points)
      - `city_name.notna()` → Red (City center nodes)

    Parameters
    ----------
    frame : nx.MultiDiGraph
        The road network graph.
    zoom_out_factor : float, optional
        Scaling factor to adjust the map extent by zooming out (default: `1.1`).

    Returns
    -------
    None
        Displays a styled transport frame plot with a basemap and legend.
    """
    # Convert graph to GeoDataFrames
    nodes, edges = momepy.nx_to_gdf(frame)

    # Convert to Web Mercator (EPSG:3857)
    nodes = nodes.to_crs(epsg=3857)
    edges = edges.to_crs(epsg=3857)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    # Get bounding box and compute margins for zoom-out effect
    xmin, ymin, xmax, ymax = edges.total_bounds
    x_margin = (xmax - xmin) * (zoom_out_factor - 1) / 2
    y_margin = (ymax - ymin) * (zoom_out_factor - 1) / 2

    # Apply new limits with margins
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(ymin - y_margin, ymax + y_margin)

    # Add basemap
    basemap = ctx.providers.OpenStreetMap.Mapnik = ctx.providers.OpenStreetMap.Mapnik
    ctx.add_basemap(ax, source=basemap, crs=3857, alpha=0.7)

    # **Plot edges (roads)**
    edges[edges["reg"] == 1].plot(ax=ax, color="red", linewidth=1.2, alpha=0.8, label="Reg 1")
    edges[edges["reg"] == 2].plot(ax=ax, color="blue", linewidth=1.2, alpha=0.8, label="Reg 2")

    # **Plot nodes (intersections, exits, cities)**
    nodes[nodes["exit"] == 1].plot(ax=ax, color="green", markersize=30, label="Exit Nodes")
    nodes[nodes["exit_country"] == 1].plot(ax=ax, color="black", markersize=30, label="Country Exit Nodes")
    nodes[nodes["city_name"].notna()].plot(ax=ax, color="red", markersize=40, label="City Nodes")

    # Remove axis labels and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Add legend
    ax.legend(loc="lower right")

    # Show plot
    plt.show()


def weigh_roads(frame: nx.MultiDiGraph, restricted_terr_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Assigns and normalizes weights for roads in the network based on their proximity to exits.

    The function calculates road segment (edge) weights by evaluating their connections to 
    exit points and the type of regions they traverse. If provided, restricted territories 
    influence weight assignment. The function then normalizes weights for further analysis.

    Parameters
    ----------
    frame : nx.MultiDiGraph
        The road network graph where nodes represent intersections or exits, 
        and edges represent road segments.
    restricted_terr_gdf : gpd.GeoDataFrame, optional
        GeoDataFrame containing restricted territories that may influence 
        road weight calculations (default: `None`).

    Returns
    -------
    nx.MultiDiGraph
        The updated road network graph with assigned and normalized road segment weights.
    """
    frame = frame.copy()
    n, e = momepy.nx_to_gdf(frame)
    n, e, frame = _mark_ref_type(n, e, frame)
    if restricted_terr_gdf is not None:
        country_exits = n[n["exit_country"] == True].copy()

        # Преобразуем CRS для совместимости
        restricted_terr_gdf = restricted_terr_gdf.to_crs(country_exits.crs)

        # Для каждой страны из restricted_terr_gdf применяем логику
        for _, row in restricted_terr_gdf.iterrows():
            border_transformed = row["geometry"]
            buffer_area = border_transformed.buffer(300)
            mask = country_exits.geometry.apply(lambda x: x.intersects(buffer_area))

            # Применяем метку (mark) к соответствующим участкам
            n.loc[mask[mask == True].index, "ref_type"] = row["mark"]

    e["weight"] = 0.0
    n["weight"] = 0.0
    exits = n[n["exit"] == 1]
    for i1, start_node in exits.iterrows():
        for i2, end_node in exits.iterrows():
            if i1 == i2:
                continue
            if pd.notna(start_node["border_region"]) and start_node["border_region"] == end_node["border_region"]:
                continue
            if start_node.geometry.buffer(15000).intersects(end_node.geometry.buffer(15000)) and (
                pd.isna(start_node["exit_country"]) == pd.isna(end_node["exit_country"])
            ):
                continue
            if start_node["exit_country"] == 1 and end_node["exit_country"] == 1:
                continue

            weight = _get_weight(start_node["ref_type"], end_node["ref_type"], end_node["exit_country"])

            try:
                path = nx.astar_path(frame, i1, i2, weight="time_min")
            except nx.NetworkXNoPath:
                continue

            for j in range(len(path) - 1):
                n.loc[(n["nodeID"] == path[j]), "weight"] += weight
                e.loc[
                    (e["node_start"] == path[j]) & (e["node_end"] == path[j + 1]),
                    "weight",
                ] += weight
            n.loc[(n["nodeID"] == path[j + 1]), "weight"] += weight

    n["weight"] = round(n.weight, 3)
    min_weight = e["weight"].min()
    max_weight = e["weight"].max()
    e["norm_weight"] = (e["weight"] - min_weight) / (max_weight - min_weight)

    for i, (e1, e2, k, data) in enumerate(frame.edges(data=True, keys=True)):
        data["weight"] = e.iloc[[i]]["weight"][i]
        data["norm_weight"] = e.iloc[[i]]["norm_weight"][i]

    for i, (node, data) in enumerate(frame.nodes(data=True)):
        data["exit"] = n.iloc[i]["exit"]
        data["exit_country"] = n.iloc[i]["exit_country"]
        data["weight"] = n.iloc[i]["weight"]
        data["ref_type"] = n.iloc[i]["ref_type"]

    return frame


def _mark_ref_type(n: gpd.GeoDataFrame, 
                   e: gpd.GeoDataFrame, 
                   frame: nx.MultiDiGraph
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]:
    """
    Assign reference types to nodes in the road network based on proximity to reference edges.

    This function identifies nodes marked as exits and assigns reference values and types 
    based on the closest reference edge. The assigned values help categorize roads 
    according to their significance.

    Parameters
    ----------
    n : gpd.GeoDataFrame
        GeoDataFrame of road network nodes, including exit nodes to be marked with reference types.
    e : gpd.GeoDataFrame
        GeoDataFrame of road network edges containing reference attributes used for classification.
    frame : nx.MultiDiGraph
        The road network graph where nodes represent intersections or exits.

    Returns
    -------
    tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]
        - Updated GeoDataFrame of nodes with assigned reference values and types.
        - The original GeoDataFrame of edges.
        - The updated road network graph with relabeled nodes.
    """
    n["ref"] = None
    ref_edges = e[e["ref"].notna()]

    for idx, node in n.iterrows():

        if node["exit"] == 1:
            point = node.geometry
            distances = ref_edges.geometry.distance(point)
            if not distances.empty:
                nearest_edge = ref_edges.loc[distances.idxmin()]
                ref_value = nearest_edge["ref"]
                if isinstance(ref_value, list):
                    ref_value = tuple(ref_value)
                if isinstance(ref_value, str):
                    ref_value = (ref_value,)
                n.at[idx, "ref"] = ref_value
                n.at[idx, "ref_type"] = _determine_ref_type(ref_value)
    n = n.set_index("nodeID")
    mapping = {node: data["nodeID"] for node, data in frame.nodes(data=True)}
    n.reset_index(inplace=True)
    frame = nx.relabel_nodes(frame, mapping)
    return n, e, frame


def _determine_ref_type(ref: str) -> float:
    """
    Convert a reference string into a numeric classification.

    This function categorizes road references based on predefined patterns, 
    assigning a numeric value that represents the road type.

    Parameters
    ----------
    ref : str
        A road reference string that follows a standard naming convention.

    Returns
    -------
    float
        A numeric code corresponding to the identified reference type. If no match is found, 
        a default classification (2.3) is assigned.
    """
    patterns = {
        1.1: r"М-\d+",
        1.2: r"Р-\d+",
        1.3: r"А-\d+",
        2.1: r"..Р-\d+",
        2.2: r"..А-\d+",
    }
    for value in ref:
        for ref_type, pattern in patterns.items():
            if re.match(pattern, value):
                return ref_type
    return 2.3


def _get_weight(start: float, end: float, exit: bool) -> float:
    """
    Compute the weight for a road segment based on reference types and exit status.

    This function determines the weight of a road connection by considering the reference types 
    of the start and end nodes, as well as whether the segment represents an exit. Different 
    weight matrices are used for exit and non-exit scenarios.

    Parameters
    ----------
    start : float
        The reference type classification of the start node.
    end : float
        The reference type classification of the end node.
    exit : bool
        Indicates whether the segment represents an exit (True if exit, False otherwise).

    Returns
    -------
    float
        The weight value derived from the corresponding weight matrix.
    """
    dict = {1.1: 0, 1.2: 1, 1.3: 2, 2.1: 3, 2.2: 4, 2.3: 5, 0.0: 6, 0.5: 7}
    if exit == 1:
        matrix = [
            [0.12, 0.12, 0.12, 0.12, 0.12, 0.12, 0.00001, 0.05],  # 2.1.1
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10, 0.00001, 0.05],  # 2.1.2
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.00001, 0.05],  # 2.1.3
            [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.00001, 0.05],  # 2.2.1
            [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.00001, 0.05],  # 2.2.2
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05],  # 2.2.3
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00001, 0.05],  # 2.2.3
            [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05],
        ]
    else:

        matrix = [
            [0.08, 0.08, 0.08, 0.08, 0.08, 0.08, 0.00001, 0.05],  # 2.1.1
            [0.07, 0.07, 0.07, 0.07, 0.07, 0.07, 0.00001, 0.05],  # 2.1.2
            [0.06, 0.06, 0.06, 0.06, 0.06, 0.06, 0.00001, 0.05],  # 2.1.3
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05],  # 2.2.1
            [0.04, 0.04, 0.04, 0.04, 0.04, 0.04, 0.00001, 0.05],  # 2.2.2
            [0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.00001, 0.05],  # 2.2.3
            [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05],
        ]
    return matrix[dict[end]][dict[start]]


def plot_weighted_roads(
    weighted_graph: nx.MultiDiGraph, scaling_factor: int = 10, zoom_out_factor: float = 1.1
) -> None:
    """
    Visualize a weighted road network on a map.

    This function plots a road network where the thickness of edges corresponds to their 
    assigned weights. The visualization includes a basemap for spatial context.

    Parameters
    ----------
    weighted_graph : nx.MultiDiGraph
        The road network graph where edges contain weight attributes.
    scaling_factor : int, optional
        Factor to scale line thickness based on edge weight (default: 10).
    zoom_out_factor : float, optional
        Factor to slightly expand the map view beyond the network boundaries (default: 1.1).

    Returns
    -------
    None
        Displays the road network plot with weighted edges.
    """
    nodes, edges = momepy.nx_to_gdf(weighted_graph)

    # Filter only weighted edges
    edges = edges[edges["weight"] > 0]

    # Normalize weight for line thickness
    edges["scaled_weight"] = edges["weight"] / edges["weight"].max() * scaling_factor

    # Convert to Web Mercator (EPSG:3857) for basemap compatibility
    edges = edges.to_crs(epsg=3857)

    # Set up the figure
    fig, ax = plt.subplots(figsize=(10, 10))

    xmin, ymin, xmax, ymax = edges.total_bounds
    x_margin = (xmax - xmin) * (zoom_out_factor - 1) / 2
    y_margin = (ymax - ymin) * (zoom_out_factor - 1) / 2

    # Apply new limits with margins
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(ymin - y_margin, ymax + y_margin)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=3857, alpha=1)

    # Plot weighted roads
    edges.plot(ax=ax, linewidth=edges["scaled_weight"], color="blue", alpha=0.8)

    # Remove axis labels and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Show plot
    plt.show()
