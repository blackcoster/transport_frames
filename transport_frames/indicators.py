""" Module for getting area and territory indicators """

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd
import pandera as pa
from iduedu import get_adj_matrix_gdf_to_gdf, get_single_public_transport_graph
from pandera.typing import Series
from shapely.geometry import MultiPolygon, Point, Polygon

from transport_frames.utils.helper_funcs import BaseSchema


class PolygonSchema(BaseSchema):
    """
    Schema for validating polygons.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the blocks, default is [shapely.Polygon]
    """
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Polygon, MultiPolygon]


class PointSchema(BaseSchema):
    """
    Schema for validating points.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the points, default is [shapely.Point]
    """
    _geom_types = [Point]


def calculate_distances(
    from_gdf: gpd.GeoDataFrame,
    to_gdf: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    weight: str = "length_meter",
    unit_div: int = 1000,
) -> gpd.GeoDataFrame:
    """
    Compute the minimum distances from each point in the origin GeoDataFrame to the nearest 
    point in the destination GeoDataFrame using the road network graph.

    Parameters
    ----------
    from_gdf : gpd.GeoDataFrame
        GeoDataFrame containing origin points.
    to_gdf : gpd.GeoDataFrame
        GeoDataFrame containing destination points.
    graph : nx.MultiDiGraph
        Road network graph representing transport connections.
    weight : str, optional
        Edge attribute used for distance calculation (default is "length_meter").
    unit_div : int, optional
        Factor to convert distance into desired units (default is 1000 for km).

    Returns
    -------
    gpd.GeoDataFrame
        Series containing the minimum distances in the specified units.
    """
    if to_gdf is None or to_gdf.empty:
        return None
    return round(
        get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3
    )


def get_distance_from(
    point: gpd.GeoDataFrame,
    settlement_points: gpd.GeoDataFrame,
    area_polygons: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    local_crs: int,
) -> gpd.GeoDataFrame:
    """
    Calculate the median distance from settlement points to a reference point within specified area polygons.

    Parameters
    ----------
    point : gpd.GeoDataFrame
        GeoDataFrame containing the reference point to measure distances from.
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame containing settlement points.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing area polygons.
    graph : nx.MultiDiGraph
        Transport network graph.
    local_crs : int
        Coordinate reference system (CRS) to use.

    Returns
    -------
    gpd.GeoDataFrame
        GeoDataFrame with median distances to the reference point for each area polygon.
    """
    distances = calculate_distances(settlement_points.to_crs(local_crs), point.to_crs(local_crs), graph)
    settlement_points = settlement_points.copy()
    settlement_points["dist"] = distances
    res = gpd.sjoin(settlement_points, area_polygons, how="left", predicate="within")
    grouped_median = res.groupby("index_right").median(numeric_only=True)
    return grouped_median["dist"]


def get_distance_to_region_admin_center(
    region_admin_center: gpd.GeoDataFrame,
    settlement_points: gpd.GeoDataFrame,
    area_polygons: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
) -> gpd.GeoDataFrame:
    """
    Compute the median distance from settlements to the regional administrative center for each area polygon.

    Parameters
    ----------
    region_admin_center : gpd.GeoDataFrame
        GeoDataFrame containing the regional administrative center point.
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame containing settlement points.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing the area polygons.
    graph : nx.MultiDiGraph
        Transport network graph.

    Returns
    -------
    gpd.GeoDataFrame
        Updated area polygons with a new column for median distances to the regional administrative center.
    """
    local_crs = graph.graph["crs"]
    area_polygons = PolygonSchema(area_polygons).copy()
    region_admin_center = PointSchema(region_admin_center)
    settlement_points = PointSchema(settlement_points)

    area_polygons["distance_to_admin_center"] = get_distance_from(
        region_admin_center, settlement_points, area_polygons, graph, local_crs
    )
    return area_polygons.to_crs(4326)



def get_distance_to_federal_roads(
    settlement_points: gpd.GeoDataFrame, area_polygons: gpd.GeoDataFrame, graph: nx.MultiDiGraph
) -> gpd.GeoDataFrame:
    """
    Compute the median distance from settlements to federal roads for each area polygon.

    Parameters
    ----------
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame containing settlement points.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing the area polygons.
    graph : nx.MultiDiGraph
        Transport network graph.

    Returns
    -------
    gpd.GeoDataFrame
        Updated area polygons with a new column for median distances to federal roads.
    """
    area_polygons = PolygonSchema(area_polygons).copy()

    n = momepy.nx_to_gdf(graph)[0]
    local_crs = graph.graph["crs"]
    area_polygons["distance_to_federal_roads"] = get_distance_from(
        n[n.reg_1 == True], settlement_points, area_polygons, graph, local_crs
    )

    return area_polygons.to_crs(4326)


def get_connectivity(
    settlement_points: gpd.GeoDataFrame,
    area_polygons: gpd.GeoDataFrame,
    local_crs: int,
    graph: nx.MultiDiGraph = None,
    adj_mx: pd.DataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Calculate connectivity scores for each area polygon based on settlement points and a transport network.

    Parameters
    ----------
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame containing settlement points.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing area polygons.
    local_crs : int
        Coordinate reference system (CRS) to use.
    graph : nx.MultiDiGraph, optional
        Transport network graph (required if adjacency matrix is not provided).
    adj_mx : pd.DataFrame, optional
        Precomputed adjacency matrix.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with computed connectivity values.
    """
    if adj_mx is None:
        if adj_mx is None and graph is None:
            print("Either graph or adjacency matrix should be provided!")
            return
        adj_mx = get_adj_matrix_gdf_to_gdf(
            settlement_points.to_crs(local_crs),
            settlement_points.to_crs(local_crs),
            graph,
            weight="time_min",
            dtype=np.float64,
        )

    settlement_points_adj = PointSchema(settlement_points).copy()
    area_polygons = PolygonSchema(area_polygons).copy()
    settlement_points_adj["connectivity_min"] = adj_mx.median(axis=1)
    res = gpd.sjoin(settlement_points_adj, area_polygons, how="left", predicate="within")
    grouped_median = res.groupby("index_right").median(numeric_only=True)
    area_polygons["connectivity"] = grouped_median["connectivity_min"]
    return area_polygons.to_crs(4326)


def get_road_length(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the total road length within each area polygon.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Transport network graph.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing area polygons.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with an additional column indicating the total road length (in kilometers).
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    e = momepy.nx_to_gdf(graph)[1]

    grouped_length = (
        gpd.sjoin(e, area_polygons.to_crs(e.crs), how="left", predicate="within")
        .groupby("index_right")["length_meter"]
        .sum()
    )

    area_polygons["road_length"] = area_polygons.index.map(grouped_length).fillna(0) / 1000

    return area_polygons.to_crs(4326)


def get_road_density(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute road density for each area polygon.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        Transport network graph.
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame representing area polygons.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with calculated road density (road length per unit area).
    """
    area_polygons = PolygonSchema(area_polygons).to_crs(graph.graph["crs"]).copy()
    e = momepy.nx_to_gdf(graph)[1]

    grouped_length = (
        gpd.sjoin(e, area_polygons.to_crs(e.crs), how="left", predicate="within")
        .groupby("index_right")["length_meter"]
        .sum()
    )
    area_polygons["road_length"] = area_polygons.index.map(grouped_length).fillna(0) / 1000
    area_polygons["area"] = area_polygons.geometry.area / 1e6
    area_polygons["density"] = area_polygons["road_length"] / area_polygons["area"]
    area_polygons.drop(columns=["road_length", "area"], inplace=True)
    return area_polygons.to_crs(4326)


def get_reg_length(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute the total length of roads by regional classification within each area polygon.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The transport network graph.
    area_polygons : gpd.GeoDataFrame
        The polygons representing the areas of interest.

    Returns
    -------
    gpd.GeoDataFrame
        Updated area polygons with additional columns for road length by regional classification.
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    e = momepy.nx_to_gdf(graph)[1]
    for reg in [1, 2, 3]:
        roads = e[e["reg"] == reg]
        grouped_length = (
            gpd.sjoin(roads, area_polygons.to_crs(roads.crs), how="left", predicate="within")
            .groupby("index_right")["length_meter"]
            .sum()
        )

        area_polygons[f"length_reg_{reg}"] = area_polygons.index.map(grouped_length).fillna(0) / 1000
    return area_polygons.to_crs(4326)


def get_service_count(area_polygons: gpd.GeoDataFrame, service: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Count the number of service points located within each area polygon.

    Parameters
    ----------
    area_polygons : gpd.GeoDataFrame
        The polygons representing areas of interest.
    service : gpd.GeoDataFrame
        The GeoDataFrame containing service points.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with an added column for the number of services.
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    service_counts = gpd.GeoDataFrame(index=area_polygons.index)

    if service is not None and not service.empty:
        service = PointSchema(service)
        joined = gpd.sjoin(service.to_crs(area_polygons.crs), area_polygons, how="left", predicate="within")
        count_series = joined.groupby("index_right").size()
        service_counts["service_number"] = area_polygons.index.map(count_series).fillna(0).astype(int)
    else:
        service_counts["service_number"] = 0
    return area_polygons.join(service_counts).to_crs(4326)


def get_service_accessibility(
    settlement_points: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    area_polygons: gpd.GeoDataFrame,
    service: gpd.GeoDataFrame,
) -> gpd.GeoDataFrame:
    """
    Compute the median accessibility time from settlements to transport services within each area polygon.

    Parameters
    ----------
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame containing settlement points.
    graph : nx.MultiDiGraph
        The transport network graph.
    area_polygons : gpd.GeoDataFrame
        The polygons representing areas of interest.
    service : gpd.GeoDataFrame
        GeoDataFrame of transport service points.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with computed median accessibility times for each service.
    """
    settlement_points = PointSchema(settlement_points).copy()
    area_polygons = PolygonSchema(area_polygons).to_crs(graph.graph["crs"]).copy()
    settlement_points = settlement_points.to_crs(graph.graph["crs"]).copy()

    settlement_points = settlement_points.to_crs(graph.graph["crs"]).copy()
    accessibility_results = pd.DataFrame(index=area_polygons.index)  # Keep structure aligned with area_polygons

    if service is not None and not service.empty:
        service = PointSchema(service)
        settlement_points[f"service_accessibility_min"] = get_adj_matrix_gdf_to_gdf(
            settlement_points, service.to_crs(graph.graph["crs"]), graph, "time_min"
        ).median(axis=1)

        # Spatial join to assign settlement accessibility values to areas
        res = gpd.sjoin(settlement_points, area_polygons, how="left", predicate="within")
        grouped_median = res.groupby("index_right")["service_accessibility_min"].median()

        # Assign results back to area_polygons
        accessibility_results["service_accessibility"] = area_polygons.index.map(grouped_median)
    else:
        accessibility_results["service_accessibility"] = None

    # Merge computed accessibility results into area_polygons
    area_polygons = area_polygons.join(accessibility_results)

    return area_polygons.to_crs(4326)


def get_bus_routes_num(
    area_polygons: gpd.GeoDataFrame,
    bus_edges: gpd.GeoDataFrame = None,
    public_transport_graph: nx.MultiDiGraph = None,
    polygon_gdf: gpd.GeoDataFrame = None,
) -> gpd.GeoDataFrame:
    """
    Calculate the number of unique bus routes intersecting each area polygon.

    Parameters
    ----------
    area_polygons : gpd.GeoDataFrame
        The polygons representing areas of interest.
    bus_edges : gpd.GeoDataFrame, optional
        GeoDataFrame containing bus network edges with route attributes.
    public_transport_graph : nx.MultiDiGraph, optional
        The public transport graph.
    polygon_gdf : gpd.GeoDataFrame, optional
        A polygon GeoDataFrame used to extract the public transport graph.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with an additional column indicating the number of unique bus routes.
    """
    ## bus routes should have route parameter in edges
    area_polygons = PolygonSchema(area_polygons).copy()

    if bus_edges is None and public_transport_graph is None and polygon_gdf is not None:
        public_transport_graph = get_single_public_transport_graph(
            public_transport_type="bus", polygon=polygon_gdf.reset_index().geometry[0]
        )
        n, e = momepy.nx_to_gdf(public_transport_graph)
        bus_edges = e[e["type"] == "bus"]
    if bus_edges is None and public_transport_graph is not None:
        n, e = momepy.nx_to_gdf(public_transport_graph)
        bus_edges = e[e["type"] == "bus"]
    if bus_edges is not None:
        joined = gpd.sjoin(bus_edges, area_polygons.to_crs(bus_edges.crs), how="left", predicate="intersects")
        grouped_routes = joined.groupby("index_right")["route"].nunique()
        area_polygons["bus_routes_count"] = area_polygons.index.map(grouped_routes).fillna(0).astype(int)

        return area_polygons.to_crs(4326)
    else:
        print("No bus routes were found.")


def get_railway_length(
    railway_paths: gpd.GeoDataFrame, area_polygons: gpd.GeoDataFrame, local_crs: int
) -> gpd.GeoDataFrame:
    """
    Calculate the total railway length within each area polygon.

    Parameters
    ----------
    railway_paths : gpd.GeoDataFrame
        GeoDataFrame containing railway path geometries.
    area_polygons : gpd.GeoDataFrame
        The polygons representing areas of interest.
    local_crs : int
        The coordinate reference system to use.

    Returns
    -------
    gpd.GeoDataFrame
        Area polygons with an added column indicating the railway length (in kilometers).
    """
    area_polygons = PolygonSchema(area_polygons.to_crs(local_crs)).copy()

    railway_paths = railway_paths.to_crs(local_crs)
    railway_union = railway_paths.unary_union
    intersection = area_polygons.geometry.apply(lambda poly: railway_union.intersection(poly))
    area_polygons["railway_length_km"] = intersection.length / 1000

    return area_polygons.to_crs(4326)


def get_terr_service_accessibility(
    graph: nx.MultiDiGraph, territory_polygon: gpd.GeoDataFrame, service: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Compute service accessibility indicators for a given territory.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The transport network graph.
    territory_polygon : gpd.GeoDataFrame
        Polygon representing the territory of interest.
    service : gpd.GeoDataFrame
        GeoDataFrame containing service points.

    Returns
    -------
    gpd.GeoDataFrame
        Updated territory polygons with:
        - 'number_of_service': Count of services inside.
        - 'service_accessibility': Travel time to the nearest service.
    """
    # terr = districts_polygons.iloc[[6]].reset_index(drop=True).copy()
    terr = PolygonSchema(territory_polygon).to_crs(graph.graph["crs"]).copy()
    terr["geometry"] = terr.geometry.buffer(3000)
    terr_center = gpd.GeoDataFrame(geometry=terr.geometry.representative_point(), crs=terr.crs)

    if service is not None and not service.empty:
        service = service.to_crs(terr.crs).copy()

        # Count services inside each territory
        joined = gpd.sjoin(service, terr, how="left", predicate="within")
        count_series = joined.groupby("index_right").size()
        terr[f"number_of_service"] = terr.index.map(count_series).fillna(0).astype(int)

        # Set accessibility = 0 if service exists inside, otherwise compute travel time
        terr[f"service_accessibility"] = get_adj_matrix_gdf_to_gdf(terr_center, service, graph, "time_min").min()
        terr.loc[terr["number_of_service"] > 0, "service_accessibility"] = 0
        terr.loc[terr["number_of_service"].isna(), "service_accessibility"] = None
        terr = terr.drop(columns=["number_of_service"])

    else:
        # terr['number_of_service'] = 0
        terr["service_accessibility"] = None

    return terr.to_crs(4326)


def get_terr_service_count(
    territory_polygon: gpd.GeoDataFrame, service: gpd.GeoDataFrame, local_crs: int = 3856
) -> gpd.GeoDataFrame:
    """
    Count the number of service points within a given territory, considering a 3 km buffer.

    Parameters
    ----------
    territory_polygon : gpd.GeoDataFrame
        The polygon(s) representing areas of interest.
    service : gpd.GeoDataFrame
        GeoDataFrame containing service points.
    local_crs : int, optional
        The coordinate reference system (default is 3856).

    Returns
    -------
    gpd.GeoDataFrame
        Territory polygons with an added column indicating the number of services.
    """
    territory_polygon = PolygonSchema(territory_polygon).to_crs(local_crs).copy()
    territory_polygon["geometry"] = territory_polygon.geometry.buffer(3000)
    return get_service_count(territory_polygon, service).to_crs(4326)


def get_terr_road_density(graph: nx.MultiDiGraph, territory_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Compute the road density within a given territory.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The transport network graph.
    territory_polygon : gpd.GeoDataFrame
        The polygon(s) representing the territory.

    Returns
    -------
    gpd.GeoDataFrame
        Territory polygons with an added column for road density.
    """
    territory_polygon = PolygonSchema(territory_polygon.to_crs(graph.graph["crs"])).copy()
    territory_polygon["geometry"] = territory_polygon.geometry.buffer(3000)

    return get_road_density(graph, territory_polygon).to_crs(4326)


def get_terr_distance_to_region_admin_center(
    region_admin_center: gpd.GeoDataFrame, territory_polygon: gpd.GeoDataFrame, graph: nx.MultiDiGraph
) -> gpd.GeoDataFrame:
    """
    Compute the median distance from a territory center to the regional administrative center.

    Parameters
    ----------
    region_admin_center : gpd.GeoDataFrame
        The regional administrative center point.
    territory_polygon : gpd.GeoDataFrame
        The polygons representing the territory.
    graph : nx.MultiDiGraph
        The transport network graph.

    Returns
    -------
    gpd.GeoDataFrame
        Updated territory polygons with a new column for the median distance to the regional admin center.
    """
    local_crs = graph.graph["crs"]
    terr = PolygonSchema(territory_polygon).to_crs(local_crs).reset_index(drop=True).copy()
    terr["geometry"] = terr.geometry.buffer(3000)
    terr_center = gpd.GeoDataFrame(geometry=terr.geometry.representative_point(), crs=terr.crs).reset_index(drop=True)
    if len(terr_center) == 1:
        ts = pd.concat([terr_center, terr_center]).reset_index(drop=True)
    else:
        ts = terr_center

    terr["distance_to_admin_center"] = round(
        get_adj_matrix_gdf_to_gdf(region_admin_center, ts, graph, weight="length_meter", dtype=np.float64).min(axis=1)
        / 1000,
        3,
    )
    return terr.to_crs(4326)


def get_terr_nature_distance(
    territory: gpd.GeoDataFrame,
    nature_objects: gpd.GeoDataFrame,
    local_crs: int = 3857,
) -> gpd.GeoDataFrame:
    """
    Compute the number of nature objects within a territory and the distance to the nearest one.

    Parameters
    ----------
    territory : gpd.GeoDataFrame
        The area polygons where nature accessibility is calculated.
    nature_objects : gpd.GeoDataFrame
        GeoDataFrame containing nature objects (e.g., parks, reserves).
    local_crs : int, optional
        The coordinate reference system (default is 3857).

    Returns
    -------
    gpd.GeoDataFrame
        Updated `territory` with:
        - 'number_of_objects': Count of nature objects within.
        - 'objects_accessibility': Distance to the nearest nature object.
    """
    terr = PolygonSchema(territory).to_crs(local_crs).copy()

    nature_objects = nature_objects.to_crs(local_crs).copy()
    terr[f"number_of_objects"] = 0  # Initialize the column with 0s

    for i, row in terr.iterrows():
        if nature_objects.empty:
            terr.at[i, "objects_accessibility"] = None
        else:
            row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
            terr.at[i, "number_of_objects"] = len(gpd.overlay(nature_objects, row_temp, keep_geom_type=False))
            if terr.at[i, "number_of_objects"] > 0:
                terr.at[i, "objects_accessibility"] = 0.0
            else:
                terr.at[i, "objects_accessibility"] = round(
                    gpd.sjoin_nearest(row_temp, nature_objects, how="inner", distance_col="dist")["dist"].min() / 1000, 3
                )

    return terr.to_crs(4326)


def get_terr_nearest_centers(
    territory: gpd.GeoDataFrame,
    graph: nx.MultiDiGraph,
    districts: gpd.GeoDataFrame = None,
    centers_points: gpd.GeoDataFrame = None,
    local_crs: int = 3857,
) -> gpd.GeoDataFrame:
    """
    Compute distances from a territory to the nearest district and settlement centers.

    Parameters
    ----------
    territory : gpd.GeoDataFrame
        The polygons representing the territory.
    graph : nx.MultiDiGraph
        The transport network graph.
    districts : gpd.GeoDataFrame, optional
        GeoDataFrame of district boundaries.
    centers_points : gpd.GeoDataFrame, optional
        GeoDataFrame containing district/settlement centers.
    local_crs : int, optional
        The coordinate reference system (default is 3857).

    Returns
    -------
    gpd.GeoDataFrame
        Updated territory polygons with:
        - 'to_nearest_district_center_km': Distance to the nearest district center.
    """
    # Convert to consistent CRS
    territory = PolygonSchema(territory).to_crs(local_crs).copy()
    centers_points = PointSchema(centers_points).to_crs(local_crs).copy()
    districts = PolygonSchema(districts).to_crs(local_crs).copy()

    # Initialize result columns with None
    territory["to_nearest_center_km"] = None

    # Filter districts and service points that intersect with the territory
    if districts is not None:
        filtered_regions_terr = districts[districts.intersects(territory.unary_union)]

        # Filter district and settlement centers based on intersection with districts
        filtered_district_centers = (
            centers_points[centers_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
            if centers_points is not None
            else None
        )

        # Compute distances
        if filtered_district_centers is not None and not filtered_district_centers.empty:
            territory["to_nearest_center_km"] = calculate_distances(territory, filtered_district_centers, graph)

    return territory.to_crs(4326)
