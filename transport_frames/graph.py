""" Module for creating city network graph from polygon """

import warnings
from numbers import Real

import geopandas as gpd
import iduedu
import momepy
import networkx as nx
import osmnx as ox
from shapely import MultiPolygon, Polygon
from shapely.geometry import LineString

from transport_frames.utils.helper_funcs import BaseSchema, convert_geometry_from_wkt


warnings.simplefilter("ignore", UserWarning)


_CATEGORY_TO_REG = {"federal": 1, "regional": 2, "local": 3}


class PolygonSchema(BaseSchema):
    """
    Schema for validating polygons.

    Attributes
    ----------
    _geom_types : list
        List of allowed geometry types for the blocks, default is [shapely.Polygon]
    """
    _geom_types = [Polygon, MultiPolygon]


def get_graph(
    *,
    osm_id: int | None = None,
    territory: gpd.GeoDataFrame | None = None,
    buffer: int = 3000,
) -> nx.MultiDiGraph:
    """
    Collect a drive graph from OSM relation id or input territory.

    Parameters
    ----------
    osm_id : int | None, optional
        OSM relation id. If provided, territory is downloaded from OSM by id.
    territory : gpd.GeoDataFrame | None, optional
        Territory boundary as GeoDataFrame.
    local_crs : int | None, optional
        Local projection CRS. If None, estimated from territory via `estimate_utm_crs()`.
    buffer : int, optional
        Buffer distance (meters) used before graph download.

    Returns
    -------
    nx.MultiDiGraph
        Region network drive graph.
    """
    if osm_id is None and territory is None:
        raise ValueError("Provide either `osm_id` or `territory`.")
    if osm_id is not None:
        territory = ox.geocode_to_gdf(f"R{osm_id}", by_osmid=True)
    if not isinstance(territory, gpd.GeoDataFrame):
        raise TypeError("`territory` must be GeoDataFrame.")

    local_crs = territory.estimate_utm_crs()
    
    polygon_gdf = PolygonSchema(territory.to_crs(local_crs))
    polygon_geom = polygon_gdf.geometry.union_all()
    polygon_with_buf = gpd.GeoDataFrame([{"geometry": polygon_geom.buffer(buffer)}], crs=local_crs)
    polygon_geometry_with_buf = polygon_with_buf.to_crs(4326).geometry.union_all()

    G_drive = iduedu.get_drive_graph(
        territory=polygon_geometry_with_buf,
        add_road_category=True,
        osm_edge_tags=["maxspeed", "category", "ref"],
    )
    G_drive = _ensure_reg_attr(G_drive)
    G_drive = _crop_edges_by_polygon(G_drive, polygon_gdf)
    G_drive.graph["crs"] = local_crs
    G_drive.graph["approach"] = "primal"
    G_drive = classify_nodes(G_drive)

    return G_drive


def get_intermodal_graph(
    *,
    osm_id: int | None = None,
    territory: gpd.GeoDataFrame | None = None,
) -> nx.MultiDiGraph:
    """
    Collect an intermodal graph (public transport + walking) from OSM id or territory.

    Parameters
    ----------
    osm_id : int | None, optional
        OSM relation id. If provided, territory is downloaded from OSM by id.
    territory : gpd.GeoDataFrame | None, optional
        Territory boundary as GeoDataFrame.
    Returns
    -------
    nx.MultiDiGraph
        Intermodal graph.
    """
    if (osm_id is None) == (territory is None):
        raise ValueError("Provide exactly one of `osm_id` or `territory`.")

    if osm_id is not None:
        G_int = iduedu.get_intermodal_graph(osm_id=osm_id)
    else:
        G_int = iduedu.get_intermodal_graph(territory=territory)

    return G_int


def _category_to_reg(category) -> int | None:
    """
    Convert iduedu road category into legacy numeric reg code.
    """
    if category is None:
        return None

    if isinstance(category, (list, tuple, set)):
        regs = [_category_to_reg(value) for value in category]
        regs = [value for value in regs if value is not None]
        return min(regs) if regs else None

    if isinstance(category, str):
        return _CATEGORY_TO_REG.get(category.strip().lower())

    return None


def _ensure_reg_attr(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Ensure every edge has numeric `reg` attribute derived from `category` if needed.
    """
    for _, _, data in graph.edges(data=True):
        if data.get("reg") in {1, 2, 3}:
            continue

        mapped_reg = _category_to_reg(data.get("category"))
        if mapped_reg is not None:
            data["reg"] = mapped_reg

    return graph


def classify_nodes(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Assigns reg_status to nodes based on edge data.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The road network graph with classified edges

    Returns  
    --------
    nx.MultiDiGraph
        City network drive graph with classified nodes and edges
    """

    for node in graph.nodes:
        graph.nodes[node]["reg_1"] = False
        graph.nodes[node]["reg_2"] = False

    for u, v, data in graph.edges(data=True):
        if data.get("reg") == 1:
            graph.nodes[u]["reg_1"] = True
            graph.nodes[v]["reg_1"] = True
        elif data.get("reg") == 2:
            graph.nodes[u]["reg_2"] = True
            graph.nodes[v]["reg_2"] = True
    return graph


def _crop_edges_by_polygon(graph: nx.MultiDiGraph, polygon: Polygon) -> nx.MultiDiGraph:
    """
    Updates edge geometries based on intersections with the city boundary.

    Parameters
    ----------
    graph : nx.MultiDiGraph
        The city network graph
    polygon : Polygon
        The Polygon to crop edges with

    Returns  
    ---------
    nx.MultiDiGraph
        City network drive graph with cropped edges
    """
    edges = momepy.nx_to_gdf(graph)[1]

    city_transformed = polygon.to_crs(edges.crs)
    edges["intersections"] = edges["geometry"].intersection(city_transformed.unary_union)
    edges["geometry"] = edges["intersections"]

    edges.drop(columns=["intersections"], inplace=True)
    edges = edges.explode(index_parts=True)
    edges = edges[~edges["geometry"].is_empty]
    edges = edges[edges["geometry"].geom_type == "LineString"]

    nodes_coord = {}

    for _, row in edges.iterrows():
        start_node = row["node_start"]
        end_node = row["node_end"]
        if start_node not in nodes_coord:
            nodes_coord[start_node] = {
                "x": row["geometry"].coords[0][0],
                "y": row["geometry"].coords[0][1],
            }
        if end_node not in nodes_coord:
            nodes_coord[end_node] = {
                "x": row["geometry"].coords[-1][0],
                "y": row["geometry"].coords[-1][1],
            }

    graph = _create_graph(edges, nodes_coord)
    nx.set_node_attributes(graph, nodes_coord)
    graph = nx.convert_node_labels_to_integers(graph)
    graph = convert_geometry_from_wkt(graph)
    return graph



def _create_graph(edges: gpd.GeoDataFrame, nodes_coord: dict) -> nx.MultiDiGraph:
    """
    Create a graph based on edges and node coordinates.

    Parameters
    -----------
    edges : gpd.GeoDataFrame: 
        The edges with their attributes and geometries
    nodes_coord : dict 
        A dictionary containing node coordinates

    Returns
    --------
    nx.MultiDiGraph 
        The constructed graph
    """
    G = nx.MultiDiGraph()
    for _, edge in edges.iterrows():
        p1 = int(edge.node_start)
        p2 = int(edge.node_end)
        geom = (
            LineString(
                (
                    [
                        (nodes_coord[p1]["x"], nodes_coord[p1]["y"]),
                        (nodes_coord[p2]["x"], nodes_coord[p2]["y"]),
                    ]
                )
            )
            if not edge.geometry
            else edge.geometry
        )
        length = round(geom.length, 3)
        attrs = edge.to_dict()
        attrs.pop("node_start", None)
        attrs.pop("node_end", None)
        attrs.pop("intersections", None)
        attrs.pop("geometry", None)
        # Avoid passing MultiDiGraph edge key as kwarg by accident.
        attrs.pop("key", None)

        attrs["geometry"] = geom
        attrs["length_meter"] = length

        speed_mpm = attrs.get("speed_mpm")
        if isinstance(speed_mpm, Real) and speed_mpm > 0:
            attrs["time_min"] = round(length / float(speed_mpm), 3)

        G.add_edge(p1, p2, **attrs)
    return G
