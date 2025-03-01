import warnings

import geopandas as gpd
import iduedu
import momepy
import networkx as nx
from shapely import MultiPolygon, Polygon
from shapely.geometry import LineString

from transport_frames.utils.helper_funcs import BaseSchema, convert_geometry_from_wkt


warnings.simplefilter("ignore", UserWarning)


class PolygonSchema(BaseSchema):
    """
    Schema for validating polygons and multipolygons.

    Attributes:
    - _geom_types (list): List of allowed geometry types (Polygon, MultiPolygon).
    """

    _geom_types = [Polygon, MultiPolygon]


def get_graph_from_polygon(polygon: gpd.GeoDataFrame, local_crs: int):
    polygon = PolygonSchema(polygon.to_crs(local_crs))
    graph = from_polygon_iduedu(polygon, local_crs)
    graph = _crop_edges_by_polygon(graph, polygon)
    graph.graph["approach"] = "primal"
    graph.graph["crs"] = local_crs
    graph = classify_nodes(graph)
    return graph


def from_polygon_iduedu(polygon: gpd.GeoDataFrame, local_crs: int, buffer=3000):
    polygon_with_buf = gpd.GeoDataFrame([{"geometry": polygon.loc[0].geometry.buffer(buffer)}], crs=local_crs)
    polygon_geometry_with_buf = polygon_with_buf.to_crs(4326).geometry.unary_union
    graph = iduedu.get_drive_graph(
        polygon=polygon_geometry_with_buf, additional_edgedata=["highway", "maxspeed", "reg", "ref", "name"]
    )

    return graph


def classify_nodes(graph):
    """
    Classify nodes in the graph based on their edges reg status.
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


def _crop_edges_by_polygon(graph: nx.MultiDiGraph, polygon: Polygon):
    """
    Update edge geometries based on intersections with the city boundary.

    Parameters:
    edges (gpd.GeoDataFrame): The edges GeoDataFrame with existing geometries.
    polygon (gpd.GeoDataFrame): The polygon representing the city boundary.
    crs (int): The coordinate reference system to use.

    Returns:
    gpd.GeoDataFrame: Updated edges with geometries trimmed to the city boundary.
    """

    nodes, edges = momepy.nx_to_gdf(graph)

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


def _rewrite_nodes_after_cropping(edges, nodes_coord):
    """
    Update node coordinates based on new edge geometries.

    Parameters:
    edges (gpd.GeoDataFrame): The edges GeoDataFrame.
    nodes_coord (dict): A dictionary of node coordinates.

    Returns:
    dict: Updated dictionary of node coordinates.
    """
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
    return nodes_coord


def _create_graph(edges, nodes_coord):
    """
    Create a graph based on edges and node coordinates.

    Parameters:
    edges (gpd.GeoDataFrame): The edges with their attributes and geometries.
    nodes_coord (dict): A dictionary containing node coordinates.

    Returns:
    nx.MultiDiGraph: The constructed graph.
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
        G.add_edge(
            p1,
            p2,
            length_meter=length,
            time_min=edge.time_min,
            maxspeed=edge.maxspeed,
            geometry=str(geom),
            highway=edge.highway,
            ref=edge.ref,
            reg=edge.reg,
        )
    return G
