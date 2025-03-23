import shapely
import pygeoops

import networkx as nx
import pandas as pd
import geopandas as gpd
import momepy as mp

from shapely import unary_union, line_locate_point
from shapely.ops import split, nearest_points
from shapely.geometry import Point, LineString


def simplify_roads(roads_graph: nx.Graph, water_gdf=None) -> gpd.GeoDataFrame:

    _, roads_gdf = mp.nx_to_gdf(roads_graph)
    roads_buffer = roads_gdf.geometry.buffer(25, cap_style="round", join_style="mitre")
    roads_union = roads_buffer.union_all()

    if water_gdf is not None:

        water_gdf.to_crs(roads_gdf.crs, inplace=True)
        water_union = water_gdf.union_all()

        roads_diff = roads_union.difference(water_union)

        bridges = roads_gdf.union_all()
        bridges_diff = bridges.intersection(water_union)
        bridges_diff = bridges_diff.buffer(25, cap_style="round", join_style="mitre")

        result_roads = roads_diff.union(bridges_diff)
        centerline = pygeoops.centerline(result_roads, densify_distance=58, simplifytolerance=-0.15)
        streets_gdf = gpd.GeoDataFrame(geometry=[centerline], crs=roads_gdf.crs)
        streets_gdf = streets_gdf.explode("geometry")
        streets_gdf.reset_index(drop=True, inplace=True)

    else:

        centerline = pygeoops.centerline(roads_union, densify_distance=58, simplifytolerance=-0.15)
        streets_gdf = gpd.GeoDataFrame(geometry=[centerline], crs=roads_gdf.crs)
        streets_gdf = streets_gdf.explode("geometry")
        streets_gdf.reset_index(drop=True, inplace=True)

    return streets_gdf


def cut(line: LineString, distance: float) -> list[LineString]:
    # Cuts a line in two at a distance from its starting point
    if distance <= 0.0 or distance >= line.length:
        return [LineString(line)]
    coords = list(line.coords)
    for i, p in enumerate(coords):
        pd = line.project(Point(p))
        if pd == distance:
            return [LineString(coords[: i + 1]), LineString(coords[i:])]
        if pd > distance:
            cp = line.interpolate(distance)
            return [LineString(coords[:i] + [(cp.x, cp.y)]), LineString([(cp.x, cp.y)] + coords[i:])]


def _project_stop_on_road(road_geom: LineString, stop_geom: Point) -> list[LineString]:
    distance = line_locate_point(road_geom, stop_geom)
    splitted_road = cut(road_geom, distance)
    return splitted_road


def project_stops_on_roads(roads_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:

    roads_gdf = roads_gdf.copy()
    stops_gdf.to_crs(roads_gdf.crs, inplace=True)

    for stop_i, row in stops_gdf.iterrows():
        stop_gdf = stops_gdf.iloc[[stop_i]]
        sjoin = gpd.sjoin_nearest(stop_gdf, roads_gdf)
        roads_i = sjoin["index_right"].iloc[0]

        road_geom = roads_gdf.loc[roads_i].geometry
        stop_geom = row.geometry
        splitted_roads = _project_stop_on_road(road_geom, stop_geom)

        splitted_roads_gdf = gpd.GeoDataFrame(geometry=splitted_roads, crs=roads_gdf.crs)
        roads_gdf = roads_gdf.drop(roads_i)
        roads_gdf = pd.concat([roads_gdf, splitted_roads_gdf]).reset_index(drop=True)

    return roads_gdf


def roads_to_graph(roads_gdf: gpd.GeoDataFrame, stops_gdf: gpd.GeoDataFrame) -> nx.Graph:
    roads_graph = mp.gdf_to_nx(roads_gdf, multigraph=False, directed=False)
    nodes_gdf, _ = mp.nx_to_gdf(roads_graph)

    sjoin = stops_gdf.sjoin_nearest(nodes_gdf)
    for node in zip(sjoin["x"], sjoin["y"]):
        roads_graph.nodes[node]["is_stop"] = True

    return roads_graph
