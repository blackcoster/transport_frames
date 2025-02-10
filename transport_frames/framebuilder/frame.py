import geopandas as gpd
import networkx as nx
import osmnx as ox
from loguru import logger
from shapely.geometry import Polygon, LineString
import momepy
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from transport_frames.utils.helper_funcs import _determine_ref_type
from transport_frames.models.graph_validation import GraphEdge, ClassifiedEdge
from transport_frames.models.polygon_validation import PolygonSchema, CentersSchema
import warnings

warnings.simplefilter("ignore", UserWarning)

def get_frame(graph, regions: gpd.GeoDataFrame, polygon: gpd.GeoDataFrame, centers: gpd.GeoDataFrame = None,
              max_distance: int = 3000, country_polygon: gpd.GeoDataFrame = ox.geocode_to_gdf('RUSSIA'),
              restricted_terr: gpd.GeoDataFrame = None):
    """
    Generate a frame from the given graph by filtering roads, weighing roads, and assigning city names.
    """
    regions = PolygonSchema(regions)
    polygon = PolygonSchema(polygon)
    if centers is not None:
        centers = CentersSchema(centers)
    if restricted_terr is not None:
        restricted_terr = PolygonSchema(restricted_terr)
    country_polygon = PolygonSchema(country_polygon)

    for d in map(lambda e: e[2], graph.graph.edges(data=True)):
        d = GraphEdge(**d).__dict__

    frame = _filter_roads(graph)
    n, e = momepy.nx_to_gdf(frame)
    if centers is not None:
        frame = _assign_city_names_to_nodes(centers, n, frame, max_distance=max_distance, local_crs=graph.crs)
    return frame

def _filter_roads(graph):
    """Filter the graph to include only reg_1 and reg_2 roads."""
    edges_to_keep = [(u, v, k) for u, v, k, d in graph.graph.edges(data=True, keys=True) if d.get("reg") in ([1, 2])]
    frame = graph.graph.edge_subgraph(edges_to_keep).copy()
    for node, data in frame.nodes(data=True):
        data['nodeID'] = node
    return frame

def mark_exits(gdf_nodes, city_polygon, regions, country_polygon):
    """Assign the 'exit' attribute to nodes based on city boundaries."""
    city_boundary = city_polygon.to_crs(gdf_nodes.crs).unary_union.boundary
    gdf_nodes['exit'] = gdf_nodes.geometry.apply(lambda point: city_boundary.intersects(point.buffer(0.1)))
    if gdf_nodes['exit'].sum() == 0:
        print('No region exits found. Try a larger polygon.')
    
    exits = gdf_nodes[gdf_nodes.exit]
    country_boundary = country_polygon.to_crs(exits.crs).unary_union.boundary
    exits['exit_country'] = exits.geometry.apply(lambda point: country_boundary.intersects(point.buffer(0.1)))
    gdf_nodes = gdf_nodes.assign(exit_country=exits['exit_country'].fillna(False))
    return gdf_nodes



def _assign_city_names_to_nodes(points, nodes, graph, max_distance=3000, local_crs=3857):
    """Assign city names to nodes in the graph based on proximity to city centers."""
    nodes.to_crs(local_crs, inplace=True)
    points.to_crs(local_crs, inplace=True)
    projected = gpd.sjoin_nearest(points, nodes, how="left", distance_col="distance", max_distance=max_distance)
    projected = projected.to_crs(graph.graph['crs'])
    
    missed_cities = set()
    for idx, node_id in enumerate(projected['nodeID'].values):
        city_name = projected.loc[idx, 'name']
        for _, d in graph.nodes(data=True):
            if d.get('nodeID') == node_id:
                d['city_name'] = city_name
            else:
                missed_cities.add(city_name)
    
    # if len(missed_cities) == len(projected):
    #     print('No cities could be assigned to nodes.')
    return graph



# functions for weighting the roads


def _mark_ref_type(
        n: gpd.GeoDataFrame,
        e: gpd.GeoDataFrame,
        frame: nx.MultiDiGraph
    ) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]:
        """
        Mark reference types for nodes in the road network based on the nearest reference edges.

        This method assigns a reference value and its type to nodes in the network based on their 
        proximity to edges that have reference attributes. It updates the nodes GeoDataFrame 
        with the reference values and types for exits.

        Parameters:
        - n (gpd.GeoDataFrame): GeoDataFrame containing nodes of the road network, which 
                                may include exit nodes to be marked with reference types.
        - e (gpd.GeoDataFrame): GeoDataFrame containing edges of the road network, which 
                                includes reference attributes used to determine node reference types.
        - frame (nx.MultiDiGraph): The road network graph where nodes represent intersections 
                                    or exits.

        Returns:
        - tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, nx.MultiDiGraph]: A tuple containing:
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
        return n,e,frame


def _get_weight(start: float, end: float, exit: bool)-> float:
        """
        Calculate the weight based on the type of start and end references and exit status.

        Parameters:
        start (float): Reference type of the start node.
        end (float): Reference type of the end node.
        exit (int): Exit status (1 if exit, else 0).

        Returns:
        float: Calculated weight based on the provided matrix.
        """
        dict = {1.1: 0, 1.2: 1, 1.3: 2, 2.1: 3, 2.2: 4, 2.3: 5, 0.0: 6, 0.5: 7}
        if exit == 1:
            matrix = [
                [0.12, 0.12, 0.12, 0.12, 0.12, 0.12,0.00001, 0.05],  # 2.1.1
                [0.10, 0.10, 0.10, 0.10, 0.10, 0.10,0.00001, 0.05],  # 2.1.2
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08,0.00001, 0.05],  # 2.1.3
                [0.07, 0.07, 0.07, 0.07, 0.07, 0.07,0.00001, 0.05],  # 2.2.1
                [0.06, 0.06, 0.06, 0.06, 0.06, 0.06,0.00001, 0.05],  # 2.2.2
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,0.00001, 0.05],  # 2.2.3
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02,0.00001, 0.05],  # 2.2.3
                [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05]            ]
        else:

            matrix = [
                [0.08, 0.08, 0.08, 0.08, 0.08, 0.08,0.00001, 0.05],  # 2.1.1
                [0.07, 0.07, 0.07, 0.07, 0.07, 0.07,0.00001, 0.05],  # 2.1.2
                [0.06, 0.06, 0.06, 0.06, 0.06, 0.06,0.00001, 0.05],  # 2.1.3
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05,0.00001, 0.05],  # 2.2.1
                [0.04, 0.04, 0.04, 0.04, 0.04, 0.04,0.00001, 0.05],  # 2.2.2
                [0.02, 0.02, 0.02, 0.02, 0.02, 0.02,0.00001, 0.05],  # 2.2.3
                [0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001, 0.00001],
                [0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.00001, 0.05]
            ]
        return matrix[dict[end]][dict[start]]

def weigh_roads(
        n: gpd.GeoDataFrame,
        e: gpd.GeoDataFrame,
        frame: nx.MultiDiGraph,
        restricted_terr_gdf: gpd.GeoDataFrame
    ) -> gpd.GeoDataFrame:
        """
        Calculate and normalize the weights of roads in a road network based on the proximity of exits.

        This method assigns weights to the road segments (edges) in the network based on their 
        connections to exits and the types of regions they traverse. It normalizes the weights 
        for further analysis.

        Parameters:
        - n (gpd.GeoDataFrame): GeoDataFrame containing nodes of the road network, where each 
                                node represents an intersection or exit.
        - e (gpd.GeoDataFrame): GeoDataFrame containing edges of the road network, where each 
                                edge represents a road segment with 'time_min' as a weight attribute.
        - frame (nx.MultiDiGraph): The road network graph where nodes represent intersections 
                                    or exits, and edges represent road segments.
        - restricted_terr (gpd.GeoDataFrame): GeoDataFrame containing restricted areas that may 
                                                affect road weights.

        Returns:
        - gpd.GeoDataFrame: A tuple containing two GeoDataFrames (nodes and edges) with 
                            updated weights and normalized weights for further analysis.
        """
        
        n,e,frame = _mark_ref_type(n, e, frame) 
        if restricted_terr_gdf is not None:
            country_exits = n[n['exit_country'] == True].copy()
            
            # Преобразуем CRS для совместимости
            restricted_terr_gdf = restricted_terr_gdf.to_crs(country_exits.crs)
            
            # Для каждой страны из restricted_terr_gdf применяем логику
            for _, row in restricted_terr_gdf.iterrows():
                border_transformed = row['geometry']
                buffer_area = border_transformed.buffer(300)
                mask = country_exits.geometry.apply(lambda x: x.intersects(buffer_area))
                
                # Применяем метку (mark) к соответствующим участкам
                n.loc[mask[mask==True].index, 'ref_type'] = row['mark']

        e["weight"] = 0.0
        n["weight"] = 0.0
        exits = n[n["exit"] == 1]
        for i1, start_node in exits.iterrows():
            for i2, end_node in exits.iterrows():
                if i1 == i2:
                    continue
                if (
                    pd.notna(start_node["border_region"])
                    and start_node["border_region"] == end_node["border_region"]
                ):
                    continue
                if start_node.geometry.buffer(15000).intersects(
                    end_node.geometry.buffer(15000)
                ) and (
                    pd.isna(start_node["exit_country"]) == pd.isna(end_node["exit_country"])
                ):
                    continue
                if start_node["exit_country"] == 1 and end_node["exit_country"] == 1:
                    continue

                weight = _get_weight(
                    start_node["ref_type"], end_node["ref_type"], end_node["exit_country"]
                )

                try:
                    path = nx.astar_path(frame, i1, i2, weight='time_min')
                except nx.NetworkXNoPath:
                    continue

                for j in range(len(path) - 1):
                    n.loc[(n["nodeID"] == path[j]), "weight"] += weight
                    e.loc[
                        (e["node_start"] == path[j]) & (e["node_end"] == path[j + 1]),
                        "weight",
                    ] += weight
                n.loc[(n["nodeID"] == path[j + 1]), "weight"] += weight

        n['weight'] = round(n.weight,3)
        min_weight = e['weight'].min()
        max_weight = e['weight'].max()
        e['norm_weight'] = (e['weight'] - min_weight) / (max_weight - min_weight)
        # n.drop(columns=['ref','border_region'], inplace=True)
        # n.drop(columns=['ref','ref_type','border_region'], inplace=True)
        # e.drop(columns=['ref','highway','maxspeed'], inplace=True)
        # for u, v, key, data in frame.edges(keys=True, data=True):  
        for i,(e1,e2,k,data) in enumerate(frame.edges(data=True,keys=True)):
            if 'ref' in data:
                del data['ref']
            if 'highway' in data:
                del data['highway']
            if 'maxspeed' in data:
                del data['maxspeed']
            data['weight'] = e.iloc[[i]]['weight'][i]
            data['norm_weight'] = e.iloc[[i]]['norm_weight'][i]


        for i, (node, data) in enumerate(frame.nodes(data=True)):
            data['exit'] = n.iloc[i]["exit"]
            data['exit_country'] = n.iloc[i]["exit_country"]
            data["weight"] = n.iloc[i]["weight"]
            data['ref_type']  = n.iloc[i]["ref_type"]

        return n,e,frame