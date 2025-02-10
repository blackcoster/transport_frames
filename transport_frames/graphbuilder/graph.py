import geopandas as gpd
import networkx as nx
import osmnx as ox
from loguru import logger
from shapely import Polygon, MultiPolygon, Point
from shapely.geometry import LineString
import momepy
import pandas as pd
import numpy as np
from shapely.ops import unary_union
from transport_frames.utils.helper_funcs import buffer_and_transform_polygon, convert_geometry_from_wkt, _determine_ref_type
from transport_frames.graphbuilder.road_classifier import RoadClassifier
from transport_frames.models.graph_validation import GraphNode, GraphEdge, GraphMetadata, ClassifiedEdge
from transport_frames.models.polygon_validation import PolygonSchema
import warnings
import iduedu
import re
import pandera as pa
from pandera.typing import Series

warnings.simplefilter("ignore", UserWarning)

class Graph:
    """
    A class to represent and manipulate a graph of road networks.
    """

    def __init__(self, nx_graph: nx.MultiDiGraph, crs: int = 3857, polygon= None):
        self.graph = nx_graph
        self.crs = crs
        self.polygon = polygon
        self._prepare_attrs()
        self.validate_graph(self.graph)
        GraphMetadata(**self.graph.graph)
        self.classify_roads()

    def get_frame(self, regions: gpd.GeoDataFrame, polygon: gpd.GeoDataFrame, centers: gpd.GeoDataFrame = None,
                  max_distance: int = 3000, country_polygon: gpd.GeoDataFrame = ox.geocode_to_gdf('RUSSIA'),
                  restricted_terr: gpd.GeoDataFrame = None):
        """
        Generate a frame from the graph by filtering, weighing roads, and assigning city names.
        """
        regions = PolygonSchema(regions)
        polygon = PolygonSchema(polygon)
        if centers is not None:
            centers = PolygonSchema(centers)
        if restricted_terr is not None:
            restricted_terr = PolygonSchema(restricted_terr)
        country_polygon = PolygonSchema(country_polygon)
        
        for d in map(lambda e: e[2], self.graph.edges(data=True)):
            d = GraphEdge(**d).__dict__
        
        self.frame = self._filter_roads()
        self.n, self.e = momepy.nx_to_gdf(self.frame)
        self.n = self._mark_exits(self.n, polygon, regions, country_polygon)
        self.n, self.e, self.frame = self._weigh_roads(self.n, self.e, self.frame, restricted_terr)
        if centers is not None:
            self.frame = self._assign_city_names_to_nodes(centers, self.n, self.frame, max_distance=max_distance)
        return self.frame

    def _filter_roads(self):
        """Filter the graph to include only reg_1 and reg_2 roads."""
        edges_to_keep = [(u, v, k) for u, v, k, d in self.graph.edges(data=True, keys=True) if d.get("reg") in ([1, 2])]
        frame = self.graph.edge_subgraph(edges_to_keep).copy()
        for node, data in frame.nodes(data=True):
            data['nodeID'] = node
        return frame

    def _mark_exits(self, gdf_nodes, city_polygon, regions, country_polygon):
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

    def _weigh_roads(self, n, e, frame, restricted_terr_gdf):
        """Calculate and normalize the weights of roads based on exits."""
        e['weight'] = 0.0
        n['weight'] = 0.0
        exits = n[n['exit'] == 1]
        
        for i1, start_node in exits.iterrows():
            for i2, end_node in exits.iterrows():
                if i1 == i2:
                    continue
                if start_node.geometry.buffer(15000).intersects(end_node.geometry.buffer(15000)):
                    continue
                
                weight = 1  # Placeholder weight calculation
                try:
                    path = nx.astar_path(frame, i1, i2, weight='time_min')
                except nx.NetworkXNoPath:
                    continue
                
                for j in range(len(path) - 1):
                    n.loc[n['nodeID'] == path[j], 'weight'] += weight
                    e.loc[(e['node_start'] == path[j]) & (e['node_end'] == path[j + 1]), 'weight'] += weight
                n.loc[n['nodeID'] == path[j + 1], 'weight'] += weight
        
        min_weight, max_weight = e['weight'].min(), e['weight'].max()
        e['norm_weight'] = (e['weight'] - min_weight) / (max_weight - min_weight)
        return n, e, frame
