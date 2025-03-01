import os
import pytest
from transport_frames.frame import get_frame
import momepy
import geopandas as gpd
import osmnx as ox
import pickle


data_path = "./tests/data"

@pytest.fixture
def graph():
    with open(os.path.join(data_path,"graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    return graph

@pytest.fixture
def admin_centers():
    adm = gpd.read_file(os.path.join(data_path, "region_points.geojson"))
    return adm

@pytest.fixture
def region_polygons():
    rp = gpd.read_file(os.path.join(data_path, "regions.geojson"))
    return rp

@pytest.fixture
def polygon():
    geocode = 81993
    polygon_gdf = ox.geocode_to_gdf(f'R{geocode}', by_osmid=True)
    return polygon_gdf


def test_frame_crs(graph, admin_centers, polygon, region_polygons):
    """Test if graph and frame crs are equal"""
    frame = get_frame(graph, admin_centers, polygon, region_polygons)
    assert frame.graph['crs'] == graph.graph['crs'], "Frame CRS does not match input graph CRS"
    
def test_frame_only_has_reg_1_and_reg_2(graph, admin_centers, polygon, region_polygons):
    """Test if frame has only reg 1,2"""
    frame = get_frame(graph, admin_centers, polygon, region_polygons)
    _, edges = momepy.nx_to_gdf(frame)

    assert "reg" in edges.columns, "No 'reg' column in edges"
    assert set(edges['reg'].unique()).issubset({1, 2}), "Frame contains roads other than reg_1 and reg_2"

def test_exits_near_boundary(graph, admin_centers, polygon, region_polygons):
    """Test if exits are near the polygon boundary"""
    frame = get_frame(graph, admin_centers, polygon, region_polygons)
    nodes, _ = momepy.nx_to_gdf(frame)

    city_boundary = polygon.to_crs(nodes.crs).boundary
    near_boundary = nodes.loc[nodes['exit'], 'geometry'].apply(lambda x: city_boundary.intersects(x.buffer(100)))

    assert near_boundary[0].all(), "Exit nodes are not near the boundary"