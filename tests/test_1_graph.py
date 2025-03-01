import pytest
import osmnx as ox
import momepy
from transport_frames.graph import get_graph_from_polygon


@pytest.fixture
def polygon():
    geocode = 81993
    polygon_gdf = ox.geocode_to_gdf(f'R{geocode}', by_osmid=True)
    return polygon_gdf

@pytest.fixture
def local_crs():
    return 32637

@pytest.fixture
def graph(polygon, local_crs):
    return get_graph_from_polygon(polygon, local_crs)

def test_graph_crs(graph, local_crs):
    """Test if the CRS of the resulting graph matches the local CRS"""
    assert graph.graph["crs"] == local_crs, f"Graph CRS {graph.graph['crs']} does not match expected {local_crs}"

def test_nodes_inside_polygon(graph, polygon, local_crs):
    """Test if all nodes are inside the polygon"""
    nodes_gdf, _ = momepy.nx_to_gdf(graph)
    assert nodes_gdf.within(polygon.to_crs(local_crs).buffer(10).unary_union).all(), "Some nodes are outside the polygon"
