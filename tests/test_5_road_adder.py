from transport_frames.road_adder import add_roads
import os
import pytest
import pickle


data_path = "./tests/data"

@pytest.fixture
def local_crs():
    return 32637

@pytest.fixture
def graph():
    with open(os.path.join(data_path,"graph.pkl"), "rb") as f:
        graph = pickle.load(f)
    return graph

@pytest.fixture
def new_roads():
    with open(os.path.join(data_path,"new_roads.pkl"), "rb") as f:
        nr = pickle.load(f)
    return nr

def test_increased_edges_number(graph, new_roads, local_crs):
    bigger_graph = add_roads(graph, new_roads,local_crs)
    assert len(bigger_graph.edges) > len(graph.edges)



