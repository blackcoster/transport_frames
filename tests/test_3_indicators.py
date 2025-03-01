from transport_frames.indicators import get_terr_service_accessibility, get_terr_service_count
import os
import pytest
import geopandas as gpd
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
def service():
    s = gpd.read_file(os.path.join(data_path, "local_aerodrome.geojson"))
    return s

@pytest.fixture
def terr():
    s = gpd.read_file(os.path.join(data_path, "territory.geojson"))
    return s



def test_service_accessibility_logic(graph,terr,service,local_crs):
    """ Test if number of services > 0, accessibility should be 0"""
    service_accessibility = get_terr_service_accessibility(graph, terr, service)
    service_count = get_terr_service_count(terr, service,local_crs)

    mask = service_count["service_number"] > 0  # Rows where service exists
    assert (service_accessibility.loc[mask, "service_accessibility"] == 0).all()

