from transport_frames.criteria import grade_territory
import os
import pytest
import geopandas as gpd
import pickle


data_path = "./tests/data"

@pytest.fixture
def weighted_frame():
    with open(os.path.join(data_path,"weighted_frame.pkl"), "rb") as f:
        weighted_frame = pickle.load(f)
    return weighted_frame


@pytest.fixture
def terr():
    s = gpd.read_file(os.path.join(data_path, "territory.geojson"))
    return s


def test_grade_range(weighted_frame, terr):
    graded_territories = grade_territory(weighted_frame, terr)
    """Test that grades are within the 0-5 range."""
    assert  graded_territories['grade'].between(0, 5).all(), "Grades should be between 0 and 5"

