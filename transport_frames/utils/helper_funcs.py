""" Some helper funcs """

import warnings

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import momepy
import networkx as nx
import shapely.wkt as wkt
from tqdm import TqdmWarning


warnings.simplefilter("ignore", TqdmWarning)
import geopandas as gpd
import pandera as pa
from pandera.typing import Index
from pandera.typing.geopandas import GeoSeries
from shapely import LineString, MultiLineString, MultiPoint, MultiPolygon, Point, Polygon


class BaseSchema(pa.DataFrameModel):
    """General class to validate gdfs on"""

    idx: Index[int] = pa.Field(unique=True)
    geometry: GeoSeries
    _geom_types = [Point, MultiPoint, Polygon, MultiPolygon, LineString, MultiLineString]

    class Config:
        strict = "filter"
        add_missing_columns = True

    @classmethod
    def to_gdf(cls):
        columns = cls.to_schema().columns.keys()
        return gpd.GeoDataFrame(data=[], columns=columns, crs=4326)

    @pa.check("geometry")
    @classmethod
    def check_geometry(cls, series):
        return series.map(lambda x: any([isinstance(x, geom_type) for geom_type in cls._geom_types]))


# перевод в геометрию
def convert_geometry_from_wkt(graph: nx.MultiDiGraph) -> nx.MultiDiGraph:

    """Convert the geometry in the graph to WKT format.

    Parameters:
    graph (MultiDiGraph): The input graph.

    Returns:
    nx.MultiDiGraph: The graph with converted geometry.
    """

    G = graph.copy()

    for _, _, data in G.edges(data=True):
        if isinstance(data.get("geometry"), str):
            geometry_wkt = wkt.loads(data["geometry"])
            data["geometry"] = geometry_wkt

    return G


def convert_list_attr_to_str(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Convert list attributes to string format for edges in a directed graph.

    Parameters:
        graph (MultiDiGraph): Directed graph.

    Returns:
        MultiDiGraph: Directed graph with attributes converted to string format.
    """
    graph = G.copy()
    for u, v, key, data in graph.edges(keys=True, data=True):
        for k, value in data.items():
            if isinstance(value, list):
                graph[u][v][key][k] = ",".join(map(str, value))
    return graph


def convert_list_attr_from_str(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Convert string attributes to list format for edges in a directed graph.

    Parameters:
        graph (MultiDiGraph): Directed graph.

    Returns:
        MultiDiGraph: Directed graph with attributes converted to list format.
    """
    graph = G.copy()
    for u, v, key, data in graph.edges(keys=True, data=True):
        for k, value in data.items():
            if isinstance(value, str) and "," in value and k != "geometry":
                graph[u][v][key][k] = list(map(str, value.split(",")))
    return graph


def buffer_and_transform_polygon(polygon: gpd.GeoDataFrame, crs: int = 3857):
    """Creating buffer around polygon and crs modification"""
    return polygon.to_crs(crs).geometry.buffer(3000).to_crs(4326).unary_union


def _determine_ref_type(ref: str) -> float:
    """
    Determine the reference type based on the reference list.

    Parameters:
    - ref (tuple): A tuple of reference types.

    Returns:
    - float: Determined reference type.
    """
    if "A" in ref:
        return 1.1
    elif "B" in ref:
        return 1.2
    elif "C" in ref:
        return 2.1
    elif "D" in ref:
        return 2.2
    elif "E" in ref:
        return 3.1
    elif "F" in ref:
        return 3.2
    elif "G" in ref:
        return 3.3
    return 3.3


def prepare_graph(graph_orig: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Prepare the graph for analysis by converting node names to integers and extract edge geometries from WKT format.

    Parameters:
    graph (networkx.MultiDiGraph): The input graph.

    Returns:
    networkx.MultiDiGraph: The prepared graph with node names as integers and geometries as WKT.
    """
    graph = nx.convert_node_labels_to_integers(graph_orig)
    for _, _, data in graph.edges(data=True):
        if isinstance(data.get("geometry"), str):
            data["geometry"] = wkt.loads(data["geometry"])

    return graph


def plot_graph(graph=None, edges=None):
    """
    Plots a graph's edges using GeoPandas and overlays it on a basemap.

    Parameters:
    - graph (gpd.GeoDataFrame): GeoDataFrame containing the graph's edges.

    Returns:
    - None: Displays the plotted graph.
    """

    # Convert to Web Mercator for correct basemap overlay
    if edges is None:
        nodes, edges = momepy.nx_to_gdf(graph)
        edges = edges.to_crs(epsg=3857).copy()

    # Initialize figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Set axis limits with 10% margin
    xmin, ymin, xmax, ymax = edges.total_bounds
    x_margin = (xmax - xmin) * 0.1 / 2
    y_margin = (ymax - ymin) * 0.1 / 2
    ax.set_xlim(xmin - x_margin, xmax + x_margin)
    ax.set_ylim(ymin - y_margin, ymax + y_margin)

    # Add basemap
    ctx.add_basemap(ax, source=ctx.providers.OpenStreetMap.Mapnik, crs=3857, alpha=1)

    # Plot edges
    edges.plot(ax=ax, color="blue", alpha=0.8)

    # Remove axis labels and frame
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

    # Show plot
    plt.show()
