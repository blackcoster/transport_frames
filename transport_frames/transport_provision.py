"""Module for transport provision calculation and visualization"""
import geopandas as gpd
import pandas as pd
from shapely.ops import nearest_points
import networkx as nx
import momepy
import matplotlib.pyplot as plt
import numpy as np
import json


def _get_sjoin_right_index_col(sjoin_result: gpd.GeoDataFrame) -> str:
    """
    Return right-side index column name from GeoPandas spatial join result.

    Depending on GeoPandas version and existing column names, it can be
    `index_right` or a suffixed variant such as `index_right0`.
    """
    if "index_right" in sjoin_result.columns:
        return "index_right"

    right_col = next((col for col in sjoin_result.columns if col.startswith("index_right")), None)
    if right_col is None:
        raise KeyError(
            "Could not find right-index column in spatial join result. "
            f"Available columns: {list(sjoin_result.columns)}"
        )
    return right_col


def calculate_transport_provision(
    territory_gdf: gpd.GeoDataFrame, 
    roads: gpd.GeoDataFrame | nx.Graph | nx.DiGraph | nx.MultiDiGraph, 
    railways_gdf: gpd.GeoDataFrame, 
    airport: gpd.GeoDataFrame, 
    towns_gdf: gpd.GeoDataFrame
) -> gpd.GeoDataFrame:
    """
    Calculate the transport infrastructure provision index for a given territory.

    Parameters
    ----------
    territory_gdf : gpd.GeoDataFrame
        GeoDataFrame representing the target territory.
    roads : gpd.GeoDataFrame | nx.Graph | nx.DiGraph | nx.MultiDiGraph
        Road network, either as a GeoDataFrame or a NetworkX graph.
    railways_gdf : gpd.GeoDataFrame
        GeoDataFrame containing railway network data.
    airport : gpd.GeoDataFrame
        GeoDataFrame with airport locations.
    towns_gdf : gpd.GeoDataFrame
        GeoDataFrame containing urban area boundaries.

    Returns
    -------
    gpd.GeoDataFrame
        Updated GeoDataFrame with calculated transport provision metrics, including:
        - 'area_km2': Territory area in square kilometers.
        - 'road_length_km': Total road length within the territory (km).
        - 'rail_length_km': Total railway length within the territory (km).
        - 'airport_distance_km': Distance to the nearest airport (km).
        - 'town_area_km2': Total urban area within the territory (km²).
        - 'transport_provision_index': Composite index measuring transport provision.
    """
    territory_gdf = territory_gdf.copy()
    railways_gdf = railways_gdf.copy()
    towns_gdf = towns_gdf.copy()
    airport = airport.copy()

    if isinstance(roads, nx.Graph) or isinstance(roads, nx.DiGraph) or isinstance(roads, nx.MultiDiGraph):
        road_gdf = momepy.nx_to_gdf(roads)[1]
    elif isinstance(roads, gpd.GeoDataFrame):
        road_gdf = roads.copy()
    else:
        raise TypeError("`roads` must be GeoDataFrame or networkx graph.")

    road_gdf['road_length_km'] = road_gdf['geometry'].length / 1000
    railways_gdf['road_length_km'] = railways_gdf['geometry'].length / 1000

    territory_gdf['area_km2'] = territory_gdf.geometry.area / 1e6 

    road_sjoin = gpd.sjoin(road_gdf, territory_gdf, predicate="intersects")
    rail_sjoin = gpd.sjoin(railways_gdf, territory_gdf, predicate="intersects")
    road_right_col = _get_sjoin_right_index_col(road_sjoin)
    rail_right_col = _get_sjoin_right_index_col(rail_sjoin)

    road_sjoin["_len_m"] = road_sjoin.geometry.length
    rail_sjoin["_len_m"] = rail_sjoin.geometry.length
    road_length = road_sjoin.groupby(road_right_col)["_len_m"].sum() / 2000  # км
    rail_length = rail_sjoin.groupby(rail_right_col)["_len_m"].sum() / 2000  # км
    

    territory_gdf['road_length_km'] = territory_gdf.index.map(road_length).fillna(0)
    territory_gdf['rail_length_km'] = territory_gdf.index.map(rail_length).fillna(0)


    towns_sjoin = gpd.sjoin(towns_gdf, territory_gdf, predicate="intersects")
    towns_right_col = _get_sjoin_right_index_col(towns_sjoin)
    towns_sjoin["_area_m2"] = towns_sjoin.geometry.area
    town_area = towns_sjoin.groupby(towns_right_col)["_area_m2"].sum() / 1e6  # км²
    territory_gdf['town_area_km2'] = territory_gdf.index.map(town_area).fillna(0)

    

    if not airport.empty:
        airport_union = airport.geometry.union_all()
        territory_gdf['airport_distance_km'] = territory_gdf.geometry.centroid.apply(
            lambda centroid: centroid.distance(nearest_points(centroid, airport_union)[1]) / 1000
        )
    else:
        territory_gdf['airport_distance_km'] = float('inf')
    

    territory_gdf['transport_provision_index'] = (
        0.4 * (territory_gdf['rail_length_km'] / territory_gdf['area_km2']) +
        0.3 * (territory_gdf['road_length_km'] / territory_gdf['area_km2']) +
        0.2 * (1 / territory_gdf['airport_distance_km'].replace(0, 1)) +
        0.1 * (1 - (territory_gdf['town_area_km2'] / territory_gdf['area_km2']))
    )
    
    return territory_gdf[['geometry', 'area_km2', 'road_length_km', 'rail_length_km', 'airport_distance_km', 'town_area_km2', 'transport_provision_index']]



def visualize_transport_provision(
    result: gpd.GeoDataFrame, 
    road_gdf: gpd.GeoDataFrame
) -> None:
    """
    Visualize the transport infrastructure provision index and road network.

    Parameters
    ----------
    result : gpd.GeoDataFrame
        GeoDataFrame containing calculated transport provision metrics.
    road_gdf : gpd.GeoDataFrame
        GeoDataFrame containing road network data.

    Returns
    -------
    None
        Displays a map visualization of transport provision.
    """
    fig, ax = plt.subplots(figsize=(50, 20))
    

    vmin, vmax = np.percentile(result['transport_provision_index'], [1, 90])


    result.plot(column='transport_provision_index', cmap='Blues', linewidth=0.5, 
                edgecolor='black', legend=True, ax=ax, vmin=vmin, vmax=vmax)

    roads_to_plot = road_gdf.copy()
    if not roads_to_plot.empty:
        reg_values = None

        if "properties" in roads_to_plot.columns:
            def _extract_reg(value):
                if isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        return None
                if isinstance(value, dict):
                    return value.get("reg")
                return None

            reg_values = roads_to_plot["properties"].apply(_extract_reg)
        elif "reg" in roads_to_plot.columns:
            reg_values = roads_to_plot["reg"]
        else:
            reg_values = pd.Series(index=roads_to_plot.index, dtype=float)


        roads_to_plot["width"] = reg_values.apply(lambda x: 1.5 if x == 1 else 0.6)


        roads_to_plot.plot(ax=ax, color='black', linewidth=roads_to_plot['width'], alpha=0.7)
    

    ax.set_axis_off()
    

    plt.title("Transport provision index", fontsize=14)
    plt.show()

