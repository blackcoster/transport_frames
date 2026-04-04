import geopandas as gpd
import pandas as pd


def get_context_connectivity(
    settlement_points: gpd.GeoDataFrame,
    context_polygons: gpd.GeoDataFrame,
    adj_mx: pd.DataFrame,
    area_polygons: gpd.GeoDataFrame,
    local_crs: int,
) -> gpd.GeoDataFrame:
    """
    Calculate context connectivity for area polygons based on settlement travel times.

    The function identifies settlements inside context polygons, sets travel time between
    those context settlements to zero in the adjacency matrix, computes median travel time
    from each settlement to context settlements, and aggregates medians by area polygons.

    Parameters
    ----------
    settlement_points : gpd.GeoDataFrame
        GeoDataFrame of settlement points. Must contain `territory_id` column used to
        match settlement ids with adjacency matrix index/columns.
    context_polygons : gpd.GeoDataFrame
        GeoDataFrame of context polygons.
    adj_mx : pd.DataFrame
        Settlement-to-settlement adjacency matrix (e.g., travel times), indexed and
        column-labeled by settlement ids (`territory_id`).
    area_polygons : gpd.GeoDataFrame
        GeoDataFrame of target polygons where context connectivity is summarized.
    local_crs : int
        Local projected CRS used for spatial operations.

    Returns
    -------
    gpd.GeoDataFrame
        `area_polygons` with additional `median_connectivity` column containing median
        context travel-time accessibility aggregated from settlements within each polygon.
    """
    adj_mx = adj_mx.copy()
    settlement_points = settlement_points.to_crs(local_crs).copy()
    context_polygons = context_polygons.to_crs(local_crs).copy()
    area_polygons = area_polygons.to_crs(local_crs).copy()

    context_settlements_ids = list(gpd.overlay(settlement_points, context_polygons[["geometry"]])["territory_id"])

    adj_mx.loc[context_settlements_ids, context_settlements_ids] = 0

    medians = adj_mx.loc[context_settlements_ids][:].transpose().median(axis=1)
    medians.name = "median_connectivity"
    settlement_points_med = settlement_points.merge(medians, "left", left_on="territory_id", right_index=True)
    pj = settlement_points_med.sjoin(area_polygons)
    poly_medians = pj.groupby("index_right")["median_connectivity"].median()
    fin = area_polygons.merge(poly_medians, "left", left_index=True, right_index=True)

    return fin
