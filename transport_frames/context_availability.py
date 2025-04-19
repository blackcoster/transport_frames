import geopandas as gpd

def get_context_connectivity(settlement_points,context_polygons, adj_mx, area_polygons,local_crs):
    adj_mx = adj_mx.copy()
    settlement_points = settlement_points.to_crs(local_crs).copy()
    context_polygons = context_polygons.to_crs(local_crs).copy()
    area_polygons = area_polygons.to_crs(local_crs).copy()

    
    context_settlements_ids = list(gpd.overlay(settlement_points, 
                       context_polygons[['geometry']])['territory_id'])
    
    adj_mx.loc[context_settlements_ids, context_settlements_ids] = 0

    medians = adj_mx.loc[context_settlements_ids][:].transpose().median(axis=1)
    medians.name = 'median_connectivity'
    settlement_points_med = settlement_points.merge(medians, 'left', left_on='territory_id',right_index=True)
    pj = settlement_points_med.sjoin(area_polygons)
    poly_medians = pj.groupby('index_right')['median_connectivity'].median()
    fin = area_polygons.merge(poly_medians,'left',left_index=True,right_index=True)

    return fin

