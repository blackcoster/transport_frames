from iduedu import get_adj_matrix_gdf_to_gdf, get_single_public_transport_graph
import osmnx as ox
import pandas as pd
import geopandas as gpd
import numpy as np
import momepy
from transport_frames.models.schema import BaseSchema
import networkx as nx
import pandera as pa
from shapely.geometry import Point, Polygon, MultiPolygon
from pandera.typing import Series
from typing import Optional



class PolygonSchema(BaseSchema):
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Polygon, MultiPolygon]

class PointSchema(BaseSchema):
    _geom_types = [Point]


def calculate_distances(from_gdf: gpd.GeoDataFrame, 
                        to_gdf: gpd.GeoDataFrame, 
                        graph: nx.MultiDiGraph, 
                        weight: str ='length_meter', 
                        unit_div: int=1000) -> gpd.GeoDataFrame:
    """
    Calculate the minimum distances between two GeoDataFrames using a graph.
    
    Parameters:
    from_gdf (gpd.GeoDataFrame): The GeoDataFrame containing origin points.
    to_gdf (gpd.GeoDataFrame): The GeoDataFrame containing destination points.
    graph (nx.MultiDiGraph): The road network graph.
    weight (str): The edge attribute used for distance calculation. Default is 'length_meter'.
    unit_div (int): Factor to convert distance into desired units (default is 1000 for km).
    
    Returns:
    gpd.GeoDataFrame: Series containing the minimum distances in specified units.
    """
    if to_gdf is None or to_gdf.empty:
        return None
    return round(get_adj_matrix_gdf_to_gdf(from_gdf, to_gdf, graph, weight=weight, dtype=np.float64).min(axis=1) / unit_div, 3)

def get_distance_from(point: gpd.GeoDataFrame, settlement_points: gpd.GeoDataFrame, area_polygons: gpd.GeoDataFrame, 
                       graph: nx.MultiDiGraph, local_crs: int) -> gpd.GeoDataFrame:
    """
    Compute the median distance from settlement points to a specific point within given area polygons.
    
    Parameters:
    point (gpd.GeoDataFrame): The reference point to measure distances from.
    settlement_points (gpd.GeoDataFrame): The GeoDataFrame of settlement points.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    graph (nx.MultiDiGraph): The transport graph.
    local_crs (int): The coordinate reference system to use.
    
    Returns:
    gpd.GeoDataFrame: The median distance to the point within each area polygon.
    """    
    distances = calculate_distances(settlement_points.to_crs(local_crs),point.to_crs(local_crs),graph)
    settlement_points = settlement_points.copy()
    settlement_points['dist']=distances
    res = gpd.sjoin(settlement_points, area_polygons, how="left", predicate="within")
    grouped_median = res.groupby('index_right').median(numeric_only=True)   
    return grouped_median['dist']

def indicator_distance_to_region_admin_center(region_admin_center: gpd.GeoDataFrame, settlement_points: gpd.GeoDataFrame, 
                                              area_polygons: gpd.GeoDataFrame, graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Calculate the median distance from settlements to the regional administrative center within area polygons.
    
    Parameters:
    region_admin_center (gpd.GeoDataFrame): The regional administrative center point.
    settlement_points (gpd.GeoDataFrame): The settlement points.
    area_polygons (gpd.GeoDataFrame): The polygons representing regions.
    graph (nx.MultiDiGraph): The transport network graph.
    
    Returns:
    gpd.GeoDataFrame: Updated area polygons with a new column for the median distance to the regional admin center.
    """
    local_crs = graph.graph['crs']
    area_polygons = PolygonSchema(area_polygons).copy()
    region_admin_center = PointSchema(region_admin_center)
    settlement_points = PointSchema(settlement_points)
    
    area_polygons['distance_to_admin_center'] = get_distance_from(region_admin_center ,settlement_points,area_polygons, graph, local_crs )
    return area_polygons

def indicator_distance_to_federal_roads(settlement_points: gpd.GeoDataFrame, area_polygons: gpd.GeoDataFrame, 
                                        graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
    Calculate the median distance from settlements to federal roads within area polygons.
    
    Parameters:
    settlement_points (gpd.GeoDataFrame): The settlement points.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    graph (nx.MultiDiGraph): The transport network graph.
    local_crs (int): The coordinate reference system.
    
    Returns:
    gpd.GeoDataFrame: Updated area polygons with a new column for the median distance to federal roads.
    """
    area_polygons = PolygonSchema(area_polygons).copy()

    n = momepy.nx_to_gdf(graph)[0]
    local_crs = graph.graph['crs']
    area_polygons['distance_to_federal_roads'] = get_distance_from(n[n.reg_1 == True] ,settlement_points,area_polygons, graph,local_crs )

    return area_polygons


def indicator_connectivity_drive(settlement_points: gpd.GeoDataFrame, drive_graph: nx.MultiDiGraph, 
                                 area_polygons: gpd.GeoDataFrame, adj_mx: pd.DataFrame = None) -> gpd.GeoDataFrame:
    """
    Compute connectivity indicator for drive-based transportation.
    
    Parameters:
    settlement_points (gpd.GeoDataFrame): The settlement points.
    drive_graph (nx.MultiDiGraph): The driving network graph.
    area_polygons (gpd.GeoDataFrame): The area polygons.
    adj_mx (pd.DataFrame, optional): Precomputed adjacency matrix. Defaults to None.
    
    Returns:
    gpd.GeoDataFrame: Area polygons with computed connectivity indicator.
    """
    return indicator_connectivity(settlement_points,drive_graph,area_polygons, adj_mx)

def indicator_connectivity_public_transport(settlement_points: gpd.GeoDataFrame, pt_graph: nx.MultiDiGraph, 
                                            area_polygons: gpd.GeoDataFrame, adj_mx: pd.DataFrame = None) -> gpd.GeoDataFrame:
    """
    Compute connectivity indicator for public transportation.
    
    Parameters:
    settlement_points (gpd.GeoDataFrame): The settlement points.
    pt_graph (nx.MultiDiGraph): The public transport network graph.
    area_polygons (gpd.GeoDataFrame): The area polygons.
    adj_mx (pd.DataFrame, optional): Precomputed adjacency matrix. Defaults to None.
    
    Returns:
    gpd.GeoDataFrame: Area polygons with computed connectivity indicator.
    """
    return indicator_connectivity(settlement_points,pt_graph,area_polygons, adj_mx)

def indicator_connectivity(settlement_points: gpd.GeoDataFrame, 
                           graph: nx.MultiDiGraph, 
                           area_polygons: gpd.GeoDataFrame, 
                           adj_mx: pd.DataFrame = None) -> gpd.GeoDataFrame:
    """
    Calculate connectivity within area polygons based on settlement points and a transport graph.

    Parameters:
    settlement_points (gpd.GeoDataFrame): GeoDataFrame of settlement points.
    graph (nx.MultiDiGraph): The transport graph.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    adj_mx (pd.DataFrame, optional): Precomputed adjacency matrix. Defaults to None.

    Returns:
    gpd.GeoDataFrame: Area polygons with connectivity values.
    """
    if adj_mx is None:
        adj_mx = get_adj_matrix_gdf_to_gdf(settlement_points.to_crs(graph.graph['crs']),
                                       settlement_points.to_crs(graph.graph['crs']),
                                       graph,
                                       weight='time_min',
                                       dtype=np.float64)
        
    settlement_points_adj = PointSchema(settlement_points).copy()
    area_polygons = PolygonSchema(area_polygons).copy()
    settlement_points_adj['connectivity_min']=adj_mx.median(axis=1)
    res = gpd.sjoin(settlement_points_adj, area_polygons, how="left", predicate="within")
    grouped_median = res.groupby('index_right').median(numeric_only=True)   
    area_polygons['connectivity'] = grouped_median['connectivity_min']
    return area_polygons

def indicator_road_length(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the total road length within each area polygon.

    Parameters:
    graph (nx.MultiDiGraph): The transport graph.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.

    Returns:
    gpd.GeoDataFrame: Area polygons with road length values.
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    n, e = momepy.nx_to_gdf(graph)
    
    grouped_length = (
        gpd.sjoin(e, area_polygons.to_crs(e.crs), how="left", predicate="within")
        .groupby('index_right')['length_meter']
        .sum()
    )
    
    area_polygons["road_length"] = area_polygons.index.map(grouped_length).fillna(0)/1000
    
    return area_polygons

def indicator_road_density(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the road density within each area polygon.

    Parameters:
    graph (nx.MultiDiGraph): The transport graph.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.

    Returns:
    gpd.GeoDataFrame: Area polygons with road density values.
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    n, e = momepy.nx_to_gdf(graph)
    
    grouped_length = (
        gpd.sjoin(e, area_polygons.to_crs(e.crs), how="left", predicate="within")
        .groupby('index_right')['length_meter']
        .sum()
    )
    area_polygons["road_length"] = area_polygons.index.map(grouped_length).fillna(0)/1000
    area_polygons['area'] = area_polygons.geometry.area / 1e6
    area_polygons['density'] = area_polygons['road_length']/area_polygons['area']
    area_polygons.drop(columns=['road_length','area'],inplace=True)
    return area_polygons

def indicator_reg_length(graph: nx.MultiDiGraph, area_polygons: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the length of roads by regional classification within each area polygon.

    Parameters:
    graph (nx.MultiDiGraph): The transport graph.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.

    Returns:
    gpd.GeoDataFrame: Area polygons with road length for each regional category.
    """
    area_polygons = PolygonSchema(area_polygons).copy()
    n,e = momepy.nx_to_gdf(graph)
    for reg in [1,2,3]:
        roads = e[e['reg']==reg] 
        grouped_length = (
                gpd.sjoin(roads, area_polygons.to_crs(roads.crs), how="left", predicate="within")
                .groupby('index_right')['length_meter']
                .sum()
            )
            
        area_polygons[f"length_reg_{reg}"] = area_polygons.index.map(grouped_length).fillna(0)/1000
    return area_polygons

def indicator_service_count(area_polygons: gpd.GeoDataFrame,
                            fuel_stations: gpd.GeoDataFrame = None, 
                            railway_stations: gpd.GeoDataFrame = None, 
                            ports: gpd.GeoDataFrame = None,
                            local_aerodrome: gpd.GeoDataFrame = None, 
                            international_aerodrome: gpd.GeoDataFrame = None,
                            bus_stops: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Count the number of services within each area polygon.

    Parameters:
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    fuel_stations (gpd.GeoDataFrame, optional): Fuel station locations.
    railway_stations (gpd.GeoDataFrame, optional): Railway station locations.
    ports (gpd.GeoDataFrame, optional): Port locations.
    local_aerodrome (gpd.GeoDataFrame, optional): Local aerodrome locations.
    international_aerodrome (gpd.GeoDataFrame, optional): International aerodrome locations.
    bus_stops (gpd.GeoDataFrame, optional): Bus stop locations.

    Returns:
    gpd.GeoDataFrame: Area polygons with service counts.
    """
    
    area_polygons = PolygonSchema(area_polygons).copy()

    service_list = [railway_stations, fuel_stations, ports, local_aerodrome, international_aerodrome, bus_stops]
    service_names = ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome', 'bus_stops']

    service_counts = gpd.GeoDataFrame(index=area_polygons.index)  

    for service, name in zip(service_list, service_names):
        if service is not None and not service.empty:
            service = PointSchema(service)
            joined = gpd.sjoin(service.to_crs(area_polygons.crs), area_polygons, how="left", predicate='within')
            count_series = joined.groupby('index_right').size()
            service_counts[f'{name}_number'] = area_polygons.index.map(count_series).fillna(0).astype(int)
        else:
            service_counts[f'{name}_number'] = 0
    return area_polygons.join(service_counts)

def indicator_service_accessibility(settlement_points: gpd.GeoDataFrame, 
                                    graph: nx.MultiDiGraph, 
                                    area_polygons: gpd.GeoDataFrame,
                                    fuel_stations: gpd.GeoDataFrame = None, 
                                    railway_stations: gpd.GeoDataFrame = None, 
                                    ports: gpd.GeoDataFrame = None,
                                    local_aerodrome: gpd.GeoDataFrame = None, 
                                    international_aerodrome: gpd.GeoDataFrame = None,
                                    bus_stops: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Compute median accessibility time to various transport services for each area polygon.

    Parameters:
    settlement_points (gpd.GeoDataFrame): GeoDataFrame containing settlement points.
    graph (nx.MultiDiGraph): The transport graph.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    fuel_stations (gpd.GeoDataFrame, optional): GeoDataFrame of fuel stations.
    railway_stations (gpd.GeoDataFrame, optional): GeoDataFrame of railway stations.
    ports (gpd.GeoDataFrame, optional): GeoDataFrame of ports.
    local_aerodrome (gpd.GeoDataFrame, optional): GeoDataFrame of local aerodromes (Point).
    international_aerodrome (gpd.GeoDataFrame, optional): GeoDataFrame of international aerodromes (Point).
    bus_stops (gpd.GeoDataFrame, optional): GeoDataFrame of bus stops.

    Returns:
    gpd.GeoDataFrame: Area polygons with computed median accessibility times for each service.
    """
    
    settlement_points = PointSchema(settlement_points).copy()
    area_polygons = PolygonSchema(area_polygons).to_crs(graph.graph['crs']).copy()
    settlement_points = settlement_points.to_crs(graph.graph['crs']).copy()

    service_list = [railway_stations, fuel_stations, ports, local_aerodrome, international_aerodrome, bus_stops]
    service_names = ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome', 'bus_stops']
    settlement_points = settlement_points.to_crs(graph.graph['crs']).copy()
    accessibility_results = pd.DataFrame(index=area_polygons.index)  # Keep structure aligned with area_polygons

    for service, name in zip(service_list, service_names):
        if service is not None and not service.empty:
            service = PointSchema(service)
            settlement_points[f'{name}_accessibility_min'] = get_adj_matrix_gdf_to_gdf(
                settlement_points, service.to_crs(graph.graph['crs']), graph, 'time_min'
            ).median(axis=1)

            # Spatial join to assign settlement accessibility values to areas
            res = gpd.sjoin(settlement_points, area_polygons, how="left", predicate="within")
            grouped_median = res.groupby('index_right')[f'{name}_accessibility_min'].median()

            # Assign results back to area_polygons
            accessibility_results[name + "_accessibility"] = area_polygons.index.map(grouped_median)
        else:
            accessibility_results[name + "_accessibility"] = None
    # Merge computed accessibility results into area_polygons
    area_polygons = area_polygons.join(accessibility_results)

    return area_polygons
        

def indicator_bus_routes(area_polygons: gpd.GeoDataFrame, 
                         bus_edges: gpd.GeoDataFrame = None, 
                         public_transport_graph: nx.MultiDiGraph = None, 
                         polygon_gdf: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
    """
    Calculate the number of unique bus routes intersecting each area polygon.

    Parameters:
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    bus_edges (gpd.GeoDataFrame, optional): GeoDataFrame containing bus network edges with route attributes.
    public_transport_graph (nx.MultiDiGraph, optional): The public transport graph.
    polygon_gdf (gpd.GeoDataFrame, optional): A polygon GeoDataFrame to extract public transport graph.

    Returns:
    gpd.GeoDataFrame: Area polygons with the number of unique bus routes.
    """

    ## bus routes should have route parameter in edges
    area_polygons = PolygonSchema(area_polygons).copy()

    if bus_edges is None and public_transport_graph is None and polygon_gdf is not None:
        public_transport_graph = get_single_public_transport_graph(public_transport_type='bus', polygon=polygon_gdf.reset_index().geometry[0])
        n,e = momepy.nx_to_gdf(public_transport_graph)
        bus_edges = e[e['type']=='bus']
    if bus_edges is None and public_transport_graph is not None:
        n,e = momepy.nx_to_gdf(public_transport_graph)
        bus_edges = e[e['type']=='bus']
    if bus_edges is not None:
        joined = gpd.sjoin(bus_edges, area_polygons.to_crs(bus_edges.crs), how="left", predicate="intersects")
        grouped_routes = joined.groupby('index_right')['route'].nunique()
        area_polygons["bus_routes_count"] = area_polygons.index.map(grouped_routes).fillna(0).astype(int)

        return area_polygons
    else:
        print('No bus routes were found.')


def indicator_railway_length(railway_paths: gpd.GeoDataFrame, 
                             area_polygons: gpd.GeoDataFrame, 
                             local_crs: int) -> gpd.GeoDataFrame:
    """
    Calculate the total railway length within each area polygon.

    Parameters:
    railway_paths (gpd.GeoDataFrame): Railway paths as a GeoDataFrame.
    area_polygons (gpd.GeoDataFrame): The polygons representing areas of interest.
    local_crs (int): The coordinate reference system.

    Returns:
    gpd.GeoDataFrame: Area polygons with railway length values.
    """
    area_polygons = PolygonSchema(area_polygons.to_crs(local_crs)).copy()
    
    railway_paths = railway_paths.to_crs(local_crs)
    railway_union = railway_paths.unary_union
    intersection = area_polygons.geometry.apply(lambda poly: railway_union.intersection(poly))
    area_polygons["railway_length_km"] = intersection.length /1000

    return area_polygons


def indicator_terr_service_accessibility(
    graph: nx.MultiDiGraph, 
    territory_polygon: gpd.GeoDataFrame,
    fuel_stations: Optional[gpd.GeoDataFrame] = None, 
    railway_stations: Optional[gpd.GeoDataFrame] = None, 
    ports: Optional[gpd.GeoDataFrame] = None,
    local_aerodrome: Optional[gpd.GeoDataFrame] = None, 
    international_aerodrome: Optional[gpd.GeoDataFrame] = None,
    bus_stops: Optional[gpd.GeoDataFrame] = None
) -> gpd.GeoDataFrame:
    """
    Computes service accessibility indicators for a given territory polygon.

    Parameters:
    -----------
    graph : nx.MultiDiGraph
        A networkx MultiDiGraph representing the transport network.
    territory_polygon : gpd.GeoDataFrame
        A GeoDataFrame containing the polygons representing the territories of interest.
    fuel_stations : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of fuel stations.
    railway_stations : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of railway stations.
    ports : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of ports.
    local_aerodrome : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of local aerodromes.
    international_aerodrome : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of international aerodromes.
    bus_stops : Optional[gpd.GeoDataFrame], default=None
        A GeoDataFrame containing locations of bus stops.

    Returns:
    --------
    gpd.GeoDataFrame
        A GeoDataFrame containing the territories with added columns:
        - 'number_of_{service}': Number of service points inside each territory.
        - '{service}_accessibility': 0 if the service is inside, otherwise the minimum travel time.
    """

    terr = PolygonSchema(territory_polygon).to_crs(graph.graph['crs']).copy()
    terr['geometry'] = terr.geometry.buffer(3000)
    terr_center = gpd.GeoDataFrame(geometry=terr.geometry.representative_point(), crs = terr.crs)
    service_list = [railway_stations, fuel_stations, ports, local_aerodrome, international_aerodrome, bus_stops]
    service_names = ['railway_stations', 'fuel_stations', 'ports', 'local_aerodrome', 'international_aerodrome', 'bus_stops']
    for service, name in zip(service_list, service_names):
        if service is not None and not service.empty:
            service = service.to_crs(terr.crs).copy()

            # Count services inside each territory
            joined = gpd.sjoin(service, terr, how="left", predicate='within')
            count_series = joined.groupby('index_right').size()
            terr[f'number_of_{name}'] = terr.index.map(count_series).fillna(0).astype(int)

            # Set accessibility = 0 if service exists inside, otherwise compute travel time
            terr[f'{name}_accessibility'] = get_adj_matrix_gdf_to_gdf(
                    terr_center, service, graph, 'time_min'
                ).min()
            terr.loc[terr[f'number_of_{name}'] > 0, f'{name}_accessibility'] = 0
            terr.loc[terr[f'number_of_{name}'].isna(), f'{name}_accessibility'] = None
        else:
            terr[f'number_of_{name}'] = 0
            terr[f'{name}_accessibility'] = None

    return terr


def indicator_terr_service_count(territory_polygon: gpd.GeoDataFrame,
                            fuel_stations: gpd.GeoDataFrame = None, 
                            railway_stations: gpd.GeoDataFrame = None, 
                            ports: gpd.GeoDataFrame = None,
                            local_aerodrome: gpd.GeoDataFrame = None, 
                            international_aerodrome: gpd.GeoDataFrame = None,
                            bus_stops: gpd.GeoDataFrame = None,
                            local_crs: int = 3856) -> gpd.GeoDataFrame:
    """
    Count the number of services within the territory polygon with 3 km buffer.

    Parameters:
    territory_polygon (gpd.GeoDataFrame): The polygon(s) representing areas of interest.
    fuel_stations (gpd.GeoDataFrame, optional): Fuel station locations.
    railway_stations (gpd.GeoDataFrame, optional): Railway station locations.
    ports (gpd.GeoDataFrame, optional): Port locations.
    local_aerodrome (gpd.GeoDataFrame, optional): Local aerodrome locations.
    international_aerodrome (gpd.GeoDataFrame, optional): International aerodrome locations.
    bus_stops (gpd.GeoDataFrame, optional): Bus stop locations.
    local_crs (int, optional)L local crs projection.

    Returns:
    gpd.GeoDataFrame: Territory polygon(s) with service counts.
    """

    territory_polygon = PolygonSchema(territory_polygon).to_crs(local_crs).copy()
    territory_polygon['geometry'] = territory_polygon.geometry.buffer(3000)
    return indicator_service_count(territory_polygon,fuel_stations,railway_stations,ports,local_aerodrome,international_aerodrome,bus_stops)


def indicator_terr_road_density(graph: nx.MultiDiGraph, territory_polygon: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """
    Calculate the road density within each area polygon.

    Parameters:
    graph (nx.MultiDiGraph): The transport graph.
    territory_polygon (gpd.GeoDataFrame): The polygon(s) representing territory.

    Returns:
    gpd.GeoDataFrame: Area polygons with road density values.
    """

    territory_polygon = PolygonSchema(territory_polygon.to_crs(graph.graph['crs'])).copy()
    territory_polygon['geometry'] = territory_polygon.geometry.buffer(3000)

    return indicator_road_density(graph,territory_polygon)


def indicator_terr_distance_to_region_admin_center(region_admin_center: gpd.GeoDataFrame, 
                                                   territory_polygon: gpd.GeoDataFrame, 
                                                   graph: nx.MultiDiGraph) -> gpd.GeoDataFrame:
    """
     Calculate the median distance from center of the territory to the regional administrative center.
    
    Parameters:
    region_admin_center (gpd.GeoDataFrame): The regional administrative center point.
    territory_polygon (gpd.GeoDataFrame): The polygons representing territory.
    graph (nx.MultiDiGraph): The transport network graph.
    local_crs (int): The coordinate reference system.
    
    Returns:
    gpd.GeoDataFrame: Updated area polygons with a new column for the median distance to the regional admin center.
    """

    local_crs = graph.graph['crs']
    terr = PolygonSchema(territory_polygon).to_crs(local_crs).reset_index(drop=True).copy()
    terr['geometry'] = terr.geometry.buffer(3000)
    terr_center = gpd.GeoDataFrame(geometry=terr.geometry.representative_point(), crs = terr.crs).reset_index(drop=True)
    if len(terr_center)==1:
        ts = pd.concat([terr_center, terr_center]).reset_index(drop=True)
    else:
        ts = terr_center
        
    terr['distance_to_admin_center'] = round(get_adj_matrix_gdf_to_gdf(region_admin_center, ts, graph, weight='length_meter', dtype=np.float64).min(axis=1) / 1000, 3)
    return terr


def indicator_terr_nature_distance(territory: gpd.GeoDataFrame, 
                                   water_objects: gpd.GeoDataFrame = None, 
                                   nature_reserve: gpd.GeoDataFrame = None, 
                                   local_crs: int = 3857) -> gpd.GeoDataFrame:
    """
    Compute the number of services and minimum accessibility distance for each polygon in `territory`
    without iteration, using only GeoPandas.

    Parameters:
    territory (gpd.GeoDataFrame): The area polygons where service accessibility is calculated.
    water_objects (gpd.GeoDataFrame, optional): Water body locations.
    nature_reserve (gpd.GeoDataFrame, optional): Nature reserve locations.
    local_crs (int): The coordinate reference system.

    Returns:
    gpd.GeoDataFrame: Updated `territory` with added service accessibility columns.
    """

    terr = PolygonSchema(territory).to_crs(local_crs).copy()

    for service, name in [(water_objects, 'water_objects'), (nature_reserve, 'nature_reserve')]:
        service = service.to_crs(local_crs).copy()
        terr[f'number_of_{name}'] = 0  # Initialize the column with 0s
        
        for i, row in terr.iterrows():
            if service.empty:
                terr.at[i, f'{name}_accessibility_min'] = None
            else: 
                row_temp = gpd.GeoDataFrame(index=[i], geometry=[row.geometry], crs=local_crs)
                terr.at[i, f'number_of_{name}'] = len(gpd.overlay(service, row_temp))
                if terr.at[i, f'number_of_{name}'] > 0 :
                    terr.at[i, f'{name}_accessibility'] = 0.0
                else:
                    terr.at[i, f'{name}_accessibility'] = round(gpd.sjoin_nearest(row_temp, service, how='inner', distance_col='dist')['dist'].min()/1000, 3)
    

    return terr


def indicator_terr_nearest_centers(
    territory: gpd.GeoDataFrame, 
    districts: gpd.GeoDataFrame = None, 
    district_points: gpd.GeoDataFrame = None, 
    settlement_points: gpd.GeoDataFrame = None, 
    local_crs: int = 3857
) -> gpd.GeoDataFrame:
    """
    Computes distances to nearest district and settlement centers and road density for a given territory.

    Parameters:
    ----------
    territory : gpd.GeoDataFrame
        The area polygons where distances and road density are calculated.
    districts : gpd.GeoDataFrame, optional
        District polygons for intersection filtering.
    district_points : gpd.GeoDataFrame, optional
        Points representing district centers.
    settlement_points : gpd.GeoDataFrame, optional
        Points representing settlements.

    local_crs : int, default=3857
        The coordinate reference system.

    Returns:
    -------
    gpd.GeoDataFrame
        Updated `territory` with:
        - 'to_nearest_district_center_km': Distance to nearest district center.
        - 'to_nearest_settlement_km': Distance to nearest settlement.
    """

    # Convert to consistent CRS
    territory = PolygonSchema(territory).to_crs(local_crs).copy()
    district_points = PointSchema(district_points).to_crs(local_crs).copy()
    settlement_points = PointSchema(settlement_points).to_crs(local_crs).copy()
    districts = PolygonSchema(districts).to_crs(local_crs).copy()

    # Initialize result columns with None
    territory['to_nearest_district_center_km'] = None
    territory['to_nearest_settlement_km'] = None

    # Filter districts and service points that intersect with the territory
    if districts is not None:
        filtered_regions_terr = districts[districts.intersects(territory.unary_union)]

        # Filter district and settlement centers based on intersection with districts
        filtered_district_centers = (
            district_points[district_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
            if district_points is not None else None
        )
        filtered_settlement_centers = (
            settlement_points[settlement_points.buffer(0.1).intersects(filtered_regions_terr.unary_union)]
            if settlement_points is not None else None
        )

        # Compute distances
        if filtered_district_centers is not None and not filtered_district_centers.empty:
            territory['to_nearest_district_center_km'] = calculate_distances(territory, filtered_district_centers,g.graph)

        if filtered_settlement_centers is not None and not filtered_settlement_centers.empty:
            territory['to_nearest_settlement_km'] = calculate_distances(territory, filtered_settlement_centers,g.graph)


    return territory