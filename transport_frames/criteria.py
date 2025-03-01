import warnings

import geopandas as gpd
import momepy
import networkx as nx
import numpy as np
import pandas as pd
import pandera as pa
from pandera.typing import Series
from shapely.geometry import MultiPolygon, Point, Polygon
from tqdm import TqdmWarning

from transport_frames.utils.helper_funcs import BaseSchema


warnings.simplefilter("ignore", TqdmWarning)


class PolygonSchema(BaseSchema):
    name: Series[str] = pa.Field(nullable=True)
    _geom_types = [Polygon, MultiPolygon]


class PointSchema(BaseSchema):
    _geom_types = [Point]


def grade_territory(frame: nx.MultiDiGraph, gdf_poly: gpd.GeoDataFrame, include_priority: bool = True):
    """
    Grades territories based on their distances to reg1, reg2 nodes,edges and train stations.

    Parameters:
        gdf_poly (GeoDataFrame): A GeoDataFrame containing the polygons of the territories to be graded.
        frame
    (networkx.MultiDiGraph): A MultiDiGraph representing the transportation network.

    Returns:
        GeoDataFrame: A GeoDataFrame containing the graded territories with added 'grade' column.
    """

    nodes, edges = momepy.nx_to_gdf(frame, points=True, lines=True, spatial_weights=False)
    gdf_poly = PolygonSchema(gdf_poly)
    poly = gdf_poly.copy().to_crs(nodes.crs)

    reg1_points = nodes[nodes["reg_1"] == 1]
    reg2_points = nodes[nodes["reg_2"] == 1]
    priority_reg1_points = nodes[
        (nodes["weight"] > np.percentile(nodes[nodes["weight"] != 0]["weight"], 60)) & (nodes["reg_1"] == 1)
    ]
    priority_reg2_points = nodes[
        (nodes["weight"] > np.percentile(nodes[nodes["weight"] != 0]["weight"], 60)) & (nodes["reg_2"] == 1)
    ]

    def _get_nearest_dist(terr, dist_to):
        terr = terr.copy()
        result = gpd.sjoin_nearest(terr, dist_to, distance_col="dist")
        result = result.loc[result.groupby(result.index)["dist"].idxmin()]
        result = result[~result.index.duplicated(keep="first")]
        return result["dist"]

    poly["dist_to_reg1"] = _get_nearest_dist(poly, reg1_points)
    poly["dist_to_reg2"] = _get_nearest_dist(poly, reg2_points)
    poly["dist_to_edge"] = _get_nearest_dist(poly, edges)
    poly["dist_to_priority_reg1"] = _get_nearest_dist(poly, priority_reg1_points)
    poly["dist_to_priority_reg2"] = _get_nearest_dist(poly, priority_reg2_points)

    poly["grade"] = poly.apply(_grade_polygon, axis=1, args=(include_priority,))
    output = poly[["name", "geometry", "grade"]].copy()
    return output.to_crs(4326)


def _grade_polygon(row: gpd.GeoDataFrame, include_priority: bool = True) -> float:
    """
    Determines the grade of a territory based on its distance to features.

    Parameters:
        row (Series): A pandas Series representing a single row of a GeoDataFrame.

    Returns:
        float: The grade of the territory.
    """
    dist_to_reg1 = row["dist_to_reg1"]
    dist_to_reg2 = row["dist_to_reg2"]
    dist_to_edge = row["dist_to_edge"]
    dist_to_priority1 = row["dist_to_priority_reg1"]
    dist_to_priority2 = row["dist_to_priority_reg2"]

    # below numbers measured in thousands are representes in meters eg 5_000 meters ie 5km
    if include_priority and dist_to_priority1 < 5000:
        grade = 5
    elif include_priority and dist_to_priority1 < 10000 and dist_to_priority2 < 5000 or dist_to_reg1 < 5000:
        grade = 4.5
    elif dist_to_reg1 < 10000 and dist_to_reg2 < 5000:
        grade = 4.0
    elif include_priority and dist_to_priority1 < 100000 and dist_to_priority2 < 5000:
        grade = 3.5
    elif dist_to_reg1 < 100000 and dist_to_reg2 < 5000:
        grade = 3.0
    elif dist_to_reg1 > 100000 and dist_to_reg2 < 5000:
        grade = 2.0
    elif dist_to_reg2 > 5000 and dist_to_reg1 > 100000 and dist_to_edge < 5000:
        grade = 1.0
    else:
        grade = 0.0

    return grade


MAX_DISTANCE = 15000


def find_median(city_points: gpd.GeoDataFrame, adj_mx: pd.DataFrame) -> gpd.GeoDataFrame:
    """
    Find the median correspondence time from one city to all others.

    Parameters:
    city_points (geopandas.GeoDataFrame): GeoDataFrame of city points.
    adj_mx (pandas.DataFrame): Adjacency matrix representing distances.

    Returns:
    geopandas.GeoDataFrame: GeoDataFrame of points with the 'to_service' column updated to median values.
    """
    points = city_points.copy()
    medians = []
    for index, row in adj_mx.iterrows():
        median = np.median(row[row.index != index])
        medians.append(median / 60)  # convert to hours
    points["to_service"] = medians
    return points


def weight_territory(
    territories: gpd.GeoDataFrame,
    railway_stops: gpd.GeoDataFrame,
    bus_stops: gpd.GeoDataFrame,
    ferry_stops: gpd.GeoDataFrame,
    airports: gpd.GeoDataFrame,
    local_crs: int,
) -> gpd.GeoDataFrame:
    """
    Calculate the weights for each territory based on the distances to various types of stops.

    Parameters:
    territories (gpd.GeoDataFrame): GeoDataFrame containing territory geometries.
    railway_stops (gpd.GeoDataFrame): GeoDataFrame containing railway stop geometries.
    bus_stops (gpd.GeoDataFrame): GeoDataFrame containing bus stop geometries.
    ferry_stops (gpd.GeoDataFrame): GeoDataFrame containing ferry stop geometries.
    airports (gpd.GeoDataFrame): GeoDataFrame containing airport geometries.

    Returns:
    gpd.GeoDataFrame: Updated territories with calculated weights.
    """

    territories = territories.to_crs(local_crs)

    # Calculate nearest distances and weights for each type of point

    services = [railway_stops, bus_stops, ferry_stops, airports]
    services_names = ["r_stops", "b_stops", "ferry", "aero"]
    weights = [0.35, 0.35, 0.2, 0.1]

    territories["weight"] = 0.0

    for service, name, weight in zip(services, services_names, weights):

        if service is None or service.empty:
            territories[f"weight_{name}"] = 0.0
        else:
            service = PointSchema(service).to_crs(local_crs).copy()
            territories_with_service = _get_nearest_distances(territories, service, f"distance_to_{name}")
            territories[f"weight_{name}"] = territories_with_service[f"distance_to_{name}"].apply(
                lambda x: weight if x <= MAX_DISTANCE else 0.0
            )

        territories["weight"] += territories[f"weight_{name}"]

    return territories


def _get_nearest_distances(
    territories: gpd.GeoDataFrame, stops: gpd.GeoDataFrame, distance_col: str
) -> gpd.GeoDataFrame:
    """
    Get the nearest distances between territories and stops.

    Parameters:
    territories (gpd.GeoDataFrame): GeoDataFrame containing territory geometries.
    stops (gpd.GeoDataFrame): GeoDataFrame containing stop geometries.
    distance_col (str): Column name to store the distance values.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame of territories with nearest distances to stops.
    """
    nearest = territories.sjoin_nearest(stops, distance_col=distance_col)
    nearest_reset = nearest.reset_index()
    min_idx = nearest_reset.groupby("geometry")[distance_col].idxmin()
    nearest_min = nearest_reset.loc[min_idx]
    nearest_min = nearest_min.sort_values("index")
    res = nearest_min.reset_index(drop=True).drop(columns=["index_right"])
    return res


def calculate_quartiles(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.qcut(df[column], q=4, labels=False) + 1


def assign_grades(
    graded_territories: gpd.GeoDataFrame, accessibility_data: gpd.GeoDataFrame, local_crs: int
) -> gpd.GeoDataFrame:
    """
    Assign grades to territories based on accessibility data.

    Parameters:
    graded_territories (gpd.GeoDataFrame): GeoDataFrame containing territories to grade.
    accessibility_data (gpd.GeoDataFrame): GeoDataFrame containing accessibility data.

    Returns:
    gpd.GeoDataFrame: Updated graded territories with assigned grades.
    """
    # Ensure both GeoDataFrames have the same CRS
    accessibility_data = accessibility_data.to_crs(epsg=local_crs)
    graded_territories = graded_territories.to_crs(epsg=local_crs)

    # Calculate quartile ranks for 'in_car' and 'in_inter'
    accessibility_data["car_access_quartile"] = calculate_quartiles(accessibility_data, "in_car")
    accessibility_data["public_access_quartile"] = calculate_quartiles(accessibility_data, "in_inter")

    all_gecs_with_dist = gpd.sjoin_nearest(
        graded_territories[
            ["grade", "weight", "geometry", "weight_r_stops", "weight_b_stops", "weight_ferry", "weight_aero"]
        ],
        accessibility_data,
        lsuffix="left",
        rsuffix="right",
        distance_col="dist",
    )

    remote_gecs = all_gecs_with_dist[all_gecs_with_dist["dist"] > 0]
    remote_gecs["public_access_quartile"] = np.minimum(remote_gecs["public_access_quartile"] + 1, 4)
    remote_gecs["car_access_quartile"] = np.minimum(remote_gecs["car_access_quartile"] + 1, 4)

    norm_gecs = all_gecs_with_dist[all_gecs_with_dist["dist"] == 0]
    norm_gecs["intersection_area"] = norm_gecs.apply(
        lambda row: row["geometry"].intersection(accessibility_data.loc[row["index_right"], "geometry"]).area,
        axis=1,
    )
    norm_gecs = norm_gecs.sort_values("intersection_area", ascending=False).drop_duplicates(subset="geometry")

    joined = pd.concat([norm_gecs, remote_gecs])

    # Initialize columns for car and public transport grades
    joined["car_grade"] = 0.0
    joined["public_transport_grade"] = 0.0

    # Define the grade tables
    car_grade_table = {
        5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4.5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        3.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        3: {"Q4": 0, "Q3": 1, "Q2": 3, "Q1": 3},
        2.5: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 3},
        2: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 2},
        1.5: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 2},
        1: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 1},
        0: {"Q4": 0, "Q3": 0, "Q2": 0, "Q1": 1},
    }

    public_transport_grade_table = {
        5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4.5: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        4: {"Q4": 2, "Q3": 3, "Q2": 4, "Q1": 5},
        3.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 5},
        3: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        2.5: {"Q4": 1, "Q3": 2, "Q2": 3, "Q1": 4},
        2: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 4},
        1.5: {"Q4": 0, "Q3": 1, "Q2": 2, "Q1": 3},
        1: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 3},
        0: {"Q4": 0, "Q3": 0, "Q2": 1, "Q1": 2},
    }

    # Apply grades based on the quartiles and grade
    for idx, row in joined.iterrows():
        grade = row["grade"]
        car_quartile = row["car_access_quartile"]
        public_transport_quartile = row["public_access_quartile"]
        car_grade = car_grade_table.get(grade, {}).get(f"Q{car_quartile}", 0)
        public_transport_grade = (
            public_transport_grade_table.get(grade, {}).get(f"Q{public_transport_quartile}", 0) * row["weight"]
        )

        joined.at[idx, "car_grade"] = round(car_grade, 3)
        joined.at[idx, "public_transport_grade"] = round(public_transport_grade, 3)
    joined["overall_assessment"] = (joined["car_grade"] + joined["public_transport_grade"]) / 2
    joined.rename(columns={"name_left": "name"}, inplace=True)
    joined = joined.sort_index()

    return joined[
        [
            "geometry",
            "name",
            "grade",
            "weight",
            "weight_r_stops",
            "weight_b_stops",
            "weight_ferry",
            "weight_aero",
            "car_access_quartile",
            "public_access_quartile",
            "car_grade",
            "public_transport_grade",
            "overall_assessment",
        ]
    ]


def get_criteria(
    graded_terr: gpd.GeoDataFrame,
    points: gpd.GeoDataFrame,
    polygons: gpd.GeoDataFrame,
    drive_graph: nx.MultiDiGraph = None,
    PT_graph: nx.MultiDiGraph = None,
    r_stops: gpd.GeoDataFrame = None,
    b_stops: gpd.GeoDataFrame = None,
    ferry: gpd.GeoDataFrame = None,
    aero: gpd.GeoDataFrame = None,
    adj_mx_drive: pd.DataFrame = None,
    adj_mx_PT: pd.DataFrame = None,
    local_crs: int = 3857,
) -> gpd.GeoDataFrame:
    """
    Get the criteria for graded territories based on points and polygons.

    Parameters:
    graded_terr (gpd.GeoDataFrame): GeoDataFrame containing graded territories.
    points (gpd.GeoDataFrame): GeoDataFrame containing points of interest.
    polygons (gpd.GeoDataFrame): GeoDataFrame containing polygons for spatial joining.
    drive_graph (nx.MultiDiGraph, optional): MultiDiGraph representing the city graph.
    PT_graph (nx.MultiDiGraph, optional): MultiDiGraph representing the interconnection graph.
    r_stops (gpd.GeoDataFrame, optional): GeoDataFrame containing railway stops.
    b_stops (gpd.GeoDataFrame, optional): GeoDataFrame containing bus stops.
    ferry (gpd.GeoDataFrame, optional): GeoDataFrame containing ferry stops.
    aero (gpd.GeoDataFrame, optional): GeoDataFrame containing airports.
    adj_mx_drive (pd.DataFrame, optional): Adjacency matrix for driving.
    adj_mx_inter (pd.DataFrame, optional): Adjacency matrix for public transport.

    Returns:
    gpd.GeoDataFrame: GeoDataFrame with updated criteria based on spatial analysis.
    """

    # graded_terr = graded_terr.reset_index(drop=True,inplace=True)
    graded_terr = graded_terr.to_crs(local_crs).copy()
    points = PointSchema(points).to_crs(local_crs).copy()

    polygons = PolygonSchema(polygons).to_crs(local_crs).copy()

    if adj_mx_drive is None:
        if drive_graph is None:
            print("You should specify a drive graph to calculate the matrix")
        # getting drive mx
        print("Getting drive matrix")
        adj_mx_drive = get_adj_matrix_gdf_to_gdf(
            settlement_points.to_crs(local_crs),
            settlement_points.to_crs(local_crs),
            drive_graph,
            weight="time_min",
            dtype=np.float64,
        )
    if adj_mx_PT is None:
        if PT_graph is None:
            print("You should specify a PT graph to calculate the matrix")
        # getting inter mx
        print("Getting PT matrix")
        adj_mx_PT = get_adj_matrix_gdf_to_gdf(
            settlement_points.to_crs(local_crs),
            settlement_points.to_crs(local_crs),
            PT_graph,
            weight="time_min",
            dtype=np.float64,
        )

    # counting drive connectivity
    p = find_median(points, adj_mx_drive)
    p_agg = p[p["to_service"] < np.finfo(np.float64).max].copy()
    res = (
        gpd.sjoin(p_agg, polygons, how="left", predicate="within")
        .groupby("index_right")
        .median(["to_service"])
        .reset_index()
    )

    result_df = pd.merge(polygons.reset_index(), res, left_index=True, right_on="index_right", how="left").rename(
        columns={"to_service": "in_car"}
    )
    result_df = result_df.drop(columns=["index_right"])

    # counting inter connectivity
    p_inter = find_median(points, adj_mx_PT)
    points_inter = p_inter[p_inter["to_service"] < np.finfo(np.float64).max].copy()

    res_inter = (
        gpd.sjoin(points_inter, polygons, how="left", predicate="within")
        .groupby("index_right")
        .median(["to_service"])
        .reset_index()
    )

    result_df_inter = (
        pd.merge(result_df, res_inter, left_index=True, right_on="index_right", how="left")
        .drop(columns=["index_right"])
        .rename(columns={"to_service": "in_inter"})
    )

    # weight territory based on distance to services
    graded_gdf = weight_territory(graded_terr, r_stops, b_stops, ferry, aero, local_crs)

    # assign the grades
    result = assign_grades(graded_gdf, result_df_inter[["name", "geometry", "in_car", "in_inter"]], local_crs)
    return result.to_crs(4326)


# text interpretation of the result

# Глобальные константы
GRADE_DICT = {
    0.0: "Территория находится в удалении (более 1 км) от любых известных дорог.",
    1.0: "Территория находится в непосредственной близости к дороге (попадают в радиус 5 км от границ территории), но нет ни одного узла УДС.",
    2.0: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в более чем 100км)",
    3.0: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в не более чем 100км)",
    3.5: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, являющейся приоритетной, а ближайший федеральный узел находится в не более чем 100км)",
    4.0: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, а ближайший федеральный узел находится в не более чем 10км)",
    4.5: "Территория расположена в непосредственной близости от одной из региональной трассы (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, которая является приоритетной, а ближайший федеральный узел находится в не более чем 10км)",
    5.0: "Территория расположена в непосредственной близости от одной из региональных трасс (в радиус 5 км от границ территории попадает хотя бы 1 узел региональной трассы, являющейся приоритетной, а ближайший федеральный узел находится в не более чем 100км)",
}

CAR_ACCESS_QUART_DICT = {
    1: "Территория попадает в I квартиль связности (лучшие 25% МО) на личном транспорте",
    2: "Территория попадает во II квартиль связности (от 50% до 25% МО) на личном транспорте",
    3: "Территория попадает в III квартиль связности (от 75% до 50% МО) на личном транспорте",
    4: "Территория попадает в IV квартиль связности (худшие 25% МО) на личном транспорте",
}

PUBLIC_ACCESS_QUART_DICT = {
    1: "Территория попадает в I квартиль связности (лучшие 25% МО) на общественном транспорте",
    2: "Территория попадает во II квартиль связности (от 50% до 25% МО) на общественном транспорте",
    3: "Территория попадает в III квартиль связности (от 75% до 50% МО) на общественном транспорте",
    4: "Территория попадает в IV квартиль связности (худшие 25% МО) на общественном транспорте",
}

WEIGHT_R_STOPS_DICT = {False: "В радиусе 15 км отсутствуют ЖД станции", True: "В радиусе 15 км есть ЖД станции"}

WEIGHT_B_STOPS_DICT = {
    False: "В радиусе 15 км отсутствуют автобусные остановки",
    True: "В радиусе 15 км есть автобусные остановки",
}

WEIGHT_FERRY_DICT = {
    False: "В радиусе 15 км отсутствуют порты/причалы/переправы",
    True: "В радиусе 15 км есть порты/причалы/переправы",
}

WEIGHT_AERO_DICT = {False: "В радиусе 15 км отсутствуют аэродромы", True: "В радиусе 15 км есть хотя бы 1 аэродром"}


def interpret_gdf(gdf):
    """Interprets geographic accessibility data for each criterion in the criteria DataFrame.

    This method iterates through the criteria DataFrame, extracts relevant weights and quartiles for
    each criterion, and generates an interpretation of the accessibility based on transport services
    availability and accessibility quartiles.

    Returns:
        list: A list of tuples, each containing the name of the criterion and its corresponding
            interpretation as a list of strings.
    """
    interpretation_list = []
    for i, row in gdf.iterrows():

        interpretation_row = interpretation(
            row["grade"],
            row["weight_r_stops"],
            row["weight_b_stops"],
            row["weight_ferry"],
            row["weight_aero"],
            row["car_access_quartile"],
            row["public_access_quartile"],
        )
        interpretation_list.append((row["name"], interpretation_row))
    return interpretation_list


def interpretation(
    grade,
    weight_r_stops,
    weight_b_stops,
    weight_ferry,
    weight_aero,
    car_access_quartile,
    public_access_quartile,
):
    """Generates a textual interpretation of accessibility data based on transport services and
    accessibility quartiles.

    This method assesses the availability of transport services, as well as the quartiles for car
    and public access, to create a comprehensive interpretation of accessibility.

    Args:
        grade (int): The grade of the area, used to describe its general quality.
        weight_r_stops (float): Weight indicating the presence of rail stops.
        weight_b_stops (float): Weight indicating the presence of bus stops.
        weight_ferry (float): Weight indicating the presence of ferry services.
        weight_aero (float): Weight indicating the presence of airports.
        car_access_quartile (int): Quartile score for car access (0-4).
        public_access_quartile (int): Quartile score for public access (0-4).

    Returns:
        list: A list of interpretation texts summarizing the accessibility of the area based on
              transport services and quartiles.
    """

    texts = []

    # Frame interpretation
    grade_text = GRADE_DICT[grade] + ";"
    texts.append(grade_text)

    # Transport services availability
    if all([weight_r_stops > 0, weight_b_stops > 0, weight_ferry > 0, weight_aero > 0]):
        services_text = "В радиусе 15 км есть инфраструктура наземного, водного, воздушного общественного транспорта."
        normalized_services = 1  # All services available
    else:
        missing_services = []

        if weight_r_stops == 0:
            missing_services.append("ЖД станции")
        if weight_b_stops == 0:
            missing_services.append("автобусные остановки")
        if weight_ferry == 0:
            missing_services.append("порты/причалы/переправы")
        if weight_aero == 0:
            missing_services.append("аэродромы")

        services_text = f"В радиусе 15 км отсутствуют {', '.join(missing_services)}."
        normalized_services = sum([weight_r_stops > 0, weight_b_stops > 0, weight_ferry > 0, weight_aero > 0]) / 4

    # Interpretation by accessibility quartiles
    car_access_text = CAR_ACCESS_QUART_DICT[car_access_quartile] + ";"
    normalized_car_access = (5 - car_access_quartile) / 4  # From 0 to 1 (reversed)

    public_access_text = PUBLIC_ACCESS_QUART_DICT[public_access_quartile] + ";"
    normalized_public_access = (5 - public_access_quartile) / 4  # From 0 to 1 (reversed)

    # Sorting scores by quartiles
    quartile_grades = sorted(
        [(normalized_car_access, car_access_text), (normalized_public_access, public_access_text)],
        reverse=True,
        key=lambda x: x[0],
    )

    # Sorting grades by service
    service_grades = [(normalized_services, services_text)]

    sorted_grades = quartile_grades + service_grades

    # Final interpretation
    texts.extend([text for _, text in sorted_grades])

    return texts
