import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from iduedu import get_adj_matrix_gdf_to_gdf


def get_OD(blocks, stops, walk_graph, crs):
    # blocks columns: population, density, diversity, land_use
    # (density and diversity columns are added when you count centrality)

    blocks.to_crs(crs, inplace=True)
    blocks.loc[blocks["density"].isna(), "density"] = 0.0
    blocks.loc[blocks["diversity"].isna(), "diversity"] = 0.0
    scaler = MinMaxScaler()
    blocks[["density", "diversity"]] = scaler.fit_transform(blocks[["density", "diversity"]])
    blocks = blocks.set_geometry("geometry")
    landuse_coeff = {
        None: 0.06,
        "industrial": 0.25,
        "business": 0.3,
        "special": 0.1,
        "transport": 0.1,
        "residential": 0.1,
        "agriculture": 0.05,
        "recreation": 0.05,
    }
    blocks["lu_coeff"] = blocks["land_use"].apply(lambda x: landuse_coeff.get(x, 0))
    blocks["attractiveness"] = blocks["density"] + blocks["diversity"] + blocks["lu_coeff"]

    stops.to_crs(crs, inplace=True)

    walk_mx = get_adj_matrix_gdf_to_gdf(
        blocks,
        stops,
        walk_graph,
        weight="time_min",
        dtype=np.float64,
    )

    walk_dict = {}

    for i, row in walk_mx.iterrows():
        walk_dict[i] = []
        for j, value in row.items():  # Iterate over columns in the row
            if value <= 10:  # Check condition
                walk_dict[i].append((j, value))
        if len(walk_dict[i]) == 0:
            walk_dict[i].append((row.idxmin(), row.min()))

    # Исходный словарь {блок: [(остановка, расстояние)]}
    block_to_stops = walk_dict.copy()

    # Новый словарь {остановка: [(блок, коэффициент)]}
    block_to_weights = {}

    for block1, stops1 in block_to_stops.items():
        # Разбираем список остановок
        stop_ids = np.array([stop[0] for stop in stops1])  # Индексы остановок
        distances = np.array([stop[1] if stop[1] > 0 else 0.1 for stop in stops1], dtype=np.float64)  # Расстояния

        # Вычисляем обратные веса
        weights = 1 / distances  # Чем ближе, тем больше вес

        # Нормируем веса, чтобы сумма по блоку была 1
        weights_normalized = weights / weights.sum()

        # Записываем результат
        block_to_weights[block1] = list(zip(stop_ids, weights_normalized))

    stops_dict = {s: {"att": 0, "pop": 0} for s in list(stops.index)}

    for key, value in block_to_weights.items():
        for stahp, k in value:

            stops_dict[stahp]["att"] += blocks.iloc[key]["attractiveness"] * k
            stops_dict[stahp]["pop"] += blocks.iloc[key]["population"] * k

    stops["att"] = [v["att"] for k, v in stops_dict.items()]
    stops["pop"] = [v["pop"] for k, v in stops_dict.items()]

    mx_stopstop = get_adj_matrix_gdf_to_gdf(stops, stops, walk_graph, "length_m", dtype=np.float64)
    # Ensure no division by zero
    adj_mx = mx_stopstop.replace(0, np.nan)

    # Compute the OD matrix using the Gravity Model formula
    od_matrix = pd.DataFrame(np.outer(stops["pop"], stops["att"]) / adj_mx, index=adj_mx.index, columns=adj_mx.columns)

    # Fill NaN values (from division by zero) with 0
    od_matrix = od_matrix.fillna(0)

    return od_matrix
