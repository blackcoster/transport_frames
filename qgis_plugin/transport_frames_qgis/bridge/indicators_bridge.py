#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames indicators."""

import argparse
import inspect
import os
import pickle
import tempfile

import geopandas as gpd
from tf_import_guard import ensure_transport_frames_from_env, import_transport_frames


def _normalize_value(value):
    if value is None:
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if isinstance(value, (list, tuple, set, dict)):
        return str(value)
    return value


def _prepare_attrs_for_export(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    result = gdf.copy()
    for col in result.columns:
        if col == "geometry":
            continue
        if result[col].dtype == "object":
            result[col] = result[col].apply(_normalize_value)
    return result


def _parse_args():
    parser = argparse.ArgumentParser(description="Calculate transport_frames indicators.")
    parser.add_argument(
        "--operation",
        type=str,
        required=True,
        choices=[
            "road_length",
            "road_density",
            "reg_length",
            "railway_length",
            "connectivity",
            "service_count",
            "service_accessibility",
            "terr_service_count",
            "terr_service_accessibility",
        ],
    )
    parser.add_argument("--graph-path", type=str, default=None)
    parser.add_argument("--area-path", type=str, default=None)
    parser.add_argument("--territory-path", type=str, default=None)
    parser.add_argument("--settlements-path", type=str, default=None)
    parser.add_argument("--service-path", type=str, default=None)
    parser.add_argument("--railways-path", type=str, default=None)
    parser.add_argument("--output-path", type=str, required=True)
    parser.add_argument("--output-layer", type=str, default="result")
    return parser.parse_args()


def _configure_runtime_environment() -> str:
    runtime_root = os.path.join(tempfile.gettempdir(), "transport_frames_qgis_runtime")
    os.makedirs(runtime_root, exist_ok=True)

    osmnx_cache = os.path.join(runtime_root, "osmnx_cache")
    iduedu_cache = os.path.join(runtime_root, "iduedu_cache")
    os.makedirs(osmnx_cache, exist_ok=True)
    os.makedirs(iduedu_cache, exist_ok=True)

    os.chdir(runtime_root)
    os.environ["OVERPASS_CACHE_DIR"] = iduedu_cache
    os.environ["OVERPASS_CACHE_ENABLED"] = "1"

    import osmnx as ox

    ox.settings.cache_folder = osmnx_cache
    ox.settings.use_cache = True
    return runtime_root


def _read_gdf(path: str | None) -> gpd.GeoDataFrame | None:
    if path is None:
        return None
    gdf = gpd.read_file(path)
    return gdf


def _load_graph(path: str | None):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _resolve_local_crs(*gdfs: gpd.GeoDataFrame, graph=None, fallback: int = 3857) -> int:
    if graph is not None and isinstance(graph.graph, dict):
        graph_crs = graph.graph.get("crs")
        if graph_crs is not None:
            try:
                return int(graph_crs)
            except Exception:
                pass

    for gdf in gdfs:
        if gdf is None or gdf.empty:
            continue
        try:
            estimated = gdf.estimate_utm_crs()
        except Exception:
            estimated = None
        if estimated is None:
            continue
        epsg = estimated.to_epsg()
        if epsg is not None:
            return int(epsg)

    return int(fallback)


def _supports_local_crs(func) -> bool:
    try:
        return "local_crs" in inspect.signature(func).parameters
    except Exception:
        return False


def _to_crs(gdf: gpd.GeoDataFrame | None, target_crs: int | None, name: str) -> gpd.GeoDataFrame | None:
    if gdf is None:
        return None
    if target_crs is None:
        return gdf.copy()

    result = gdf.copy()
    if result.crs is None:
        raise ValueError(
            f"Input layer '{name}' has no CRS defined. "
            f"Cannot convert it to local CRS EPSG:{target_crs}."
        )
    return result.to_crs(target_crs)


def _require(value, name: str):
    if value is None:
        raise ValueError(f"Missing required input: {name}")
    if isinstance(value, gpd.GeoDataFrame) and value.empty:
        raise ValueError(f"Input layer is empty: {name}")


def _save_output(gdf: gpd.GeoDataFrame, output_path: str, layer_name: str) -> None:
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    if output_path.lower().endswith(".gpkg"):
        gdf.to_file(output_path, layer=layer_name, driver="GPKG")
    else:
        gdf.to_file(output_path)


def main():
    args = _parse_args()
    runtime_root = _configure_runtime_environment()
    tf_module_file = ensure_transport_frames_from_env()

    get_connectivity = import_transport_frames("transport_frames.indicators", "get_connectivity")
    get_railway_length = import_transport_frames("transport_frames.indicators", "get_railway_length")
    get_reg_length = import_transport_frames("transport_frames.indicators", "get_reg_length")
    get_road_density = import_transport_frames("transport_frames.indicators", "get_road_density")
    get_road_length = import_transport_frames("transport_frames.indicators", "get_road_length")
    get_service_accessibility = import_transport_frames("transport_frames.indicators", "get_service_accessibility")
    get_service_count = import_transport_frames("transport_frames.indicators", "get_service_count")
    get_terr_service_accessibility = import_transport_frames(
        "transport_frames.indicators", "get_terr_service_accessibility"
    )
    get_terr_service_count = import_transport_frames("transport_frames.indicators", "get_terr_service_count")

    graph = _load_graph(args.graph_path)
    area = _read_gdf(args.area_path)
    territory = _read_gdf(args.territory_path)
    settlements = _read_gdf(args.settlements_path)
    service = _read_gdf(args.service_path)
    railways = _read_gdf(args.railways_path)
    output_local_crs = None

    if args.operation == "road_length":
        _require(graph, "graph")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(area, graph=graph)
        area_local = _to_crs(area, output_local_crs, "area polygons")
        result = get_road_length(graph=graph, area_polygons=area_local)
    elif args.operation == "road_density":
        _require(graph, "graph")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(area, graph=graph)
        area_local = _to_crs(area, output_local_crs, "area polygons")
        result = get_road_density(graph=graph, area_polygons=area_local)
    elif args.operation == "reg_length":
        _require(graph, "graph")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(area, graph=graph)
        area_local = _to_crs(area, output_local_crs, "area polygons")
        result = get_reg_length(graph=graph, area_polygons=area_local)
    elif args.operation == "railway_length":
        _require(railways, "railways")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(railways, area, graph=graph)
        railways_local = _to_crs(railways, output_local_crs, "railways")
        area_local = _to_crs(area, output_local_crs, "area polygons")
        if _supports_local_crs(get_railway_length):
            local_crs = output_local_crs
            result = get_railway_length(railway_paths=railways_local, area_polygons=area_local, local_crs=local_crs)
        else:
            result = get_railway_length(railway_paths=railways_local, area_polygons=area_local)
    elif args.operation == "connectivity":
        _require(graph, "graph")
        _require(settlements, "settlement points")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(settlements, area, graph=graph)
        settlements_local = _to_crs(settlements, output_local_crs, "settlement points")
        area_local = _to_crs(area, output_local_crs, "area polygons")
        if _supports_local_crs(get_connectivity):
            local_crs = output_local_crs
            result = get_connectivity(
                settlement_points=settlements_local,
                area_polygons=area_local,
                local_crs=local_crs,
                graph=graph,
            )
        else:
            result = get_connectivity(
                settlement_points=settlements_local,
                area_polygons=area_local,
                graph=graph,
            )
    elif args.operation == "service_count":
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(area, graph=graph)
        area_local = _to_crs(area, output_local_crs, "area polygons")
        service_local = _to_crs(service, output_local_crs, "service points") if service is not None else None
        result = get_service_count(area_polygons=area_local, service=service_local)
    elif args.operation == "service_accessibility":
        _require(graph, "graph")
        _require(settlements, "settlement points")
        _require(area, "area polygons")
        output_local_crs = _resolve_local_crs(settlements, area, graph=graph)
        settlements_local = _to_crs(settlements, output_local_crs, "settlement points")
        area_local = _to_crs(area, output_local_crs, "area polygons")
        service_local = _to_crs(service, output_local_crs, "service points") if service is not None else None
        result = get_service_accessibility(
            settlement_points=settlements_local,
            graph=graph,
            area_polygons=area_local,
            service=service_local,
        )
    elif args.operation == "terr_service_count":
        _require(territory, "territory polygons")
        output_local_crs = _resolve_local_crs(territory, service, graph=graph)
        territory_local = _to_crs(territory, output_local_crs, "territory polygons")
        service_local = _to_crs(service, output_local_crs, "service points") if service is not None else None
        if _supports_local_crs(get_terr_service_count):
            local_crs = output_local_crs
            result = get_terr_service_count(territory_polygon=territory_local, service=service_local, local_crs=local_crs)
        else:
            result = get_terr_service_count(territory_polygon=territory_local, service=service_local)
    elif args.operation == "terr_service_accessibility":
        _require(graph, "graph")
        _require(territory, "territory polygons")
        output_local_crs = _resolve_local_crs(territory, graph=graph)
        territory_local = _to_crs(territory, output_local_crs, "territory polygons")
        service_local = _to_crs(service, output_local_crs, "service points") if service is not None else None
        result = get_terr_service_accessibility(graph=graph, territory_polygon=territory_local, service=service_local)
    else:
        raise ValueError(f"Unsupported operation: {args.operation}")

    if result is None:
        raise ValueError("Indicator returned no data.")

    # transport_frames indicators commonly return EPSG:4326.
    # Convert back to local/projected CRS for QGIS layer output.
    if output_local_crs is not None:
        try:
            if result.crs is None:
                result = result.set_crs(4326, allow_override=True)
            result = result.to_crs(output_local_crs)
        except Exception as exc:
            raise ValueError(f"Failed to convert result to local CRS EPSG:{output_local_crs}") from exc

    result = _prepare_attrs_for_export(result)
    _save_output(result, args.output_path, args.output_layer)

    print(f"Runtime cache root: {runtime_root}")
    print(f"transport_frames loaded from: {tf_module_file}")
    print(f"Operation completed: {args.operation}")
    print(f"Result saved to: {args.output_path}")


if __name__ == "__main__":
    main()
