#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames criteria methods."""

import argparse
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
    parser = argparse.ArgumentParser(description="Calculate transport_frames criteria.")
    parser.add_argument("--operation", type=str, required=True, choices=["grade_territory"])

    parser.add_argument("--frame-graph-path", type=str, default=None)
    parser.add_argument("--territories-path", type=str, default=None)
    parser.add_argument("--include-priority", type=int, default=1)

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


def _load_graph(path: str | None):
    if path is None:
        return None
    if not os.path.exists(path):
        raise FileNotFoundError(f"Graph file not found: {path}")
    with open(path, "rb") as f:
        return pickle.load(f)


def _read_gdf(path: str | None) -> gpd.GeoDataFrame | None:
    if path is None:
        return None
    return gpd.read_file(path)


def _require(value, name: str):
    if value is None:
        raise ValueError(f"Missing required input: {name}")
    if isinstance(value, gpd.GeoDataFrame) and value.empty:
        raise ValueError(f"Input layer is empty: {name}")


def _resolve_local_crs(
    polygons: gpd.GeoDataFrame | None = None,
    graph=None,
    fallback: int = 3857,
) -> int:
    if polygons is not None and not polygons.empty:
        try:
            estimated = polygons.estimate_utm_crs()
        except Exception:
            estimated = None
        if estimated is not None:
            epsg = estimated.to_epsg()
            if epsg is not None:
                return int(epsg)

    if graph is not None and isinstance(graph.graph, dict):
        graph_crs = graph.graph.get("crs")
        if graph_crs is not None:
            try:
                return int(graph_crs)
            except Exception:
                pass

    return int(fallback)


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
    grade_territory = import_transport_frames("transport_frames.criteria", "grade_territory")

    result = None
    output_local_crs = None

    if args.operation == "grade_territory":
        frame_graph = _load_graph(args.frame_graph_path)
        territories = _read_gdf(args.territories_path)
        _require(frame_graph, "frame graph")
        _require(territories, "territories polygons")

        include_priority = bool(args.include_priority)
        output_local_crs = _resolve_local_crs(territories, graph=frame_graph)
        result = grade_territory(frame=frame_graph, gdf_poly=territories, include_priority=include_priority)

    else:
        raise ValueError(f"Unsupported operation: {args.operation}")

    if result is None:
        raise ValueError("Criteria method returned no data.")

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
