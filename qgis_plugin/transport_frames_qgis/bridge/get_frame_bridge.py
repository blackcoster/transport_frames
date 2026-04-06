#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames get_frame + weigh_roads."""

import argparse
import os
import pickle
import tempfile

import geopandas as gpd
import momepy
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
    parser = argparse.ArgumentParser(description="Build weighted frame via transport_frames.")
    parser.add_argument("--input-graph", type=str, required=True)
    parser.add_argument("--admin-centers-path", type=str, required=True)
    parser.add_argument("--area-path", type=str, default=None)
    parser.add_argument("--area-osm-id", type=int, default=None)
    parser.add_argument("--regions-path", type=str, required=True)
    parser.add_argument("--graph-out", type=str, required=True)
    parser.add_argument("--edges-out", type=str, required=True)
    parser.add_argument("--nodes-out", type=str, required=True)
    parser.add_argument("--edges-layer", type=str, default=None)
    parser.add_argument("--nodes-layer", type=str, default=None)
    return parser.parse_args()


def _configure_runtime_environment() -> str:
    """
    Configure writable runtime/cache paths for external execution from QGIS.
    """
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


def _save_graph(graph, graph_path: str) -> None:
    graph_dir = os.path.dirname(graph_path)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    with open(graph_path, "wb") as f:
        pickle.dump(graph, f)


def _export_gdf(gdf: gpd.GeoDataFrame, output_path: str, layer_name: str | None) -> None:
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

    if (args.area_path is None) == (args.area_osm_id is None):
        raise ValueError("Provide exactly one of --area-path or --area-osm-id.")

    if not os.path.exists(args.input_graph):
        raise FileNotFoundError(f"Input graph not found: {args.input_graph}")

    with open(args.input_graph, "rb") as f:
        graph = pickle.load(f)

    admin_centers = gpd.read_file(args.admin_centers_path)
    regions_polygons = gpd.read_file(args.regions_path)
    if admin_centers.empty:
        raise ValueError("Administrative centers layer is empty.")
    if regions_polygons.empty:
        raise ValueError("Region boundaries are empty.")

    if args.area_path is not None:
        area_polygon = gpd.read_file(args.area_path)
    else:
        from iduedu.modules.overpass.overpass_downloaders import get_4326_boundary

        try:
            boundary = get_4326_boundary(osm_id=args.area_osm_id)
        except Exception as exc:
            raise ValueError(
                f"Failed to fetch boundary for OSM relation id {args.area_osm_id}. "
                "Make sure this is a valid OSM relation ID."
            ) from exc
        area_polygon = gpd.GeoDataFrame(geometry=[boundary], crs=4326)
    if area_polygon.empty:
        raise ValueError("Area boundary is empty.")

    get_frame = import_transport_frames("transport_frames.frame", "get_frame")
    weigh_roads = import_transport_frames("transport_frames.frame", "weigh_roads")

    frame = get_frame(
        graph=graph,
        admin_centers=admin_centers,
        area_polygon=area_polygon,
        region_polygons=regions_polygons,
    )
    weighted_graph = weigh_roads(frame)

    _save_graph(weighted_graph, args.graph_out)

    nodes, edges = momepy.nx_to_gdf(weighted_graph)
    nodes = _prepare_attrs_for_export(nodes)
    edges = _prepare_attrs_for_export(edges)

    _export_gdf(nodes, args.nodes_out, args.nodes_layer or "nodes")
    _export_gdf(edges, args.edges_out, args.edges_layer or "edges")

    print(f"Runtime cache root: {runtime_root}")
    print(f"transport_frames loaded from: {tf_module_file}")
    print(f"Weighted frame graph saved to: {args.graph_out}")
    print(f"Nodes saved to: {args.nodes_out}")
    print(f"Edges saved to: {args.edges_out}")


if __name__ == "__main__":
    main()
