#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames get_intermodal_graph."""

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
    parser = argparse.ArgumentParser(description="Build intermodal graph via transport_frames.")
    parser.add_argument("--osm-id", type=int, default=None)
    parser.add_argument("--territory-path", type=str, default=None)
    parser.add_argument("--graph-out", type=str, required=True)
    parser.add_argument("--edges-out", type=str, required=True)
    parser.add_argument("--nodes-out", type=str, required=True)
    parser.add_argument("--edges-layer", type=str, default=None)
    parser.add_argument("--nodes-layer", type=str, default=None)
    return parser.parse_args()


def _configure_runtime_environment() -> str:
    runtime_root = os.path.join(tempfile.gettempdir(), "transport_frames_qgis_runtime")
    os.makedirs(runtime_root, exist_ok=True)

    iduedu_cache = os.path.join(runtime_root, "iduedu_cache")
    os.makedirs(iduedu_cache, exist_ok=True)

    os.chdir(runtime_root)
    os.environ["OVERPASS_CACHE_DIR"] = iduedu_cache
    os.environ["OVERPASS_CACHE_ENABLED"] = "1"
    return runtime_root


def main():
    args = _parse_args()
    runtime_root = _configure_runtime_environment()
    tf_module_file = ensure_transport_frames_from_env()

    if (args.osm_id is None) == (args.territory_path is None):
        raise ValueError("Provide exactly one of --osm-id or --territory-path.")

    get_intermodal_graph = import_transport_frames("transport_frames.graph", "get_intermodal_graph")

    territory = None
    if args.territory_path is not None:
        territory = gpd.read_file(args.territory_path)
        if territory.empty:
            raise ValueError("Territory layer is empty.")

    if args.osm_id is not None:
        graph = get_intermodal_graph(osm_id=args.osm_id)
    else:
        graph = get_intermodal_graph(territory=territory)

    graph_dir = os.path.dirname(args.graph_out)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    with open(args.graph_out, "wb") as f:
        pickle.dump(graph, f)

    nodes, edges = momepy.nx_to_gdf(graph)
    nodes = _prepare_attrs_for_export(nodes)
    edges = _prepare_attrs_for_export(edges)

    nodes_dir = os.path.dirname(args.nodes_out)
    if nodes_dir:
        os.makedirs(nodes_dir, exist_ok=True)
    if args.nodes_out.lower().endswith(".gpkg"):
        layer = args.nodes_layer or "nodes"
        nodes.to_file(args.nodes_out, layer=layer, driver="GPKG")
    else:
        nodes.to_file(args.nodes_out)

    edges_dir = os.path.dirname(args.edges_out)
    if edges_dir:
        os.makedirs(edges_dir, exist_ok=True)
    if args.edges_out.lower().endswith(".gpkg"):
        layer = args.edges_layer or "edges"
        edges.to_file(args.edges_out, layer=layer, driver="GPKG")
    else:
        edges.to_file(args.edges_out)

    print(f"Runtime cache root: {runtime_root}")
    print(f"transport_frames loaded from: {tf_module_file}")
    print(f"Graph saved to: {args.graph_out}")
    print(f"Nodes saved to: {args.nodes_out}")
    print(f"Edges saved to: {args.edges_out}")


if __name__ == "__main__":
    main()
