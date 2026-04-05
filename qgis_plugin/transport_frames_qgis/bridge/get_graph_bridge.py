#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames get_graph."""

import argparse
import os
import pickle
import tempfile

import geopandas as gpd
import momepy


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


def _prepare_edges_for_export(edges_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    edges = edges_gdf.copy()
    for col in edges.columns:
        if col == "geometry":
            continue
        if edges[col].dtype == "object":
            edges[col] = edges[col].apply(_normalize_value)
    return edges


def _parse_args():
    parser = argparse.ArgumentParser(description="Build drive graph via transport_frames.")
    parser.add_argument("--osm-id", type=int, default=None)
    parser.add_argument("--territory-path", type=str, default=None)
    parser.add_argument("--buffer", type=int, default=3000)
    parser.add_argument("--graph-out", type=str, required=True)
    parser.add_argument("--edges-out", type=str, required=True)
    parser.add_argument("--edges-layer", type=str, default=None)
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

    # Keep all relative cache paths away from potential read-only cwd (e.g., "/").
    os.chdir(runtime_root)

    # iduedu reads these env vars in config.
    os.environ["OVERPASS_CACHE_DIR"] = iduedu_cache
    os.environ["OVERPASS_CACHE_ENABLED"] = "1"

    # osmnx uses settings.cache_folder (default "./cache"), set explicit writable path.
    import osmnx as ox

    ox.settings.cache_folder = osmnx_cache
    ox.settings.use_cache = True
    return runtime_root


def main():
    args = _parse_args()
    runtime_root = _configure_runtime_environment()

    if (args.osm_id is None) == (args.territory_path is None):
        raise ValueError("Provide exactly one of --osm-id or --territory-path.")

    from transport_frames.graph import get_graph

    territory = None
    if args.territory_path is not None:
        territory = gpd.read_file(args.territory_path)
        if territory.empty:
            raise ValueError("Territory layer is empty.")

    if args.osm_id is not None:
        graph = get_graph(osm_id=args.osm_id, buffer=args.buffer)
    else:
        graph = get_graph(territory=territory, buffer=args.buffer)

    graph_dir = os.path.dirname(args.graph_out)
    if graph_dir:
        os.makedirs(graph_dir, exist_ok=True)
    with open(args.graph_out, "wb") as f:
        pickle.dump(graph, f)

    _, edges = momepy.nx_to_gdf(graph)
    edges = _prepare_edges_for_export(edges)
    edges_dir = os.path.dirname(args.edges_out)
    if edges_dir:
        os.makedirs(edges_dir, exist_ok=True)
    if args.edges_out.lower().endswith(".gpkg"):
        layer = args.edges_layer or "edges"
        edges.to_file(args.edges_out, layer=layer, driver="GPKG")
    else:
        edges.to_file(args.edges_out)

    print(f"Runtime cache root: {runtime_root}")
    print(f"Graph saved to: {args.graph_out}")
    print(f"Edges saved to: {args.edges_out}")


if __name__ == "__main__":
    main()
