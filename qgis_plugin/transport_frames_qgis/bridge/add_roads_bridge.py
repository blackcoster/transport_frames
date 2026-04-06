#!/usr/bin/env python3
"""External bridge script for QGIS -> transport_frames add_roads."""

import argparse
import os
import pickle
import tempfile

import geopandas as gpd
import momepy
import pandas as pd
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
    parser = argparse.ArgumentParser(description="Add roads to graph via transport_frames.")
    parser.add_argument("--input-graph", type=str, required=True)
    parser.add_argument("--new-roads-path", type=str, required=True)
    parser.add_argument("--graph-out", type=str, required=True)
    parser.add_argument("--edges-out", type=str, required=True)
    parser.add_argument("--nodes-out", type=str, required=True)
    parser.add_argument("--edges-layer", type=str, default=None)
    parser.add_argument("--nodes-layer", type=str, default=None)
    return parser.parse_args()


def _configure_runtime_environment() -> str:
    """
    Configure writable runtime directory for external execution from QGIS.
    """
    runtime_root = os.path.join(tempfile.gettempdir(), "transport_frames_qgis_runtime")
    os.makedirs(runtime_root, exist_ok=True)
    os.chdir(runtime_root)
    return runtime_root


def _resolve_local_crs(graph) -> int:
    """
    Resolve local CRS from graph metadata or fallback to node CRS.
    """
    graph_crs = None
    if hasattr(graph, "graph") and isinstance(graph.graph, dict):
        graph_crs = graph.graph.get("crs")

    if graph_crs is not None:
        try:
            return int(graph_crs)
        except Exception:
            try:
                from pyproj import CRS

                epsg = CRS.from_user_input(graph_crs).to_epsg()
                if epsg is not None:
                    return int(epsg)
            except Exception:
                pass

    try:
        nodes, _ = momepy.nx_to_gdf(graph)
        if nodes.crs is not None:
            epsg = nodes.crs.to_epsg()
            if epsg is not None:
                return int(epsg)
    except Exception:
        pass

    raise ValueError(
        "Could not determine local CRS from graph. "
        "Expected graph.graph['crs'] or a valid nodes CRS."
    )


def _save_graph(graph, graph_path: str) -> None:
    out_dir = os.path.dirname(graph_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
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

    if not os.path.exists(args.input_graph):
        raise FileNotFoundError(f"Input graph not found: {args.input_graph}")
    if not os.path.exists(args.new_roads_path):
        raise FileNotFoundError(f"New roads layer not found: {args.new_roads_path}")

    with open(args.input_graph, "rb") as f:
        graph = pickle.load(f)

    new_roads = gpd.read_file(args.new_roads_path)
    if new_roads.empty:
        raise ValueError("New roads layer is empty.")
    if "reg" not in new_roads.columns:
        raise ValueError("New roads layer must contain 'reg' column.")
    if new_roads.crs is None:
        raise ValueError("New roads layer has no CRS. Please define CRS in the source layer.")

    local_crs = _resolve_local_crs(graph)
    new_roads = new_roads.to_crs(local_crs).copy()
    new_roads["reg"] = pd.to_numeric(new_roads["reg"], errors="coerce")
    if new_roads["reg"].isna().any():
        raise ValueError("Column 'reg' contains non-numeric or empty values.")
    new_roads["reg"] = new_roads["reg"].astype(int)

    add_roads = import_transport_frames("transport_frames.road_adder", "add_roads")
    updated_graph = add_roads(
        citygraph=graph,
        line_gdf=new_roads,
        local_crs=local_crs,
    )

    _save_graph(updated_graph, args.graph_out)

    nodes, edges = momepy.nx_to_gdf(updated_graph)
    nodes = _prepare_attrs_for_export(nodes)
    edges = _prepare_attrs_for_export(edges)

    _export_gdf(nodes, args.nodes_out, args.nodes_layer or "nodes")
    _export_gdf(edges, args.edges_out, args.edges_layer or "edges")

    print(f"Runtime root: {runtime_root}")
    print(f"transport_frames loaded from: {tf_module_file}")
    print(f"Local CRS resolved from graph: EPSG:{local_crs}")
    print(f"Updated graph saved to: {args.graph_out}")
    print(f"Nodes saved to: {args.nodes_out}")
    print(f"Edges saved to: {args.edges_out}")


if __name__ == "__main__":
    main()

