#!/usr/bin/env python3
"""Helpers to ensure transport_frames is imported from active Python environment package dirs."""

from __future__ import annotations

import importlib
import os


def _normalize_path(path: str) -> str:
    return os.path.realpath(path).replace("\\", "/").lower()


def _is_env_package_path(path: str) -> bool:
    norm = _normalize_path(path)
    return "/site-packages/" in norm or "/dist-packages/" in norm


def ensure_transport_frames_from_env() -> str:
    """
    Ensure `transport_frames` is imported from installed package directories.

    Returns
    -------
    str
        Resolved path to imported `transport_frames` module file.

    Raises
    ------
    RuntimeError
        If `transport_frames` is imported from a source tree/local path instead of package dirs.
    """
    module = importlib.import_module("transport_frames")
    module_file = getattr(module, "__file__", None)
    if not module_file:
        raise RuntimeError("Could not resolve 'transport_frames.__file__'.")

    if not _is_env_package_path(module_file):
        raise RuntimeError(
            "transport_frames must be imported from active environment package directories "
            "(site-packages/dist-packages), but got: "
            f"{module_file}. "
            "Use 'Setup Python Environment' with a PyPI package spec."
        )

    return os.path.realpath(module_file)


def import_transport_frames(module_name: str, attr_name: str):
    """
    Import attribute from transport_frames submodule after source guard check.
    """
    ensure_transport_frames_from_env()
    module = importlib.import_module(module_name)
    module_file = getattr(module, "__file__", None)
    if module_file and not _is_env_package_path(module_file):
        raise RuntimeError(
            f"{module_name} must be loaded from site-packages/dist-packages, got: {module_file}"
        )
    return getattr(module, attr_name)
