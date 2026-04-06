import os

from PyQt5.QtCore import QSettings
from qgis.core import QgsApplication


SETTINGS_PREFIX = "transport_frames_qgis"
MODE_MANAGED = "managed"
MODE_CUSTOM = "custom"


def _key(name: str) -> str:
    return f"{SETTINGS_PREFIX}/{name}"


def get_mode() -> str:
    value = QSettings().value(_key("mode"), MODE_MANAGED, type=str)
    if value not in {MODE_MANAGED, MODE_CUSTOM}:
        return MODE_MANAGED
    return value


def set_mode(mode: str) -> None:
    if mode not in {MODE_MANAGED, MODE_CUSTOM}:
        raise ValueError(f"Unknown mode: {mode}")
    QSettings().setValue(_key("mode"), mode)


def get_custom_python_path() -> str:
    return (QSettings().value(_key("custom_python_path"), "", type=str) or "").strip()


def set_custom_python_path(path: str) -> None:
    QSettings().setValue(_key("custom_python_path"), path)


def get_default_managed_env_dir() -> str:
    return os.path.join(QgsApplication.qgisSettingsDirPath(), "transport_frames_env")


def get_managed_env_dir() -> str:
    return (QSettings().value(_key("managed_env_dir"), get_default_managed_env_dir(), type=str) or "").strip()


def set_managed_env_dir(path: str) -> None:
    QSettings().setValue(_key("managed_env_dir"), path)


def get_managed_python_path() -> str:
    env_dir = get_managed_env_dir()
    if os.name == "nt":
        return os.path.join(env_dir, "Scripts", "python.exe")
    return os.path.join(env_dir, "bin", "python")


def has_managed_python() -> bool:
    python_path = get_managed_python_path()
    return os.path.exists(python_path) and os.access(python_path, os.X_OK)


def normalize_user_path(path: str) -> str:
    result = path.strip().strip('"').strip("'")
    result = os.path.expanduser(result)
    if not os.path.isabs(result) and os.path.exists(os.path.sep + result):
        result = os.path.sep + result
    return result


def resolve_python_executable(path: str) -> str:
    result = normalize_user_path(path)

    if os.path.isdir(result):
        if os.name == "nt":
            candidates = [
                os.path.join(result, "Scripts", "python.exe"),
                os.path.join(result, "python.exe"),
            ]
        else:
            candidates = [
                os.path.join(result, "bin", "python"),
                os.path.join(result, "bin", "python3"),
                os.path.join(result, "bin", "python3.11"),
            ]
        for candidate in candidates:
            if os.path.exists(candidate):
                result = candidate
                break
        else:
            raise ValueError(f"Provided folder does not contain Python executable: {result}")

    if not os.path.exists(result):
        raise ValueError(f"Python executable does not exist: {result}")
    if not os.access(result, os.X_OK):
        raise ValueError(f"Python executable is not executable: {result}")
    return result


def build_subprocess_env(python_bin: str) -> dict:
    env = os.environ.copy()
    for var in ("PYTHONHOME", "PYTHONPATH", "PYTHONEXECUTABLE", "__PYVENV_LAUNCHER__"):
        env.pop(var, None)

    venv_bin = os.path.dirname(python_bin)
    env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
    return env


def get_last_setup_python_source() -> str:
    return (QSettings().value(_key("last_setup_python_source"), "", type=str) or "").strip()


def set_last_setup_python_source(path: str) -> None:
    QSettings().setValue(_key("last_setup_python_source"), path)


def get_last_package_spec() -> str:
    return (QSettings().value(_key("last_package_spec"), "transport_frames", type=str) or "transport_frames").strip()


def set_last_package_spec(package_spec: str) -> None:
    QSettings().setValue(_key("last_package_spec"), package_spec)
