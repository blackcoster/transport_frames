import os
import shutil
import subprocess

from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterString,
)

from ..env_manager import (
    MODE_MANAGED,
    build_subprocess_env,
    get_managed_env_dir,
    get_managed_python_path,
    get_last_package_spec,
    get_last_setup_python_source,
    resolve_python_executable,
    set_last_package_spec,
    set_last_setup_python_source,
    set_managed_env_dir,
    set_mode,
)


class SetupEnvironmentAlgorithm(QgsProcessingAlgorithm):
    PYTHON_SOURCE = "PYTHON_SOURCE"
    PACKAGE_SPEC = "PACKAGE_SPEC"
    RECREATE = "RECREATE"

    def name(self):
        return "setup_environment"

    def displayName(self):
        return "Setup Python Environment"

    def group(self):
        return "1 - Environment"

    def groupId(self):
        return "environment"

    def shortHelpString(self):
        return (
            "Create/update managed Python environment for Transport Frames plugin.\n\n"
            "Steps:\n"
            "1) use provided Python 3.11+ executable (or folder with python),\n"
            "2) create venv in active QGIS profile,\n"
            "3) install/upgrade package from package spec.\n\n"
            "After success, plugin mode is switched to Managed."
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterString(
                self.PYTHON_SOURCE,
                "Python 3.11+ executable path (or folder)",
                defaultValue=get_last_setup_python_source(),
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PACKAGE_SPEC,
                "Package spec to install",
                defaultValue=get_last_package_spec(),
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.RECREATE,
                "Recreate environment from scratch",
                defaultValue=False,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        source_input = self.parameterAsString(parameters, self.PYTHON_SOURCE, context).strip()
        if not source_input:
            raise QgsProcessingException("Python path is required.")

        package_spec = self.parameterAsString(parameters, self.PACKAGE_SPEC, context).strip()
        if not package_spec:
            raise QgsProcessingException("Package spec is required.")

        recreate = self.parameterAsBool(parameters, self.RECREATE, context)
        if recreate and self._looks_like_local_package_spec(package_spec):
            feedback.pushInfo(
                "Recreate mode: local package spec detected. "
                "Falling back to PyPI package 'transport-frames'."
            )
            package_spec = "transport-frames"

        try:
            base_python = resolve_python_executable(source_input)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc

        feedback.pushInfo(f"Using base Python: {base_python}")
        major, minor = self._detect_python_version(base_python)
        if (major, minor) < (3, 11):
            raise QgsProcessingException(
                f"Python >=3.11 required. Detected {major}.{minor} at {base_python}"
            )

        env_dir = get_managed_env_dir()
        env_python = get_managed_python_path()
        set_managed_env_dir(env_dir)

        if recreate and os.path.isdir(env_dir):
            feedback.pushInfo(f"Removing existing environment: {env_dir}")
            shutil.rmtree(env_dir, ignore_errors=True)

        if not os.path.isdir(env_dir):
            feedback.pushInfo(f"Creating virtual environment: {env_dir}")
            self._run_cmd(
                [base_python, "-m", "venv", env_dir],
                build_subprocess_env(base_python),
                feedback,
                "venv creation",
            )
        else:
            feedback.pushInfo(f"Using existing environment: {env_dir}")

        if not os.path.exists(env_python):
            raise QgsProcessingException(f"Managed python not found after setup: {env_python}")

        env_subprocess = build_subprocess_env(env_python)
        feedback.pushInfo("Upgrading pip/setuptools/wheel...")
        self._run_cmd(
            [env_python, "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"],
            env_subprocess,
            feedback,
            "pip bootstrap",
        )

        feedback.pushInfo(f"Installing package: {package_spec}")
        self._run_cmd(
            [env_python, "-m", "pip", "install", "--upgrade", package_spec],
            env_subprocess,
            feedback,
            "package install",
        )

        set_last_setup_python_source(base_python)
        set_last_package_spec(package_spec)
        set_mode(MODE_MANAGED)
        feedback.pushInfo("Managed mode activated.")
        feedback.pushInfo(f"Managed python: {env_python}")

        return {
            "managed_env_dir": env_dir,
            "managed_python": env_python,
            "package_spec": package_spec,
        }

    def createInstance(self):
        return SetupEnvironmentAlgorithm()

    @staticmethod
    def _detect_python_version(python_bin: str):
        proc = subprocess.run(
            [python_bin, "-c", "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')"],
            capture_output=True,
            text=True,
            env=build_subprocess_env(python_bin),
        )
        if proc.returncode != 0:
            raise QgsProcessingException(
                f"Failed to detect Python version for {python_bin}.\n"
                f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"
            )
        ver = (proc.stdout or "").strip()
        try:
            major_str, minor_str = ver.split(".")
            return int(major_str), int(minor_str)
        except Exception as exc:
            raise QgsProcessingException(f"Unexpected Python version output: {ver!r}") from exc

    @staticmethod
    def _run_cmd(cmd, env, feedback, step_name):
        proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if proc.stdout:
            feedback.pushInfo(proc.stdout.strip())
        if proc.returncode != 0:
            raise QgsProcessingException(
                f"Failed at step: {step_name}\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}\n"
                f"stdout:\n{proc.stdout}"
            )

    @staticmethod
    def _looks_like_local_package_spec(package_spec: str) -> bool:
        spec = (package_spec or "").strip().strip('"').strip("'")
        if not spec:
            return False

        expanded = os.path.expanduser(spec)
        if os.path.exists(expanded):
            return True

        local_prefixes = (".", "/", "~", "file://")
        if spec.startswith(local_prefixes):
            return True

        local_suffixes = (".whl", ".tar.gz", ".zip")
        if spec.endswith(local_suffixes):
            return True

        return False
