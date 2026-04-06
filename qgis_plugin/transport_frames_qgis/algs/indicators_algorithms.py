import os
import subprocess
import tempfile

from qgis import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
    QgsProcessingParameterFile,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)

from ..env_manager import (
    MODE_CUSTOM,
    MODE_MANAGED,
    build_subprocess_env,
    get_custom_python_path,
    get_managed_python_path,
    get_mode,
    has_managed_python,
    resolve_python_executable,
    set_custom_python_path,
    set_mode,
)


class _BridgeIndicatorBase(QgsProcessingAlgorithm):
    RUN_MODE = "RUN_MODE"
    PYTHON_BIN = "PYTHON_BIN"
    OUTPUT = "OUTPUT"

    def group(self):
        return "5 - Indicators"

    def groupId(self):
        return "indicators"

    def _bridge_script_path(self) -> str:
        return os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bridge",
            "indicators_bridge.py",
        )

    def _add_python_mode_params(self):
        current_mode = get_mode()
        mode_default = 0 if current_mode == MODE_MANAGED else 1
        self.addParameter(
            QgsProcessingParameterEnum(
                self.RUN_MODE,
                "Python mode",
                options=[
                    "Managed environment (recommended)",
                    "Custom python path",
                ],
                defaultValue=mode_default,
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PYTHON_BIN,
                "Custom mode: Python executable (3.11+) or venv folder",
                defaultValue=get_custom_python_path(),
                optional=True,
            )
        )

    def _add_output_param(self, label: str = "Output layer"):
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                label,
                type=QgsProcessing.TypeVectorPolygon,
            )
        )

    def _resolve_python_bin(self, parameters, context) -> str:
        mode_idx = self.parameterAsEnum(parameters, self.RUN_MODE, context)
        mode = MODE_MANAGED if mode_idx == 0 else MODE_CUSTOM
        set_mode(mode)

        if mode == MODE_MANAGED:
            if not has_managed_python():
                raise QgsProcessingException(
                    "Managed environment is not configured.\n"
                    "Run Transport Frames -> Environment -> Setup Python Environment first."
                )
            try:
                return resolve_python_executable(get_managed_python_path())
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc

        python_input = self.parameterAsString(parameters, self.PYTHON_BIN, context).strip()
        if not python_input:
            python_input = get_custom_python_path()
        if not python_input:
            raise QgsProcessingException("Custom mode selected, but Python path is empty.")
        try:
            python_bin = resolve_python_executable(python_input)
        except ValueError as exc:
            raise QgsProcessingException(str(exc)) from exc
        set_custom_python_path(python_input)
        return python_bin

    def _ensure_graph_file(self, parameters, context, param_name: str) -> str:
        graph_path = self.parameterAsFile(parameters, param_name, context)
        if not graph_path or not os.path.exists(graph_path):
            raise QgsProcessingException(f"Graph file not found: {graph_path}")
        return graph_path

    def _save_tmp_layer(self, layer, context, feedback, tmp_files: list[str]) -> str:
        tmp_path = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
        tmp_files.append(tmp_path)
        processing.run(
            "native:savefeatures",
            {"INPUT": layer, "OUTPUT": tmp_path},
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        return tmp_path

    def _run_bridge(self, python_bin: str, cmd: list[str], feedback):
        feedback.pushInfo(f"Running external Python: {python_bin}")
        proc = subprocess.run(cmd, text=True, capture_output=True, env=build_subprocess_env(python_bin))
        if proc.stdout:
            feedback.pushInfo(proc.stdout.strip())
        if proc.returncode != 0:
            raise QgsProcessingException(
                "Bridge execution failed.\n"
                f"Command: {' '.join(cmd)}\n"
                f"stderr:\n{proc.stderr}\n"
                f"stdout:\n{proc.stdout}"
            )

    def _finalize_output(self, parameters, context, feedback, tmp_result_path: str, layer_name: str = "result"):
        output_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT, context)
        if not output_uri:
            output_uri = "TEMPORARY_OUTPUT"

        saved = processing.run(
            "native:savefeatures",
            {
                "INPUT": f"{tmp_result_path}|layername={layer_name}",
                "OUTPUT": output_uri,
            },
            context=context,
            feedback=feedback,
            is_child_algorithm=True,
        )
        return {self.OUTPUT: saved["OUTPUT"]}


class GetRoadLengthAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    AREA = "AREA"

    def name(self):
        return "get_road_length"

    def displayName(self):
        return "Get Roads Length"

    def shortHelpString(self):
        return (
            "Calculate total road length for each territory polygon.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph method (.pkl)\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with road_length indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Road length by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        tmp_result = None
        try:
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "road_length",
                "--graph-path",
                graph_path,
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetRoadLengthAlgorithm()


class GetRoadDensityAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    AREA = "AREA"

    def name(self):
        return "get_road_density"

    def displayName(self):
        return "Get Roads Density"

    def shortHelpString(self):
        return (
            "Calculate road density (km/km2) for each territory polygon.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph method (.pkl)\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with density indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Road density by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "road_density",
                "--graph-path",
                graph_path,
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetRoadDensityAlgorithm()


class GetRegLengthAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    AREA = "AREA"

    def name(self):
        return "get_reg_length"

    def displayName(self):
        return "Get Roads Length by Type"

    def shortHelpString(self):
        return (
            "Calculate road length by road type (reg classes) for each territory polygon.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph method (.pkl)\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with length_reg_1, length_reg_2, length_reg_3 indicators"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Road length by reg status")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "reg_length",
                "--graph-path",
                graph_path,
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetRegLengthAlgorithm()


class GetRailwayLengthAlgorithm(_BridgeIndicatorBase):
    AREA = "AREA"
    RAILWAYS = "RAILWAYS"

    def name(self):
        return "get_railway_length"

    def displayName(self):
        return "Get Railways Length"

    def shortHelpString(self):
        return (
            "Calculate railway length for each territory polygon.\n\n"
            "Inputs:\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Railway paths (required columns: geometry (LineString or MultiLineString))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with railway_length_km indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.RAILWAYS,
                "Railway paths (required columns: geometry (LineString))",
                types=[QgsProcessing.TypeVectorLine],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Railway length by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        railways_layer = self.parameterAsVectorLayer(parameters, self.RAILWAYS, context)
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")
        if railways_layer is None:
            raise QgsProcessingException("Railways layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            railways_path = self._save_tmp_layer(railways_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "railway_length",
                "--area-path",
                area_path,
                "--railways-path",
                railways_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetRailwayLengthAlgorithm()


class GetConnectivityAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    AREA = "AREA"
    SETTLEMENTS = "SETTLEMENTS"

    def name(self):
        return "get_connectivity"

    def displayName(self):
        return "Get Connectivity"

    def shortHelpString(self):
        return (
            "Calculate median connectivity (minutes) for each territory polygon.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph or Get Intermodal Graph (.pkl)\n"
            "- Settlement points (required columns: name and geometry (Point))\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with connectivity indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SETTLEMENTS,
                "Settlement points (required columns: name and geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Connectivity by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        settlements_layer = self.parameterAsVectorLayer(parameters, self.SETTLEMENTS, context)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        if settlements_layer is None:
            raise QgsProcessingException("Settlement points layer is required.")
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            settlements_path = self._save_tmp_layer(settlements_layer, context, feedback, tmp_files)
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "connectivity",
                "--graph-path",
                graph_path,
                "--settlements-path",
                settlements_path,
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetConnectivityAlgorithm()


class GetServiceCountAlgorithm(_BridgeIndicatorBase):
    AREA = "AREA"
    SERVICE = "SERVICE"

    def name(self):
        return "get_service_count"

    def displayName(self):
        return "Get Servie Count"

    def shortHelpString(self):
        return (
            "Calculate number of services inside each territory polygon.\n\n"
            "Inputs:\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Service points (required columns: geometry (Point))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with service_number indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SERVICE,
                "Service points (required columns: name and geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Service count by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        service_layer = self.parameterAsVectorLayer(parameters, self.SERVICE, context)
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            service_path = None
            if service_layer is not None:
                service_path = self._save_tmp_layer(service_layer, context, feedback, tmp_files)

            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "service_count",
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            if service_path is not None:
                cmd.extend(["--service-path", service_path])

            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetServiceCountAlgorithm()


class GetServiceAccessibilityAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    AREA = "AREA"
    SETTLEMENTS = "SETTLEMENTS"
    SERVICE = "SERVICE"

    def name(self):
        return "get_service_accessibility"

    def displayName(self):
        return "Get Service Accessibility"

    def shortHelpString(self):
        return (
            "Calculate service accessibility (in minutes) for each territory polygon.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph or Get Intermodal Graph (.pkl)\n"
            "- Settlement points (required columns: name and geometry (Point))\n"
            "- Area polygons for aggregation (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Service points (required columns: geometry (Point))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with service_accessibility indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SETTLEMENTS,
                "Settlement points (required columns: name and geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA,
                "Area polygons (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SERVICE,
                "Service points (required columns: name and geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Service accessibility by area")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        settlements_layer = self.parameterAsVectorLayer(parameters, self.SETTLEMENTS, context)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA, context)
        service_layer = self.parameterAsVectorLayer(parameters, self.SERVICE, context)
        if settlements_layer is None:
            raise QgsProcessingException("Settlement points layer is required.")
        if area_layer is None:
            raise QgsProcessingException("Area polygons layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            settlements_path = self._save_tmp_layer(settlements_layer, context, feedback, tmp_files)
            area_path = self._save_tmp_layer(area_layer, context, feedback, tmp_files)
            service_path = None
            if service_layer is not None:
                service_path = self._save_tmp_layer(service_layer, context, feedback, tmp_files)

            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "service_accessibility",
                "--graph-path",
                graph_path,
                "--settlements-path",
                settlements_path,
                "--area-path",
                area_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            if service_path is not None:
                cmd.extend(["--service-path", service_path])

            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetServiceAccessibilityAlgorithm()


class GetTerrServiceCountAlgorithm(_BridgeIndicatorBase):
    TERRITORY = "TERRITORY"
    SERVICE = "SERVICE"

    def name(self):
        return "get_terr_service_count"

    def displayName(self):
        return "Get Service Count for Territory"

    def shortHelpString(self):
        return (
            "Calculate number of services for selected territory polygons.\n\n"
            "Inputs:\n"
            "- Territory geolayer (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Service points (required columns geometry (Point))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with number_of_service indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.TERRITORY,
                "Territory geolayer (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SERVICE,
                "Service points (required columns geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Service count by territory")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        territory_layer = self.parameterAsVectorLayer(parameters, self.TERRITORY, context)
        service_layer = self.parameterAsVectorLayer(parameters, self.SERVICE, context)
        if territory_layer is None:
            raise QgsProcessingException("Territory layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            territory_path = self._save_tmp_layer(territory_layer, context, feedback, tmp_files)
            service_path = None
            if service_layer is not None:
                service_path = self._save_tmp_layer(service_layer, context, feedback, tmp_files)

            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "terr_service_count",
                "--territory-path",
                territory_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            if service_path is not None:
                cmd.extend(["--service-path", service_path])
            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetTerrServiceCountAlgorithm()


class GetTerrServiceAccessibilityAlgorithm(_BridgeIndicatorBase):
    INPUT_GRAPH = "INPUT_GRAPH"
    TERRITORY = "TERRITORY"
    SERVICE = "SERVICE"

    def name(self):
        return "get_terr_service_accessibility"

    def displayName(self):
        return "Get Service Accessibility for Territory"

    def shortHelpString(self):
        return (
            "Calculate service accessibility (minutes) for selected territory polygons.\n\n"
            "Inputs:\n"
            "- Input graph obtained via Get Drive Graph or Get Intermodal Graph (.pkl)\n"
            "- Territory geolayer (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Service points (required columns: geometry (Point))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with service_accessibility indicator"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_GRAPH,
                "Input graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.TERRITORY,
                "Territory geolayer (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.SERVICE,
                "Service points (required columns: geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self._add_python_mode_params()
        self._add_output_param("Service accessibility by territory")

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        graph_path = self._ensure_graph_file(parameters, context, self.INPUT_GRAPH)
        territory_layer = self.parameterAsVectorLayer(parameters, self.TERRITORY, context)
        service_layer = self.parameterAsVectorLayer(parameters, self.SERVICE, context)
        if territory_layer is None:
            raise QgsProcessingException("Territory layer is required.")

        script_path = self._bridge_script_path()
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            territory_path = self._save_tmp_layer(territory_layer, context, feedback, tmp_files)
            service_path = None
            if service_layer is not None:
                service_path = self._save_tmp_layer(service_layer, context, feedback, tmp_files)

            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)
            cmd = [
                python_bin,
                script_path,
                "--operation",
                "terr_service_accessibility",
                "--graph-path",
                graph_path,
                "--territory-path",
                territory_path,
                "--output-path",
                tmp_result,
                "--output-layer",
                "result",
            ]
            if service_path is not None:
                cmd.extend(["--service-path", service_path])

            self._run_bridge(python_bin, cmd, feedback)
            return self._finalize_output(parameters, context, feedback, tmp_result)
        finally:
            for path in tmp_files:
                if os.path.exists(path):
                    try:
                        os.remove(path)
                    except OSError:
                        pass

    def createInstance(self):
        return GetTerrServiceAccessibilityAlgorithm()
