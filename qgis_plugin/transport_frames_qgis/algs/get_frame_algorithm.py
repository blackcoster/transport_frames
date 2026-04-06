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
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
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


class GetFrameAlgorithm(QgsProcessingAlgorithm):
    INPUT_GRAPH = "INPUT_GRAPH"
    ADMIN_CENTERS = "ADMIN_CENTERS"
    AREA_BOUNDARY = "AREA_BOUNDARY"
    AREA_OSM_ID = "AREA_OSM_ID"
    REGIONS = "REGIONS"
    RUN_MODE = "RUN_MODE"
    PYTHON_BIN = "PYTHON_BIN"
    OUTPUT_GRAPH = "OUTPUT_GRAPH"
    OUTPUT_EDGES = "OUTPUT_EDGES"
    OUTPUT_NODES = "OUTPUT_NODES"

    def name(self):
        return "get_frame"

    def displayName(self):
        return "Get Weighted Frame"

    def group(self):
        return "3 - Frame"

    def groupId(self):
        return "frame"

    def shortHelpString(self):
        return (
            "Build transport frame and calculate road congestion based on road category and traffic volume.\n\n"
            "Inputs:\n"
            "- Input drive graph obtained via Get Drive Graph method (.pkl)\n"
            "- Administrative centers (required columns: name and geometry (Point))\n"
            "- Territory boundary (required columns: name and geometry (Polygon or MultiPolygon)) or OSM relation ID\n"
            "- Neighbor regions boundaries (required columns: name and geometry (Polygon or MultiPolygon))\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Outputs:\n"
            "- weighted_graph.pkl\n"
            "- edges layer\n"
            "- nodes layer"
        )

    def initAlgorithm(self, config=None):
        current_mode = get_mode()
        mode_default = 0 if current_mode == MODE_MANAGED else 1

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
                self.ADMIN_CENTERS,
                "Administrative centers (required columns: name and geometry (Point))",
                types=[QgsProcessing.TypeVectorPoint],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.AREA_BOUNDARY,
                "Territory boundary (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.AREA_OSM_ID,
                "Territory boundary OSM relation ID (alternative to boundary layer)",
                type=QgsProcessingParameterNumber.Integer,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.REGIONS,
                "Neighbor regions boundaries (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
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
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_GRAPH,
                "Output weighted graph file (.pkl)",
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_EDGES,
                "Output weighted frame edges",
                type=QgsProcessing.TypeVectorLine,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_NODES,
                "Output weighted frame nodes",
                type=QgsProcessing.TypeVectorPoint,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
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
                python_bin = resolve_python_executable(get_managed_python_path())
            except ValueError as exc:
                raise QgsProcessingException(str(exc)) from exc
        else:
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

        input_graph = self.parameterAsFile(parameters, self.INPUT_GRAPH, context)
        if not input_graph or not os.path.exists(input_graph):
            raise QgsProcessingException(f"Input graph file not found: {input_graph}")

        admin_layer = self.parameterAsVectorLayer(parameters, self.ADMIN_CENTERS, context)
        area_layer = self.parameterAsVectorLayer(parameters, self.AREA_BOUNDARY, context)
        regions_layer = self.parameterAsVectorLayer(parameters, self.REGIONS, context)

        area_osm_id_raw = self.parameterAsString(parameters, self.AREA_OSM_ID, context).strip()
        if area_osm_id_raw in {"", "None", "NULL"}:
            area_osm_id = None
        else:
            area_osm_id = int(area_osm_id_raw)

        if (area_layer is None and area_osm_id is None) or (area_layer is not None and area_osm_id is not None):
            raise QgsProcessingException(
                "Provide exactly one area boundary input: either polygon layer or OSM relation ID."
            )

        graph_path = self.parameterAsFileOutput(parameters, self.OUTPUT_GRAPH, context)
        edges_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT_EDGES, context)
        if not edges_uri or edges_uri.startswith("memory:"):
            raise QgsProcessingException("Please choose file-based output for frame edges (e.g., GeoPackage).")
        edges_path, edges_layer = self._parse_output_uri(edges_uri)
        nodes_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT_NODES, context)
        if not nodes_uri or nodes_uri.startswith("memory:"):
            raise QgsProcessingException("Please choose file-based output for frame nodes (e.g., GeoPackage).")
        nodes_path, nodes_layer = self._parse_output_uri(nodes_uri)

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bridge",
            "get_frame_bridge.py",
        )
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            def _save_tmp(layer_obj):
                tmp_path = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
                tmp_files.append(tmp_path)
                processing.run(
                    "native:savefeatures",
                    {"INPUT": layer_obj, "OUTPUT": tmp_path},
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
                return tmp_path

            admin_path = _save_tmp(admin_layer)
            regions_path = _save_tmp(regions_layer)
            area_path = _save_tmp(area_layer) if area_layer is not None else None

            cmd = [
                python_bin,
                script_path,
                "--input-graph",
                input_graph,
                "--admin-centers-path",
                admin_path,
                "--regions-path",
                regions_path,
                "--graph-out",
                graph_path,
                "--edges-out",
                edges_path,
                "--nodes-out",
                nodes_path,
            ]
            if area_path is not None:
                cmd.extend(["--area-path", area_path])
            else:
                cmd.extend(["--area-osm-id", str(area_osm_id)])
            if edges_layer:
                cmd.extend(["--edges-layer", edges_layer])
            if nodes_layer:
                cmd.extend(["--nodes-layer", nodes_layer])

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

            return {
                self.OUTPUT_GRAPH: graph_path,
                self.OUTPUT_EDGES: edges_uri,
                self.OUTPUT_NODES: nodes_uri,
            }
        finally:
            for tmp in tmp_files:
                if os.path.exists(tmp):
                    try:
                        os.remove(tmp)
                    except OSError:
                        pass

    def createInstance(self):
        return GetFrameAlgorithm()

    @staticmethod
    def _parse_output_uri(uri: str):
        if "|" not in uri:
            return uri, None

        parts = uri.split("|")
        path = parts[0]
        layer_name = None
        for token in parts[1:]:
            if token.startswith("layername="):
                layer_name = token.split("=", 1)[1]
                break
        return path, layer_name
