import os
import subprocess
import tempfile

from qgis import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterEnum,
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


class GetIntermodalGraphAlgorithm(QgsProcessingAlgorithm):
    OSM_ID = "OSM_ID"
    TERRITORY = "TERRITORY"
    RUN_MODE = "RUN_MODE"
    PYTHON_BIN = "PYTHON_BIN"
    OUTPUT_GRAPH = "OUTPUT_GRAPH"
    OUTPUT_EDGES = "OUTPUT_EDGES"
    OUTPUT_NODES = "OUTPUT_NODES"

    def name(self):
        return "get_intermodal_graph"

    def displayName(self):
        return "Get Intermodal Graph"

    def group(self):
        return "2 - Graph"

    def groupId(self):
        return "graph"

    def shortHelpString(self):
        return (
            "Build intermodal graph (public transport routes and walk paths) for territory.\n\n"
            "You must provide exactly one input:\n"
            "- OSM relation ID, or\n"
            "- Territory boundary (required columns: name and geometry (Polygon or MultiPolygon)).\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Outputs:\n"
            "- graph.pkl\n"
            "- edges layer\n"
            "- nodes layer"
        )

    def initAlgorithm(self, config=None):
        current_mode = get_mode()
        mode_default = 0 if current_mode == MODE_MANAGED else 1

        self.addParameter(
            QgsProcessingParameterNumber(
                self.OSM_ID,
                "OSM relation ID",
                type=QgsProcessingParameterNumber.Integer,
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.TERRITORY,
                "Territory boundary (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
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
                "Output intermodal graph file (.pkl)",
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_EDGES,
                "Output intermodal graph edges",
                type=QgsProcessing.TypeVectorLine,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_NODES,
                "Output intermodal graph nodes",
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

        osm_id_raw = self.parameterAsString(parameters, self.OSM_ID, context).strip()
        if osm_id_raw in {"", "None", "NULL"}:
            osm_id = None
        else:
            osm_id = int(osm_id_raw)

        territory_layer = self.parameterAsVectorLayer(parameters, self.TERRITORY, context)

        if (osm_id is None and territory_layer is None) or (osm_id is not None and territory_layer is not None):
            raise QgsProcessingException(
                "Provide exactly one input: either OSM relation ID or Territory boundary layer."
            )

        graph_path = self.parameterAsFileOutput(parameters, self.OUTPUT_GRAPH, context)
        edges_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT_EDGES, context)
        if not edges_uri or edges_uri.startswith("memory:"):
            raise QgsProcessingException("Please choose file-based output for graph edges (e.g., GeoPackage).")
        edges_path, edges_layer = self._parse_output_uri(edges_uri)
        nodes_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT_NODES, context)
        if not nodes_uri or nodes_uri.startswith("memory:"):
            raise QgsProcessingException("Please choose file-based output for graph nodes (e.g., GeoPackage).")
        nodes_path, nodes_layer = self._parse_output_uri(nodes_uri)

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bridge",
            "get_intermodal_graph_bridge.py",
        )
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_territory = None
        try:
            cmd = [
                python_bin,
                script_path,
                "--graph-out",
                graph_path,
                "--edges-out",
                edges_path,
                "--nodes-out",
                nodes_path,
            ]
            if edges_layer:
                cmd.extend(["--edges-layer", edges_layer])
            if nodes_layer:
                cmd.extend(["--nodes-layer", nodes_layer])

            if osm_id is not None:
                cmd.extend(["--osm-id", str(osm_id)])
            else:
                tmp_territory = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
                processing.run(
                    "native:savefeatures",
                    {
                        "INPUT": territory_layer,
                        "OUTPUT": tmp_territory,
                    },
                    context=context,
                    feedback=feedback,
                    is_child_algorithm=True,
                )
                cmd.extend(["--territory-path", tmp_territory])

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
            if tmp_territory and os.path.exists(tmp_territory):
                try:
                    os.remove(tmp_territory)
                except OSError:
                    pass

    def createInstance(self):
        return GetIntermodalGraphAlgorithm()

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
