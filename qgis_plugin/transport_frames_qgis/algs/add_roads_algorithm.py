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


class AddRoadsAlgorithm(QgsProcessingAlgorithm):
    INPUT_GRAPH = "INPUT_GRAPH"
    NEW_ROADS = "NEW_ROADS"
    RUN_MODE = "RUN_MODE"
    PYTHON_BIN = "PYTHON_BIN"
    OUTPUT_GRAPH = "OUTPUT_GRAPH"
    OUTPUT_EDGES = "OUTPUT_EDGES"
    OUTPUT_NODES = "OUTPUT_NODES"

    def name(self):
        return "add_roads"

    def displayName(self):
        return "Add Roads"

    def group(self):
        return "2 - Graph"

    def groupId(self):
        return "graph"

    def shortHelpString(self):
        return (
            "Add new road lines to an existing drive graph.\n\n"
            "Inputs:\n"
            "- Input drive graph obtained via Get Drive Graph method (.pkl)\n"
            "- New roads geolayer (required columns: geometry (LineString or MultiLineString), reg)\n\n"
            "Notes:\n"
            "- `reg` should contain road class values (typically 1, 2, 3).\n"
            "- Local CRS is read automatically from the input graph and applied internally.\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Outputs:\n"
            "- updated_graph.pkl\n"
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
                self.NEW_ROADS,
                "New roads geolayer (required columns: geometry and reg)",
                types=[QgsProcessing.TypeVectorLine],
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
                "Output updated graph file (.pkl)",
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_EDGES,
                "Output updated graph edges",
                type=QgsProcessing.TypeVectorLine,
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_NODES,
                "Output updated graph nodes",
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

        graph_path = self.parameterAsFile(parameters, self.INPUT_GRAPH, context)
        if not graph_path or not os.path.exists(graph_path):
            raise QgsProcessingException(f"Input graph file not found: {graph_path}")

        roads_layer = self.parameterAsVectorLayer(parameters, self.NEW_ROADS, context)
        if roads_layer is None:
            raise QgsProcessingException("New roads layer is required.")

        output_graph_path = self.parameterAsFileOutput(parameters, self.OUTPUT_GRAPH, context)
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
            "add_roads_bridge.py",
        )
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_roads = None
        try:
            tmp_roads = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            processing.run(
                "native:savefeatures",
                {
                    "INPUT": roads_layer,
                    "OUTPUT": tmp_roads,
                },
                context=context,
                feedback=feedback,
                is_child_algorithm=True,
            )

            cmd = [
                python_bin,
                script_path,
                "--input-graph",
                graph_path,
                "--new-roads-path",
                tmp_roads,
                "--graph-out",
                output_graph_path,
                "--edges-out",
                edges_path,
                "--nodes-out",
                nodes_path,
            ]
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
                self.OUTPUT_GRAPH: output_graph_path,
                self.OUTPUT_EDGES: edges_uri,
                self.OUTPUT_NODES: nodes_uri,
            }
        finally:
            if tmp_roads and os.path.exists(tmp_roads):
                try:
                    os.remove(tmp_roads)
                except OSError:
                    pass

    def createInstance(self):
        return AddRoadsAlgorithm()

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
