import os
import subprocess
import tempfile

from qgis import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingAlgorithm,
    QgsProcessingException,
    QgsProcessingParameterFile,
    QgsProcessingParameterFileDestination,
    QgsProcessingParameterNumber,
    QgsProcessingParameterString,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)


class GetGraphAlgorithm(QgsProcessingAlgorithm):
    OSM_ID = "OSM_ID"
    TERRITORY = "TERRITORY"
    BUFFER = "BUFFER"
    PYTHON_BIN = "PYTHON_BIN"
    OUTPUT_GRAPH = "OUTPUT_GRAPH"
    OUTPUT_EDGES = "OUTPUT_EDGES"

    def name(self):
        return "get_graph"

    def displayName(self):
        return "Get Drive Graph"

    def group(self):
        return "Graph"

    def groupId(self):
        return "graph"

    def shortHelpString(self):
        return (
            "Build drive graph via external Python 3.11+ environment.\n\n"
            "You must provide exactly one input:\n"
            "- OSM relation ID, or\n"
            "- Territory polygon layer.\n\n"
            "You can pass either:\n"
            "- direct executable path (e.g. .../.venv/bin/python), or\n"
            "- virtual environment folder (e.g. .../.venv).\n\n"
            "This algorithm runs a bridge script in external Python where "
            "transport_frames and its dependencies are installed."
        )

    def initAlgorithm(self, config=None):
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
                "Territory polygons",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=True,
            )
        )
        self.addParameter(
            QgsProcessingParameterNumber(
                self.BUFFER,
                "Buffer (meters)",
                type=QgsProcessingParameterNumber.Integer,
                defaultValue=3000,
                minValue=0,
            )
        )
        self.addParameter(
            QgsProcessingParameterString(
                self.PYTHON_BIN,
                "External Python executable (3.11+) or venv folder",
                defaultValue="",
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterFileDestination(
                self.OUTPUT_GRAPH,
                "Output graph file (.pkl)",
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT_EDGES,
                "Output graph edges",
                type=QgsProcessing.TypeVectorLine,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        python_input = self.parameterAsString(parameters, self.PYTHON_BIN, context).strip().strip('"').strip("'")
        if not python_input:
            raise QgsProcessingException("External Python path is required.")
        python_bin = self._resolve_python_path(python_input)

        osm_id_raw = self.parameterAsString(parameters, self.OSM_ID, context).strip()
        if osm_id_raw in {"", "None", "NULL"}:
            osm_id = None
        else:
            osm_id = int(osm_id_raw)

        territory_layer = self.parameterAsVectorLayer(parameters, self.TERRITORY, context)

        if (osm_id is None and territory_layer is None) or (osm_id is not None and territory_layer is not None):
            raise QgsProcessingException("Provide exactly one input: either OSM relation ID or territory polygons.")

        buffer_dist = self.parameterAsInt(parameters, self.BUFFER, context)
        graph_path = self.parameterAsFileOutput(parameters, self.OUTPUT_GRAPH, context)
        edges_uri = self.parameterAsOutputLayer(parameters, self.OUTPUT_EDGES, context)
        if not edges_uri or edges_uri.startswith("memory:"):
            raise QgsProcessingException("Please choose file-based output for graph edges (e.g., GeoPackage).")
        edges_path, edges_layer = self._parse_output_uri(edges_uri)

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bridge",
            "get_graph_bridge.py",
        )
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_territory = None
        try:
            cmd = [
                python_bin,
                script_path,
                "--buffer",
                str(buffer_dist),
                "--graph-out",
                graph_path,
                "--edges-out",
                edges_path,
            ]
            if edges_layer:
                cmd.extend(["--edges-layer", edges_layer])

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
            proc = subprocess.run(cmd, text=True, capture_output=True, env=self._build_subprocess_env(python_bin))
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
            }
        finally:
            if tmp_territory and os.path.exists(tmp_territory):
                try:
                    os.remove(tmp_territory)
                except OSError:
                    pass

    def createInstance(self):
        return GetGraphAlgorithm()

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

    @staticmethod
    def _resolve_python_path(path: str) -> str:
        path = path.strip().strip('"').strip("'")
        path = os.path.expanduser(path)
        if not os.path.isabs(path) and os.path.exists(os.path.sep + path):
            path = os.path.sep + path

        if os.path.isdir(path):
            candidates = [
                os.path.join(path, "bin", "python"),
                os.path.join(path, "bin", "python3"),
                os.path.join(path, "bin", "python3.11"),
            ]
            for candidate in candidates:
                if os.path.exists(candidate):
                    path = candidate
                    break
            else:
                raise QgsProcessingException(
                    "Provided folder does not contain Python executable in `bin/`.\n"
                    f"Folder: {path}"
                )

        if not os.path.exists(path):
            raise QgsProcessingException(
                "Python executable does not exist.\n"
                f"Path: {path}\n"
                "If path is correct, check macOS privacy access for QGIS."
            )
        if not os.access(path, os.X_OK):
            raise QgsProcessingException(
                "File is not executable.\n"
                f"Path: {path}"
            )
        return path

    @staticmethod
    def _build_subprocess_env(python_bin: str):
        env = os.environ.copy()

        # QGIS runtime exports PYTHONHOME/PYTHONPATH for its bundled Python.
        # If inherited, external Python from venv may fail at startup
        # (e.g. "No module named encodings").
        for var in (
            "PYTHONHOME",
            "PYTHONPATH",
            "PYTHONEXECUTABLE",
            "__PYVENV_LAUNCHER__",
        ):
            env.pop(var, None)

        venv_bin = os.path.dirname(python_bin)
        env["PATH"] = venv_bin + os.pathsep + env.get("PATH", "")
        return env
