import os
import tempfile

from qgis import processing
from qgis.core import (
    QgsProcessing,
    QgsProcessingException,
    QgsProcessingParameterBoolean,
    QgsProcessingParameterFile,
    QgsProcessingParameterVectorDestination,
    QgsProcessingParameterVectorLayer,
)

from .indicators_algorithms import _BridgeIndicatorBase


class GradeTerritoryAlgorithm(_BridgeIndicatorBase):
    INPUT_FRAME_GRAPH = "INPUT_FRAME_GRAPH"
    TERRITORIES = "TERRITORIES"
    INCLUDE_PRIORITY = "INCLUDE_PRIORITY"

    def name(self):
        return "grade_territory"

    def displayName(self):
        return "Grade Territory"

    def group(self):
        return "4 - Grade"

    def groupId(self):
        return "grade"

    def shortHelpString(self):
        return (
            "Calculate territory grades using weighted transport frame graph.\n\n"
            "Inputs:\n"
            "- Input weighted transport frame graph obtained via Get Weighted Transport Frame method (.pkl)\n"
            "- Territory geolayer (required columns: name and geometry (Polygon or MultiPolygon))\n"
            "- Include priority roads (boolean)\n\n"
            "Grade metrics (frame-based):\n"
            "- `reg = 1` means federal roads, `reg = 2` means regional roads.\n"
            "- Priority nodes are high-load frame nodes (top 40% by node weight within each class).\n"
            "- Grade is assigned by nearest-distance thresholds:\n"
            "  5.0  -> priority federal node within 5 km\n"
            "  4.5  -> federal node within 5 km, or priority federal < 10 km and priority regional < 5 km\n"
            "  4.0  -> federal < 10 km and regional < 5 km\n"
            "  3.5  -> priority federal < 100 km and priority regional < 5 km\n"
            "  3.0  -> federal < 100 km and regional < 5 km\n"
            "  2.0  -> federal > 100 km and regional < 5 km\n"
            "  1.0  -> no close nodes, but nearest graph edge within 5 km\n"
            "  0.0  -> otherwise\n"
            "- If `Include priority roads = False`, priority-based steps are skipped.\n\n"
            "Python mode:\n"
            "- Managed (recommended): uses plugin-managed environment in QGIS profile.\n"
            "- Custom: uses provided Python path.\n\n"
            "Output:\n"
            "- Geolayer with grade indicators"
        )

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterFile(
                self.INPUT_FRAME_GRAPH,
                "Input weighted frame graph file (.pkl)",
                behavior=QgsProcessingParameterFile.File,
                fileFilter="Pickle files (*.pkl)",
            )
        )
        self.addParameter(
            QgsProcessingParameterVectorLayer(
                self.TERRITORIES,
                "Territories geolayer (required columns: name and geometry (Polygon or MultiPolygon))",
                types=[QgsProcessing.TypeVectorPolygon],
                optional=False,
            )
        )
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.INCLUDE_PRIORITY,
                "Include priority roads",
                defaultValue=True,
            )
        )
        self._add_python_mode_params()
        self.addParameter(
            QgsProcessingParameterVectorDestination(
                self.OUTPUT,
                "Graded territories",
                type=QgsProcessing.TypeVectorPolygon,
            )
        )

    def processAlgorithm(self, parameters, context, feedback):
        python_bin = self._resolve_python_bin(parameters, context)
        frame_graph_path = self._ensure_graph_file(parameters, context, self.INPUT_FRAME_GRAPH)
        territories_layer = self.parameterAsVectorLayer(parameters, self.TERRITORIES, context)
        include_priority = self.parameterAsBool(parameters, self.INCLUDE_PRIORITY, context)
        if territories_layer is None:
            raise QgsProcessingException("Territories layer is required.")

        script_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "bridge",
            "criteria_bridge.py",
        )
        if not os.path.exists(script_path):
            raise QgsProcessingException(f"Bridge script not found: {script_path}")

        tmp_files = []
        try:
            territories_path = self._save_tmp_layer(territories_layer, context, feedback, tmp_files)
            tmp_result = tempfile.NamedTemporaryFile(suffix=".gpkg", delete=False).name
            tmp_files.append(tmp_result)

            cmd = [
                python_bin,
                script_path,
                "--operation",
                "grade_territory",
                "--frame-graph-path",
                frame_graph_path,
                "--territories-path",
                territories_path,
                "--include-priority",
                "1" if include_priority else "0",
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
        return GradeTerritoryAlgorithm()
