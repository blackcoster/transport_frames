from qgis.core import (
    QgsProcessingAlgorithm,
    QgsProcessingOutputString,
    QgsProcessingParameterBoolean,
)

from ..env_manager import (
    MODE_MANAGED,
    get_custom_python_path,
    get_managed_env_dir,
    get_managed_python_path,
    get_mode,
    has_managed_python,
)


class EnvironmentStatusAlgorithm(QgsProcessingAlgorithm):
    LOG_TO_FEEDBACK = "LOG_TO_FEEDBACK"

    def name(self):
        return "environment_status"

    def displayName(self):
        return "Environment Status"

    def group(self):
        return "1 - Environment"

    def groupId(self):
        return "environment"

    def shortHelpString(self):
        return "Show current plugin environment mode and saved Python paths."

    def initAlgorithm(self, config=None):
        self.addParameter(
            QgsProcessingParameterBoolean(
                self.LOG_TO_FEEDBACK,
                "Print status to execution log",
                defaultValue=True,
            )
        )
        self.addOutput(QgsProcessingOutputString("mode", "Current mode"))
        self.addOutput(QgsProcessingOutputString("managed_env_dir", "Managed environment directory"))
        self.addOutput(QgsProcessingOutputString("managed_python", "Managed Python executable"))
        self.addOutput(QgsProcessingOutputString("managed_ready", "Managed environment ready"))
        self.addOutput(QgsProcessingOutputString("custom_python", "Custom Python path"))

    def processAlgorithm(self, parameters, context, feedback):
        log_to_feedback = self.parameterAsBool(parameters, self.LOG_TO_FEEDBACK, context)

        mode = get_mode()
        managed_env_dir = get_managed_env_dir()
        managed_python = get_managed_python_path()
        custom_python = get_custom_python_path()
        managed_ready = has_managed_python()

        if log_to_feedback:
            feedback.pushInfo(f"Mode: {mode}")
            feedback.pushInfo(f"Managed env dir: {managed_env_dir}")
            feedback.pushInfo(f"Managed python: {managed_python}")
            feedback.pushInfo(f"Managed env ready: {managed_ready}")
            feedback.pushInfo(f"Custom python: {custom_python or '<empty>'}")
            if mode == MODE_MANAGED and not managed_ready:
                feedback.reportError(
                    "Managed mode is selected but managed environment is missing. "
                    "Run 'Setup Python Environment'."
                )

        return {
            "mode": mode,
            "managed_env_dir": managed_env_dir,
            "managed_python": managed_python,
            "managed_ready": str(managed_ready),
            "custom_python": custom_python,
        }

    def createInstance(self):
        return EnvironmentStatusAlgorithm()
