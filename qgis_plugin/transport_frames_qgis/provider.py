from qgis.core import QgsProcessingProvider
from .algs.add_roads_algorithm import AddRoadsAlgorithm
from .algs.criteria_algorithms import GradeTerritoryAlgorithm
from .algs.environment_status_algorithm import EnvironmentStatusAlgorithm
from .algs.get_frame_algorithm import GetFrameAlgorithm
from .algs.get_graph_algorithm import GetGraphAlgorithm
from .algs.get_intermodal_graph_algorithm import GetIntermodalGraphAlgorithm
from .algs.indicators_algorithms import (
    GetConnectivityAlgorithm,
    GetRailwayLengthAlgorithm,
    GetRegLengthAlgorithm,
    GetRoadDensityAlgorithm,
    GetRoadLengthAlgorithm,
    GetServiceAccessibilityAlgorithm,
    GetServiceCountAlgorithm,
    GetTerrServiceAccessibilityAlgorithm,
    GetTerrServiceCountAlgorithm,
)
from .algs.setup_environment_algorithm import SetupEnvironmentAlgorithm


class TransportFramesProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(SetupEnvironmentAlgorithm())
        self.addAlgorithm(EnvironmentStatusAlgorithm())
        self.addAlgorithm(GetGraphAlgorithm())
        self.addAlgorithm(AddRoadsAlgorithm())
        self.addAlgorithm(GetFrameAlgorithm())
        self.addAlgorithm(GetIntermodalGraphAlgorithm())
        self.addAlgorithm(GradeTerritoryAlgorithm())
        self.addAlgorithm(GetRoadLengthAlgorithm())
        self.addAlgorithm(GetRoadDensityAlgorithm())
        self.addAlgorithm(GetRegLengthAlgorithm())
        self.addAlgorithm(GetRailwayLengthAlgorithm())
        self.addAlgorithm(GetConnectivityAlgorithm())
        self.addAlgorithm(GetServiceCountAlgorithm())
        self.addAlgorithm(GetServiceAccessibilityAlgorithm())
        self.addAlgorithm(GetTerrServiceCountAlgorithm())
        self.addAlgorithm(GetTerrServiceAccessibilityAlgorithm())

    def id(self):
        return "transport_frames"

    def name(self):
        return "Transport Frames"

    def longName(self):
        return self.name()
