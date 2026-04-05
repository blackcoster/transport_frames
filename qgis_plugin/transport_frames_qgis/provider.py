from qgis.core import QgsProcessingProvider
from .algs.get_graph_algorithm import GetGraphAlgorithm


class TransportFramesProvider(QgsProcessingProvider):
    def loadAlgorithms(self):
        self.addAlgorithm(GetGraphAlgorithm())

    def id(self):
        return "transport_frames"

    def name(self):
        return "Transport Frames"

    def longName(self):
        return self.name()
