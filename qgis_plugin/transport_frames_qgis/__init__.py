def classFactory(iface):
    from .plugin import TransportFramesPlugin
    return TransportFramesPlugin(iface)