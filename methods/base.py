import abc

class TrackerMethod(abc.ABC):
    """Classe de base pour toutes les méthodes de tracking."""

    @abc.abstractmethod
    def __init__(self, init_frame, roi):
        """
        init_frame: image initiale
        roi: tuple (x, y, w, h)
        """
        pass

    @abc.abstractmethod
    def update(self, frame):
        """
        Doit retourner (bbox, score)
        bbox: (x, y, w, h) ou None si non trouvé
        score: confiance ou score de similarité
        """
        pass
