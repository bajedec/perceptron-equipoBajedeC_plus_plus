"""Excepciones personalizadas para el proyecto del Perceptrón."""


class ModeloNoEntrenadoError(Exception):
    """Se lanza cuando se intenta predecir con un modelo que no ha sido entrenado."""
    pass