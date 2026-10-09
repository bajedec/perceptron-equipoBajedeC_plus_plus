"""Excepciones propias del proyecto."""

class DatosInvalidosError(Exception):
    """Se lanza cuando los datos de entrada no son válidos."""
    pass

class ModeloNoEntrenadoError(Exception):
    """Se lanza cuando se intenta predecir con un modelo sin entrenar."""
    pass
