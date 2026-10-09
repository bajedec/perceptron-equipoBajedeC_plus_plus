"""Métricas para evaluar un clasificador binario."""
import numpy as np


def accuracy(y, y_pred):
    """Proporción de predicciones correctas (entre 0 y 1)."""
    return float(np.mean(np.asarray(y) == np.asarray(y_pred)))


def error_clasificacion(y, y_pred):
    """Proporción de predicciones incorrectas (entre 0 y 1)."""
    return 1.0 - accuracy(y, y_pred)


def matriz_confusion(y, y_pred):
    """Devuelve la matriz 2x2 [[TN, FP], [FN, TP]] como array de NumPy."""
    y_true = np.asarray(y)
    pred = np.asarray(y_pred)
    tn = int(np.sum((y_true == 0) & (pred == 0)))
    fp = int(np.sum((y_true == 0) & (pred == 1)))
    fn = int(np.sum((y_true == 1) & (pred == 0)))
    tp = int(np.sum((y_true == 1) & (pred == 1)))
    return [[tn, fp], [fn, tp]]
