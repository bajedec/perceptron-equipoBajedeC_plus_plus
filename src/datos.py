"""Carga y preparación de los datos para el Perceptrón."""
import os
import numpy as np
import pandas as pd

from src.excepciones import DatosInvalidosError


def cargar_datos(ruta, objetivo):
    """Lee el CSV y devuelve el DataFrame.

    `objetivo` es el nombre de la columna con la clase (0 o 1).
    """
    if not os.path.exists(ruta):
        raise DatosInvalidosError(f"No existe el archivo {ruta}")

    df = pd.read_csv(ruta)

    if objetivo not in df.columns:
        raise DatosInvalidosError(f"No existe la columna objetivo '{objetivo}'")

    if df[objetivo].nunique() != 2:
        raise DatosInvalidosError(f"La columna '{objetivo}' no es binaria")

    return df


def limpiar(df, features):
    """Elimina filas con NaN en las columnas indicadas."""
    return df.dropna(subset=features)


def estandarizar(X):
    """Resta la media y divide entre la desviación estándar.

    Si la desviación es 0, usa 1 para evitar división entre cero.
    """
    media = X.mean(axis=0)
    desv = X.std(axis=0)
    desv[desv == 0] = 1
    return (X - media) / desv


def dividir(X, y, proporcion=0.7, semilla=42):
    """Divide en entrenamiento y prueba."""
    np.random.seed(semilla)
    n = len(X)
    indices = np.random.permutation(n)
    n_ent = int(n * proporcion)
    X_tr, X_te = X[indices[:n_ent]], X[indices[n_ent:]]
    y_tr, y_te = y[indices[:n_ent]], y[indices[n_ent:]]
    return X_tr, X_te, y_tr, y_te
