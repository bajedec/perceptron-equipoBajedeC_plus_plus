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
    """Devuelve una copia con los nulos de cada feature rellenados con su mediana."""
    datos = df.copy()
    for col in features:
        datos[col] = datos[col].fillna(datos[col].median())
    return datos


def estandarizar(X):
    """Resta la media y divide entre la desviación estándar.

    Si la desviación es 0, usa 1 para evitar división entre cero.
    """
    media = X.mean(axis=0)
    desv = X.std(axis=0)
 feature/excepciones
    desv[desv == 0] = 1

    desv = np.where(desv == 0, 1, desv)
 develop
    return (X - media) / desv


def dividir(X, y, prueba=0.2, semilla=42):
    """Mezcla las filas y separa entrenamiento y prueba."""
    idx = np.random.default_rng(semilla).permutation(len(X))
    n_prueba = int(len(X) * prueba)
    te, tr = idx[:n_prueba], idx[n_prueba:]
    return X[tr], X[te], y[tr], y[te]
