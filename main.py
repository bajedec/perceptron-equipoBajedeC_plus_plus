"""Entrena y evalúa los modelos de Perceptrón.

Uso:
    python main.py --datos datos/pacientes.csv --objetivo diagnostico
"""
import argparse
import sys

from src.datos import cargar_datos, limpiar, estandarizar, dividir
from src.perceptron import Perceptron
from src.metricas import accuracy, error_clasificacion, matriz_confusion
from src.excepciones import DatosInvalidosError

FEATURES_BASE = ["radio", "textura", "perimetro", "area"]
EPOCAS = 30


def evaluar_modelo(nombre, df, features, objetivo, modelo):
    """Prepara los datos, entrena el perceptrón e imprime las métricas."""
    datos = limpiar(df, features)
    X = estandarizar(datos[features].to_numpy(dtype=float))
    y = datos[objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)

    modelo.entrenar(X_tr, y_tr)
    y_pred = modelo.predecir(X_te)

    print(f"Modelo {nombre}")
    print("  accuracy:", round(accuracy(y_te, y_pred), 3))
    print("  error de clasificacion:", round(error_clasificacion(y_te, y_pred), 3))
    print("  matriz de confusion:", matriz_confusion(y_te, y_pred))
    if hasattr(modelo, "errores_por_epoca"):
        print("  errores por época:", modelo.errores_por_epoca[:10], "...\n")
    else:
        print()


def main():
    parser = argparse.ArgumentParser(description="Clasificador con Perceptrón")
    parser.add_argument("--datos", default="datos/pacientes.csv")
    parser.add_argument("--objetivo", default="diagnostico")
    parser.add_argument(
        "--features",
        nargs="+",
        default=["concavidad", "puntos_concavos", "area", "textura"]
    )
    args = parser.parse_args()

    df = cargar_datos(args.datos, args.objetivo)

    # ---------------- Modelo 1: básico ----------------
    modelo1 = Perceptron(tasa_aprendizaje=0.01, epocas=EPOCAS)
    evaluar_modelo("1 (tasa 0.01)", df, FEATURES_BASE, args.objetivo, modelo1)

    # ---------------- Modelo 2: con decaimiento ----------------
    modelo2 = Perceptron(tasa_aprendizaje=0.5, epocas=EPOCAS, decaimiento=0.1)
    evaluar_modelo(
        "2 (tasa 0.5 con decaimiento)", df, FEATURES_BASE, args.objetivo, modelo2
    )

    # ---------------- Modelo 3: otras features ----------------
    features_invalidas = [f for f in args.features if f not in df.columns]
    if features_invalidas:
        cols = " ".join(features_invalidas)
        raise DatosInvalidosError(f"Columnas inexistentes: {cols}")

    modelo3 = Perceptron(tasa_aprendizaje=0.01, epocas=EPOCAS)
    evaluar_modelo(
        f"3 (features: {', '.join(args.features)})",
        df,
        args.features,
        args.objetivo,
        modelo3,
    )


if __name__ == "__main__":
    try:
        main()
    except (DatosInvalidosError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
