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
    datos = limpiar(df, FEATURES_BASE)
    X = estandarizar(datos[FEATURES_BASE].to_numpy(dtype=float))
    y = datos[args.objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    modelo1 = Perceptron(tasa_aprendizaje=0.01, epocas=30)
    modelo1.entrenar(X_tr, y_tr)
    y_pred1 = modelo1.predecir(X_te)
    print("Modelo 1 (tasa 0.01)")
    print("  accuracy:", round(accuracy(y_te, y_pred1), 3))
    print("  error de clasificacion:", round(error_clasificacion(y_te, y_pred1), 3))
    print("  matriz de confusion:", matriz_confusion(y_te, y_pred1))
    print("  errores por época:", modelo1.errores_por_epoca[:10], "...\n")

    # ---------------- Modelo 2: con decaimiento ----------------
    datos = limpiar(df, FEATURES_BASE)
    X = estandarizar(datos[FEATURES_BASE].to_numpy(dtype=float))
    y = datos[args.objetivo].to_numpy()
    X_tr, X_te, y_tr, y_te = dividir(X, y)
    modelo2 = Perceptron(tasa_aprendizaje=0.5, epocas=30, decaimiento=0.1)
    modelo2.entrenar(X_tr, y_tr)
    y_pred2 = modelo2.predecir(X_te)
    print("Modelo 2 (tasa 0.5 con decaimiento)")
    print("  accuracy:", round(accuracy(y_te, y_pred2), 3))
    print("  error de clasificacion:", round(error_clasificacion(y_te, y_pred2), 3))
    print("  matriz de confusion:", matriz_confusion(y_te, y_pred2))
    print("  errores por época:", modelo2.errores_por_epoca[:10], "...")

    # ---------------- Modelo 3: otras features ----------------
    features_invalidas = [f for f in args.features if f not in df.columns]
    if features_invalidas:
        raise DatosInvalidosError(f"Columnas inexistentes: {' '.join(features_invalidas)}")

    datos3 = limpiar(df, args.features)
    X3 = estandarizar(datos3[args.features].to_numpy(dtype=float))
    X3_tr, X3_te, y3_tr, y3_te = dividir(X3, y)
    modelo3 = Perceptron(tasa_aprendizaje=0.01, epocas=30)
    modelo3.entrenar(X3_tr, y3_tr)
    y_pred3 = modelo3.predecir(X3_te)
    print(f"Modelo 3 (features: {', '.join(args.features)})")
    print("  accuracy:", round(accuracy(y3_te, y_pred3), 3))
    print("  error_clasificacion:", round(error_clasificacion(y3_te, y_pred3), 3))
    print("  matriz_confusion:", matriz_confusion(y3_te, y_pred3))


if __name__ == "__main__":
    try:
        main()
    except (DatosInvalidosError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)
