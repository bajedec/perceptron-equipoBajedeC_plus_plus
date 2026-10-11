# Changelog

Formato: cada versión lista lo que se agregó (`Agregado`), lo que se corrigió (`Corregido`) y lo que cambió (`Cambiado`).

## [1.0.0]

### Agregado
- Excepciones personalizadas `DatosInvalidosError` y `ModeloNoEntrenadoError` para un control robusto de fallos (Tarea 2).
- Validaciones en `cargar_datos()` para archivo inexistente, columna objetivo faltante y variables no binarias (Tarea 2).
- Validaciones en `Perceptron` para tasa de aprendizaje y épocas válidas, y verificación en `predecir()` antes de entrenar (Tarea 2).
- Implementación de las métricas `error_clasificacion` y `matriz_confusion` en `src/metricas.py` (Tarea 3).
- Parámetro de `decaimiento` en el `Perceptron` para decrecer la tasa de aprendizaje en cada época (Tarea 4).
- Argumento `--features` en `main.py` para permitir la selección flexible de variables explicativas con validación de columnas (Tarea 5).
- Modelo 3 entrenado con características morfológicas discriminantes (`concavidad`, `puntos_concavos`, `area`, `textura`) (Tarea 5).

### Cambiado
- Manejo de excepciones en el punto de entrada `main.py` mostrando mensajes limpios y saliendo con código 1 sin trazas de error (Tarea 2).
- Presentación completa de métricas en consola con accuracy, error de clasificación y matriz de confusión para todos los modelos (Tarea 3).
- Configuración del Modelo 2 a tasa 0.5 con decaimiento de 0.1, logrando un mejor rendimiento predictivo (Tarea 4).
- Refactorización de `main.py` extrayendo la preparación, entrenamiento y evaluación repetida en la función `evaluar_modelo()` (Tarea 6).
- Reemplazo del número mágico 30 por la constante global `EPOCAS = 30` (Tarea 6).
- Limpieza de dependencias e imports no utilizados y verificación de conformidad con el estándar PEP 8 (Tarea 6).

## [0.9.1]

### Corregido
- Se corrigió la división entre cero en `estandarizar()` cuando una columna tiene desviación estándar 0 (por ejemplo, la columna `textura` en `hospital_b.csv`).

## [0.9.0]

### Agregado
- Carga, limpieza, estandarización y división de los datos.
- Perceptrón simple con entrenamiento por épocas.
- Modelos 1 y 2 en `main.py`, evaluados con accuracy.