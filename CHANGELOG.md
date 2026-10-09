# Changelog

Formato: cada versión lista lo que se agregó (`Agregado`), lo que se corrigió (`Corregido`) y lo que cambió (`Cambiado`).

## [0.9.1]

### Corregido
- Se corrigió la división entre cero en `estandarizar()` cuando una columna tiene desviación estándar 0 (por ejemplo, la columna `textura` en `hospital_b.csv`).

## [0.9.0]

### Agregado
- Carga, limpieza, estandarización y división de los datos.
- Perceptrón simple con entrenamiento por épocas.
- Modelos 1 y 2 en `main.py`, evaluados con accuracy.