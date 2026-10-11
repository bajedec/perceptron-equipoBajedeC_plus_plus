# Clasificador de tumores con Perceptrón — versión 0.9.0

Proyecto base de la **Evaluación Parcial** de Construcción de Software. Un Perceptrón implementado desde cero
con NumPy clasifica tumores de mama como **malignos (0)** o **benignos (1)** a partir de medidas de las células.

## Datos

- `datos/pacientes.csv`: 569 pacientes, 10 medidas y la columna `diagnostico`.
  Proviene del dataset *Breast Cancer Wisconsin (Diagnostic)* (UCI). Se borraron algunos valores para practicar la limpieza.
- `datos/hospital_b.csv`: 60 pacientes de otro hospital. Se usa para reproducir el error del Issue de la Tarea 1.

## Estructura

```text
├── datos/                 # archivos CSV
├── src/
│   ├── datos.py           # cargar, limpiar, estandarizar y dividir
│   ├── perceptron.py      # clase Perceptron
│   └── metricas.py        # accuracy y otras métricas
├── main.py                # entrena y evalúa los modelos
├── .github/workflows/ci.yml
├── CHANGELOG.md
└── requirements.txt
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py --datos datos/pacientes.csv --objetivo diagnostico
```

## Integrantes

| Nombre | Código | Rol |
|---|---|---|
| SAUL ADAIN HUILLCA RODRIGUEZ | - | Líder |
| HUMBERTO ZAMORA MOSCOSO |  |  |
| DIEGO ISAIAS GUTIERREZ RUIZ | 77021535 |  |
| OSCAR JHOSUE PASTOR QUISPE |  |  |

## Resultados


_Se completa en la Tarea 7._
