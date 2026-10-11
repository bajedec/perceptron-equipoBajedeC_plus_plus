# Clasificador de tumores con Perceptrón — versión 1.0.0

Proyecto de la **Evaluación Parcial** de Construcción de Software. Un Perceptrón implementado desde cero
con NumPy clasifica tumores de mama como **malignos (0)** o **benignos (1)** a partir de medidas de células.

## Datos

- `datos/pacientes.csv`: 569 pacientes, 10 medidas y la columna `diagnostico`.
  Proviene del dataset *Breast Cancer Wisconsin (Diagnostic)* (UCI). Se limpiaron valores nulos y se estandarizaron las características.
- `datos/hospital_b.csv`: 60 pacientes de otro hospital. Se utiliza para validar casos de varianza cero y división segura.

## Estructura

```text
├── datos/                 # archivos CSV
├── src/
│   ├── datos.py           # cargar, limpiar, estandarizar y dividir
│   ├── perceptron.py      # clase Perceptron con decaimiento
│   ├── metricas.py        # accuracy, error de clasificación y matriz de confusión
│   └── excepciones.py     # excepciones propias DatosInvalidosError y ModeloNoEntrenadoError
├── main.py                # pipeline refactorizado para entrenar y evaluar modelos
├── .github/workflows/ci.yml
├── CHANGELOG.md
└── requirements.txt
```

## Cómo ejecutarlo

```bash
pip install -r requirements.txt
python main.py --datos datos/pacientes.csv --objetivo diagnostico
```

También es posible seleccionar características específicas con el argumento opcional `--features`:

```bash
python main.py --features concavidad puntos_concavos area textura
```

## Integrantes

| Nombre | Código | Rol |
|---|---|---|
| SAUL ADAIN HUILLCA RODRIGUEZ | 74401464 | Líder |
| HUMBERTO ZAMORA MOSCOSO | - | Colaborador |
| DIEGO ISAIAS GUTIERREZ RUIZ | 77021535 | Colaborador |
| OSCAR JHOSUE PASTOR QUISPE | 73138388 | Colaborador |

## Resultados

A continuación se resumen las métricas obtenidas al evaluar los tres modelos sobre el conjunto de prueba (test split):

| Modelo | Configuración | Accuracy | Error de clasificación | Matriz de confusión `[[TN, FP], [FN, TP]]` |
|---|---|---|---|---|
| **Modelo 1** | Tasa 0.01, 30 épocas (Features base) | **0.867** | 0.133 | `[[34, 4], [11, 64]]` |
| **Modelo 2** | Tasa 0.5 con decaimiento 0.1 (Features base) | **0.876** | 0.124 | `[[37, 1], [13, 62]]` |
| **Modelo 3** | Tasa 0.01, 30 épocas (`concavidad`, `puntos_concavos`, `area`, `textura`) | **0.920** | 0.080 | `[[35, 3], [6, 69]]` |

### Análisis de resultados

#### 1. ¿Por qué los Modelos 1 y 2 daban igual antes de la Tarea 4, si sus tasas eran distintas?
Antes de la Tarea 4, ambos modelos empleaban una tasa de aprendizaje constante sin decaimiento y partían con los pesos inicializados en cero. En el Perceptrón clásico, multiplicar la regla de actualización por una constante fija escala la longitud del vector de pesos pero preserva la misma dirección geométrica del hiperplano separador ante las mismas muestras mal clasificadas. Por ello, la frontera de decisión final resultaba equivalente y predecía exactamente las mismas clases (accuracy 0.867). Al implementar una tasa con decaimiento dinámico por época ($\eta_t = \frac{\eta_0}{1 + d \cdot t}$), la magnitud de las correcciones varía a lo largo de las épocas, explorando una trayectoria diferente en el espacio de pesos que convergió a un hiperplano superior (accuracy 0.876).

#### 2. ¿Qué cambió con las features del Modelo 3 y por qué?
El Modelo 3 reemplazó las características por defecto (`radio`, `textura`, `perimetro`, `area`) por atributos morfológicos nucleares más distintivos: `concavidad`, `puntos_concavos`, `area` y `textura`. En patología celular, la concavidad y la severidad de las indentaciones en el contorno del núcleo son marcadores altamente discriminantes de malignidad celular, mucho más específicos que las medidas dimensionales simples de radio y perímetro. Esta selección incrementó la separación geométrica entre ambas clases en el espacio euclidiano, permitiendo al Perceptrón elevar su exactitud a **0.920** y disminuir el error de clasificación a tan solo **0.080**.

#### 3. Los errores por época nunca llegan a 0: ¿qué dice eso sobre si los datos son linealmente separables?
De acuerdo con el Teorema de Convergencia del Perceptrón de Rosenblatt, si un conjunto de entrenamiento es linealmente separable, el algoritmo garantiza converger a cero errores en un número finito de pasos. En nuestro entrenamiento, el número de errores por época oscila en cada iteración (entre 63 y 74 errores en el Modelo 1, y entre 36 y 44 en el Modelo 3) sin alcanzar jamás el valor 0. Esto demuestra formalmente que **los datos no son linealmente separables**, indicando que existe solapamiento entre pacientes con tumores benignos y malignos en ese espacio de características, lo que impide que un hiperplano plano perfecto los divida sin error.

#### 4. En este problema, ¿qué error es más grave, un falso positivo o un falso negativo?
En un contexto clínico oncológico, un **falso negativo** es sustancialmente más peligroso y crítico. Un falso negativo implica diagnosticar como benigno a un paciente que en realidad presenta un tumor maligno, lo que provocaría que no reciba el tratamiento oncológico oportuno y pondría directamente en peligro su vida. En contraposición, un falso positivo clasifica erróneamente un tejido sano como maligno, lo cual conlleva estrés emocional temporal y la realización de biopsias o pruebas confirmatorias adicionales, pero permite salvaguardar la salud del paciente. Por tanto, en aplicaciones médicas la prioridad diagnóstica es siempre minimizar los falsos negativos.
