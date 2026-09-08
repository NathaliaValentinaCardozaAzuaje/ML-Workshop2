# Workshop 2 — Machine Learning

Pipelines, entrenamiento, comparación de modelos y validación cruzada aplicados a un problema de **regresión** y uno de **clasificación**.

Universidad EAFIT — Machine Learning, Unidad 2.

## Objetivo

Aplicar el ciclo completo de un proyecto de ML supervisado: limpieza y preprocesamiento, partición train/validation/test, construcción de pipelines con `ColumnTransformer`, entrenamiento y comparación de múltiples algoritmos, y validación cruzada. Cada decisión metodológica —qué se limpia, cómo se transforma, qué se escala— está justificada en celdas de texto, no solo ejecutada.

## Contenido del repositorio

```
├── flightprice_workshop2.ipynb   # Regresión — predicción de precios de vuelos
├── thyroid_workshop2.ipynb       # Clasificación — recurrencia de cáncer de tiroides
└── data/
    ├── Clean_Dataset.csv         # Flight Price Prediction (24 MB)
    └── Thyroid_Diff.csv          # Thyroid Disease Data (44 KB)
```

Ambos notebooks están **ejecutados**: todas las salidas, tablas y gráficas visibles en GitHub corresponden a una corrida real y completa.

## Datasets

| | Regresión | Clasificación |
|---|---|---|
| **Fuente** | [Flight Price Prediction](https://www.kaggle.com/datasets/shubhambathwal/flight-price-prediction) | [Thyroid Disease Data](https://www.kaggle.com/datasets/jainaru/thyroid-disease-data/data) |
| **Target** | `price` (continuo, rupias) | `Recurred` (binario, Yes/No) |
| **Registros** | 300,153 × 11 | 383 × 17 → **364** tras eliminar 19 duplicados |
| **Naturaleza** | 2 numéricas, 8 categóricas | 1 numérica (`Age`), 16 categóricas |
| **Particularidad** | Precio bimodal por clase de cabina | Desbalance 70.3% / 29.7% |

## Cómo ejecutar

Se requiere Python 3.13+ y las siguientes librerías. Versiones con las que se ejecutaron los notebooks:

```
pandas 3.0.5      numpy 2.5.1       scikit-learn 1.9.0
matplotlib 3.11.1 seaborn 0.13.2    scipy 1.18.1
```

```bash
pip install pandas numpy scikit-learn matplotlib seaborn scipy jupyter
jupyter notebook
```

Los notebooks leen los datos con rutas relativas (`data/...`), así que deben ejecutarse desde la raíz del repositorio. Todos los modelos usan `random_state=42`, de modo que los resultados son reproducibles.

## Metodología

El trabajo sigue las ocho fases del enunciado:

| Fase | Contenido |
|---|---|
| **1** | EDA, limpieza y preprocesamiento — nulos, duplicados, outliers por IQR, gráficas y justificación de encoding y escalamiento |
| **2** | División 70/15/15 y `Pipeline` con `ColumnTransformer`, con verificación explícita de ausencia de *data leakage* |
| **3** | Regresión: Regresión Lineal Múltiple, KNN, Decision Tree, Random Forest y Gradient Boosting |
| **4** | Clasificación: KNN, Decision Tree, Random Forest y Gradient Boosting |
| **5** | Métricas en train y validación — detección de overfitting/underfitting |
| **6** | Evaluación final sobre el conjunto de test, intacto hasta este punto |
| **7** | Validación cruzada K-Fold sobre el mejor modelo de cada problema |
| **8** | Conclusiones comparativas y subplot (2,2) de hallazgos |

**Preprocesamiento.** Todas las transformaciones que aprenden de los datos —imputación, escalamiento, encoding— se ajustan con `.fit()` **únicamente sobre `X_train`**; sobre validación y test solo se aplica `.transform()`. Cada modelo recibe su propia copia del `ColumnTransformer` vía `clone()`, de modo que su ajuste es independiente.

**Encoding.** One-Hot para variables nominales y Ordinal solo donde existe un orden real: `stops`, `departure_time`, `arrival_time` y `class` en vuelos; `Risk`, `T`, `N` y `Stage` en tiroides. Se evita el Ordinal Encoding en variables sin orden para no introducir distancias falsas que penalizarían a KNN y a la regresión lineal.

**Escalamiento.** `RobustScaler` en vuelos, por los outliers reales detectados en `duration`; `StandardScaler` en tiroides, donde `Age` no presenta ninguno.

## Resultados

### Regresión — conjunto de test

| Modelo | MAE (rupias) | MSE | R² |
|---|---:|---:|---:|
| **Random Forest** | **1,095.36** | 7,618,250 | **0.9851** |
| Decision Tree | 1,183.64 | 12,427,652 | 0.9756 |
| KNN | 1,809.18 | 14,581,272 | 0.9714 |
| Gradient Boosting | 2,918.72 | 24,260,309 | 0.9525 |
| Regresión Lineal | 4,470.99 | 45,249,820 | 0.9113 |

El ranking se mantiene idéntico entre validación y test. Un MAE de 1,095 rupias representa un error cercano al 5% frente al precio promedio de ~20,890.

### Clasificación — conjunto de test (clase positiva `Yes`)

| Modelo | Accuracy | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| **Random Forest** | **0.9636** | 0.9375 | 0.9375 | **0.9375** |
| Gradient Boosting | 0.9636 | 0.9375 | 0.9375 | 0.9375 |
| KNN | 0.9455 | 0.9333 | 0.8750 | 0.9032 |
| Decision Tree | 0.9273 | 0.8333 | 0.9375 | 0.8824 |

Se prioriza el **Recall** sobre la Accuracy: un falso negativo —un paciente que recae y queda sin seguimiento reforzado— es clínicamente más costoso que un falso positivo.

### Validación cruzada (K = 5, sobre train + validation)

| | Métrica | Resultado |
|---|---|---|
| Regresión (`KFold`) | MAE | **1,108.81 ± 8.76** rupias |
| Clasificación (`StratifiedKFold`) | Recall | **0.9035 ± 0.1310** |

El contraste entre ambas desviaciones es uno de los hallazgos centrales: con 255,130 filas el error del modelo de regresión es extraordinariamente estable (coeficiente de variación del 0.79%), mientras que con 309 pacientes el Recall de clasificación oscila entre 0.68 y 1.00 según el fold. La estratificación es necesaria en el segundo caso porque el target está desbalanceado.

## Conclusiones principales

**Random Forest ganó en ambos problemas, pero por razones distintas.** En regresión el factor decisivo fue el **sesgo**: el precio es bimodal por clase de cabina —`class` concentra el 88% de la importancia del modelo— y la Regresión Lineal, incapaz de representar ese salto, quedó en *underfitting* con un error cuatro veces mayor. En clasificación el factor fue el **manejo de categóricas**: 16 de 17 variables son categóricas, terreno natural para los árboles, mientras KNN se hundió (Recall de 0.647 en validación) porque la distancia euclidiana pierde sentido en un espacio de 41 dimensiones casi todas binarias.

**La limpieza determinante fue la opuesta en cada problema.** En tiroides, *eliminar* los 19 duplicados: sin identificador de paciente y con solo 364 filas, dejarlos habría permitido que un mismo registro cayera en train y en validación a la vez. En vuelos, *no* eliminar los outliers de `price`: eran los tiquetes Business que dominan la señal, y recortarlos por IQR habría amputado el segmento que el modelo necesita aprender. En ambos casos lo correcto fue lo contrario del reflejo automático.

## Limitaciones

- **No se realizó búsqueda de hiperparámetros**; todos los modelos usan los valores por defecto de scikit-learn, lo que perjudica especialmente a Gradient Boosting.
- En vuelos la **partición es aleatoria y no temporal**, lo que probablemente sobreestima el desempeño frente a un escenario real de predicción de precios futuros.
- En tiroides el **tamaño muestral es la limitación dominante**: con 55 pacientes en test, un solo caso mal clasificado mueve el Recall 6.25 puntos.
- En tiroides, `Response` —la variable más importante, con el 41.8% del peso del modelo— es una **evaluación posterior al tratamiento**. El modelo es válido como herramienta de seguimiento, pero no serviría para predecir recurrencia en el momento del diagnóstico, cuando esa variable aún no existe.

## Equipo

| Nombre | GitHub | Correo |
|---|---|---|
| Andrés Felipe Vélez Álvarez | [@AndresVelez31](https://github.com/AndresVelez31) | afveleza@eafit.edu.co |
| Sebastián Salazar Henao | [@Salazar1022](https://github.com/Salazar1022) | ssalazarh3@eafit.edu.co |
| Nathalia Valentina Cardoza Azuaje | [@NathaliaValentinaCardozaAzuaje](https://github.com/NathaliaValentinaCardozaAzuaje) | nvcardozaa@eafit.edu.co |
| Samuel Samper Cardona | [@Ssamperc](https://github.com/Ssamperc) | ssamperc@eafit.edu.co |
