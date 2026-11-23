# Pronóstico de asentamientos milimétricos en terraplenes viales con aprendizaje automático

Repositorio del proyecto semestral de **Inteligencia Artificial 2** (José Carvajal, Daniel Díaz, Mauricio Mora).
Estudio controlado con **datos sintéticos calibrados**: compara tres líneas base geotécnicas (persistencia,
método hiperbólico, método de Asaoka) con tres modelos de aprendizaje automático (proceso gaussiano, gradient
boosting, LSTM) y un híbrido hiperbólico + proceso gaussiano, en un régimen de asentamientos de pocos milímetros,
campañas irregulares y puntos que se levantan.

> **Todos los datos son simulados.** No se usan registros de ninguna obra real. El generador está calibrado con la
> geometría y el orden de magnitud de un control topográfico de referencia.

## Datos

`src/settlement/data.py` genera seis terraplenes (292 puntos de control) con dos escenarios de consolidación:

- **A (`hyperbolic`)**: consolidación hiperbólica, la forma que suponen B1 y M1-H.
- **B (`exponential`)**: consolidación primaria exponencial + compresión secundaria logarítmica.

Cada escenario se genera con las semillas 2026, 2027 y 2028 (tres réplicas, 39 846 observaciones en total).
La observación incluye ruido de nivelación, error común de campaña, errores gruesos (2 %) y lecturas faltantes (4 %);
las columnas `true_mm` y `outlier_injected` guardan la señal sin ruido y la marca de error grueso.

## Estructura

```
src/settlement/
  data.py         generador sintético (dos escenarios) + carga de CSV con el mismo esquema
  preprocess.py   filtro de Hampel, interpolación a paso constante
  baselines.py    B0 persistencia, B1 hiperbólico, B2 Asaoka (con registro de fallos)
  models.py       M1 proceso gaussiano, M1-H híbrido, M2 LightGBM, M3 LSTM
  protocol.py     E1 (origen temporal), E2 (transferencia), E3 (ablación), E4 (volumen de datos)
  metrics.py      MAE, RMSE, MASE, relMAE, Wilcoxon por bloques de estación + Holm
scripts/
  run_all.py      ejecuta los experimentos de un conjunto (o de los seis con --all)
  report.py       tablas (results/tables/*.md|csv) y figuras (results/figures/*.png)
notebooks/
  colab_run.ipynb ejecución completa en Google Colab
docs/
  bitacora.md     registro de experimentos, incluidos los fallidos
  uso_ia.md       declaración de uso de IA generativa
```

## Ejecución

```bash
pip install -r requirements.txt
python scripts/run_all.py --all        # 2 escenarios x 3 réplicas, ~7 min por conjunto en CPU
python scripts/report.py               # tablas y figuras
```

Un solo conjunto: `python scripts/run_all.py --curve exponential --seed 2027`. Prueba rápida: `--quick`.
En CPU con pocos núcleos conviene `OMP_NUM_THREADS=1` y no ejecutar dos procesos de LightGBM con varios hilos a la vez.

## Esquema de datos (formato largo, una fila por punto y campaña)

| Columna | Descripción |
|---|---|
| `embankment` | código del terraplén (T1 … T6) |
| `point_id` | identificador del punto, p. ej. `T1-E03-L2` |
| `station` | índice de la estación a lo largo del eje |
| `abscissa_m` | abscisa relativa al inicio del terraplén (m) |
| `edge_dist_m` | distancia al borde (m) |
| `day` | días desde la primera campaña del terraplén |
| `settlement_mm` | movimiento observado respecto de la primera lectura (mm; − asentamiento, + levantamiento) |
| `true_mm` | señal sin ruido (solo sintético) |
| `outlier_injected` | error grueso insertado (solo sintético) |

## Reproducibilidad

- Semillas del generador: 2026–2028. Semillas de los modelos estocásticos: 0–2 (E1) y 0 (E2–E4).
- Ninguna partición mezcla observaciones al azar: todas son temporales o por terraplén.
- Toda normalización y todo filtrado se ajustan solo con la ventana de entrenamiento.
- `results/raw/<escenario>_s<semilla>/run_meta.json` registra cada ejecución.

## Uso de IA generativa

Ver `docs/uso_ia.md`.
