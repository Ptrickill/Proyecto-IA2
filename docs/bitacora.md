# Bitácora de experimentos

Registro cronológico de ejecuciones, incluidos resultados negativos y decisiones.

## Semana 9–10 · Primer conjunto sintético (3 terraplenes, 142 puntos, semilla 2026)

**E-01 · Prueba rápida E1 al 50 %, una semilla.** El proceso gaussiano puro (M1) obtuvo MAE 2,10 mm frente a
1,82 mm de la persistencia. *Diagnóstico:* el núcleo Matérn estacionario vuelve a la media al extrapolar.
*Decisión:* adelantar el híbrido hiperbólico + GP (M1-H).

**E-02 · Ejecución completa con 3 terraplenes (20,6 min).** M1-H fue el único modelo mejor que B1; M2 y M3 fallaron al
30 %; la LSTM eligió la época 1 en 2 de 5 semillas. *Problemas detectados:* (a) un solo generador hiperbólico
podía favorecer a B1 y M1-H; (b) E2 por estación medía interpolación; (c) tres terraplenes eran pocos para estudiar
la transferencia. Resultados sustituidos por E-03.

## Semana 10 · Cambio de alcance: los datos reales no fueron autorizados

**Decisión.** La empresa no autorizó el uso de sus registros. El proyecto pasa a estudio controlado con datos
sintéticos (plan B del Entregable 1). Se rediseña el conjunto según lo que necesita cada experimento:
6 terraplenes (292 puntos), 2 escenarios (A hiperbólico, B exponencial + fluencia logarítmica) y 3 réplicas
(semillas 2026–2028). Se elimina E2 por estación, se añade E4 (volumen de datos) y las pruebas pasan a bloques de
estación.

**Incidencia técnica.** Dos procesos con LightGBM multihilo en 2 núcleos se ralentizaron por sobresuscripción de
OpenMP. *Solución:* `n_jobs=1` y `OMP_NUM_THREADS=1`; cada conjunto tarda ~7 min.

**Ajuste de costo.** Los pares de entrenamiento de M2/M3 crecían de forma cuadrática con las campañas. Se limita a
6 objetivos por ancla, espaciados uniformemente.

**E-03 · Ejecución completa, 6 conjuntos (40,7 min).** Resultados en `results/tables/`. Hallazgos:

- B1 es la mejor línea base en A y en B. M1-H lo mejora de forma significativa al 30 y 50 %, pero solo 0,04–0,05 mm.
- **Resultado negativo:** M2 y M3 no superan a B1 cuando aprenden solo de la historia temprana.
- **Resultado negativo:** las variables espaciales no tienen efecto consistente (E3).
- E2: la transferencia reduce el error de M2 un 9–12 % y el de M3 un 22–25 %.
- E4: M2 necesita ~150 puntos de entrenamiento (2 terraplenes completos) para igualar a B1.
- Cobertura al 95 %: M1 91–94 % con intervalos de 10–15 mm; M1-H 85–89 % con 6–8 mm.
- Hampel: precisión 91–95 %, exhaustividad 39–44 %.

**E-04 · Prueba exploratoria (escenario A, réplica 2026, semilla 0).** M2 sin la variable "día del ancla":
MAE al 70 % 2,05 → 1,89 mm, pero al 30 % 2,00 → 2,48 mm. *Conclusión:* el deterioro de M2 al 70 % se debe en parte a
la extrapolación conjunta (ancla tardía, horizonte largo). *Decisión:* sustituir el día del ancla por variables del
estado de consolidación en la semana 11.
