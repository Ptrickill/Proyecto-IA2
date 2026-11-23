# Declaración de uso de IA generativa

El grupo utilizó el asistente Claude (Anthropic) en el Entregable 2 para:

| Parte | Uso de la IA | Validación del grupo |
|---|---|---|
| Generador sintético (`data.py`) | Diseño de los dos escenarios, del tamaño del conjunto y primera implementación | Revisión de que la geometría y las magnitudes correspondan al caso de referencia; inspección visual de series (figura 1) |
| Líneas base y modelos (`baselines.py`, `models.py`) | Implementación inicial | Revisión cruzada del código entre integrantes; verificación de ausencia de fuga de información (particiones, escalado, filtro) |
| Protocolo, métricas y pruebas (`protocol.py`, `metrics.py`) | Implementación inicial y corrección de la definición de MASE | Comprobación manual de métricas en casos pequeños |
| Informe | Borrador de redacción de Métodos, Resultados y Discusión | Lectura crítica, ajuste del texto y verificación de cada cifra contra `results/tables/` |

Los integrantes asumen la responsabilidad total del código y del contenido. Toda cifra del informe procede de
`results/tables/`, generada por `scripts/report.py` a partir de las predicciones guardadas en `results/raw/`.
