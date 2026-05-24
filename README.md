# TRITONHEAL

Evaluación experimental TC3002B: verificación pre-compilación de kernels Triton DSL con SLM local y arquitectura dual (Triton Heal).

## Estructura

| Ruta | Descripción |
|------|-------------|
| `triton_heal_experiment/` | Pipeline: benchmark, experimentos, análisis, figuras |
| `results/` | `predictions.csv`, estadísticos, `stats_summary.json` |
| `figures/` | Gráficas PDF para el reporte |
| `report/` | Reporte LaTeX y `main.pdf` |

## Reproducibilidad

```bash
make venv benchmark experiments stats figures pdf
```

Corrida documentada: `VERIFIER_MODE=ollama` (ver `results/run_meta.json`).

## Reporte

PDF entregable: [`report/main.pdf`](report/main.pdf).

Autor en documento: **Equipo 11**.
