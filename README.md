# TRITONHEAL

Evaluación experimental TC3002B: **generación de kernels Triton DSL con SLM**, verificación pre-compilación y método dual Triton Heal (Llama + reparación Coder-V2).

## Estructura

| Ruta | Descripción |
|------|-------------|
| `triton_heal_experiment/` | Benchmark, generación, verificación, estadística, figuras |
| `results/` | `generation.csv`, `predictions.csv`, `*_summary.json` |
| `figures/` | Gráficas PNG/PDF |
| `report/` | LaTeX + **`main.pdf`** (reporte completo rúbrica 1.1--2.7) |

## Reproducibilidad

```bash
make venv benchmark generation experiments stats-all figures pdf
```

Variables útiles:
- `VERIFIER_MODE=ollama`
- `GENERATION_MAX_TASKS=20` (por defecto 20 tareas en `generation_tasks.jsonl`)
- `GENERATION_REPEATS=3`
- `GENERATION_SKIP_CONFIGS=solo_deepseek` (si R1-14B hace timeout)

Hardware documentado: **Apple M4, 24 GB**, Ollama (`llama3.1:8b`, `deepseek-r1:14b`, `deepseek-coder-v2`).

Repositorio: [github.com/G4ndhiV/TRITONHEAL](https://github.com/G4ndhiV/TRITONHEAL)

## Reporte

PDF: [`report/main.pdf`](report/main.pdf) — Parte I (pre-registro 1.1--1.10), Parte II (resultados 2.1--2.7), matriz de cumplimiento en apéndice.

Equipo 11 — Mayo 2026.
