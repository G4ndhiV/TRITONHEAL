# Triton Heal — Pipeline experimental

## Requisitos

- Python 3.10+
- Opcional: [Ollama](https://ollama.com) con `llama3.1:8b` y `deepseek-coder-v2`
- Opcional: `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`
- LaTeX (`latexmk`, `pdflatex`) para el PDF

## Uso rápido

```bash
cd /Users/gandhivaldez/Estadistica
make all
```

## Modos de verificación

| `VERIFIER_MODE` | Comportamiento |
|-----------------|----------------|
| `heuristic` | Reglas estáticas (local lenient / frontera estricta) |
| `ollama` | Solo Ollama; fallback heurístico si no hay daemon |
| `api` | Solo APIs si hay keys |
| `hybrid` | APIs → Ollama → heurístico (por prioridad) |

Copia `.env.example` a `.env` y configura keys.

## Salidas

- `results/predictions.csv` — predicciones crudas
- `results/descriptive.csv`, `inference.csv`, `stats_summary.json`
- `figures/*.pdf`
- `report/main.pdf`
