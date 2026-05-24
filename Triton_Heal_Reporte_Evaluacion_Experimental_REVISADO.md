# PROYECTO TRITON HEAL — Reporte de Evaluación Experimental (Revisado)

**Equipo:** [nombres]  
**Curso:** TC3002B — Reporte Estadístico  
**Fecha de entrega:** 25 de mayo de 2026  
**Versión:** Pre-registro + resultados *(marcar qué secciones ya tienen datos reales)*

---

## Nota para el equipo (leer antes de entregar)

Este documento corrige y alinea el borrador original con la rúbrica del curso. Cambios principales:

1. **Encuadre del método propuesto:** El curso pide comparar un método basado en **SLM** frente a baselines. En Triton Heal, el aporte del equipo no es “generar kernels con SLM”, sino la **arquitectura generativa-verificativa con verificadores locales (SLM) + veto de frontera**. Eso debe quedar explícito en 1.1.
2. **PRI-2 ahora tiene hipótesis y prueba pre-registradas** (antes solo se discutía en narrativa).
3. **Corrección estadística:** latencia con diseño within-subjects no debe analizarse con Mann-Whitney entre grupos independientes.
4. **Parte II:** Si aún no corren experimentos, **no incluyan números** como los de la Tabla 5 del PDF (94.2 %, χ² = 21.4, etc.); eso contradice el espíritu del pre-registro y puede interpretarse como HARKing.
5. **DeepSeek-Coder-V2:** En el PDF original figura como generador; aquí se documentan **condiciones experimentales** (rol fijo por corrida).

---

# PARTE I — PRE-REGISTRO EXPERIMENTAL

*Registrar esta parte **antes** de ejecutar experimentos. No modificar hipótesis ni pruebas sin justificar en un apéndice.*

## 1.1 Objetivo experimental

### Problema

Los kernels escritos en **Triton DSL** pueden contener errores de memoria, incompatibilidades de tipo y violaciones de restricciones de hardware (tiling, SRAM, alineación) que solo se manifiestan tras compilación (MLIR/LLVM) o en ejecución GPU, con costo alto (fallos, reinicios de contexto CUDA, tiempo de depuración).

### Método propuesto (equipo)

**Triton Heal:** capa de filtrado **pre-ejecución** con arquitectura **generativa-verificativa dual**:

- **Verificador local (SLM):** Llama-3.1-8B-Instruct — primera línea, baja latencia.
- **Verificador de frontera (baseline fuerte):** Claude 3.5 Sonnet — veto jerárquico cuando hay desacuerdo o riesgo alto.
- **Generación / reparación:** puede usar GPT-4o o DeepSeek según configuración del pipeline; en este estudio se **fija un rol por corrida** (ver 1.3).
- **Auto-reparación:** loop con backoff exponencial (máx. 3 reintentos) si el kernel no compila o el verificador marca `unsafe`.

### Comparaciones requeridas por el curso (≥2 baselines)

| Rol | Sistema / modelo | Justificación |
|-----|------------------|---------------|
| **Propuesto** | Triton Heal (SLM local + arquitectura dual + veto) | Método del equipo |
| **Baseline 1** | Solo verificador local: Llama-3.1-8B-Instruct | Aislar valor del SLM sin veto |
| **Baseline 2** | Solo verificador frontera: Claude 3.5 Sonnet | Baseline comercial fuerte |
| **Baseline 3 (opcional)** | Solo verificador frontera: GPT-4o | Segundo baseline comercial |
| **Referencia adicional** | DeepSeek-Coder-V2-Instruct (local, ~16B activos) | Segundo SLM/local para PRI-1 |

### Preguntas de investigación

- **PRI-1:** ¿Los modelos de frontera (Claude 3.5 Sonnet, GPT-4o) superan a los SLM locales (Llama-3.1-8B, DeepSeek-Coder-V2) en detección de errores de memoria y restricciones de hardware?
- **PRI-2:** ¿La arquitectura con **veto jerárquico** (local + frontera) reduce la tasa de **falsos negativos** frente a usar **solo** el verificador local?
- **PRI-3:** ¿Cuál es el costo de **latencia** del verificador de frontera frente al local, y es aceptable frente a la ganancia en detección?

### Lo que se desea demostrar (criterio de éxito del estudio)

No basta con que Claude tenga MSR más alto en promedio. Se busca evidencia de que:

1. La diferencia es **estadísticamente significativa** tras corrección por comparaciones múltiples.
2. El **tamaño del efecto** supera un umbral de relevancia práctica pre-definido.
3. El **sistema propuesto** (dual + veto) mejora **sensibilidad** sin hacer inviable el filtrado por latencia.

---

## 1.2 Hipótesis

Todas las pruebas usan **α = 0.05** antes de corrección; la α ajustada se define en 1.8.

### H1 — MSR: frontera vs SLM local (PRI-1)

**Métrica:** Memory Safety Rate (MSR) — ver 1.5.

- **H₀₁:** No hay diferencia en MSR entre Claude 3.5 Sonnet y Llama-3.1-8B-Instruct en el mismo conjunto de kernels.
- **H₁₁ (unilateral):** Claude 3.5 Sonnet tiene **mayor** MSR que Llama-3.1-8B-Instruct.

**Par adicional pre-registrado:**

- **H₀₁b:** No hay diferencia en MSR entre GPT-4o y DeepSeek-Coder-V2-Instruct.
- **H₁₁b (unilateral):** GPT-4o tiene **mayor** MSR que DeepSeek-Coder-V2-Instruct.

### H2 — Falsos negativos: arquitectura dual vs verificador único (PRI-2)

**Métrica:** Tasa de falsos negativos (FNR) sobre kernels con ground truth `unsafe`.

\[
\text{FNR} = \frac{FN}{FN + TP}
\]

- **H₀₂:** No hay diferencia en FNR entre **Triton Heal (dual + veto)** y **solo Llama-3.1-8B**.
- **H₁₂ (unilateral):** Triton Heal tiene **menor** FNR que solo Llama.

*Justificación:* PRI-2 es el núcleo del diseño del sistema; debe contrastarse estadísticamente, no solo narrativamente.

### H3 — Detección de violaciones SRAM / tiling (hipótesis secundaria)

Restringida al subconjunto de kernels etiquetados con violación de SRAM o tiling desalineado.

- **H₀₃:** No hay diferencia en la tasa de detección en ese subconjunto entre frontera y local.
- **H₁₃ (unilateral):** Los modelos de frontera detectan **más** casos que los locales.

### H4 — Latencia (PRI-3)

- **H₀₄:** No hay diferencia en la mediana de latencia de inferencia entre verificadores de frontera y locales.
- **H₁₄ (unilateral):** La mediana de latencia de frontera es **mayor** que la de locales *(esperado; relevancia práctica se evalúa aparte)*.

---

## 1.3 Variables experimentales

### Independientes

| Variable | Niveles |
|----------|---------|
| **Configuración del sistema** | Solo Llama; solo Claude; solo GPT-4o; solo DeepSeek; **Triton Heal dual** |
| **Modelo LLM** | Llama-3.1-8B-Instruct, DeepSeek-Coder-V2-Instruct, Claude 3.5 Sonnet, GPT-4o |
| **Escala** | Local SLM (&lt;10B–16B) vs frontera (API, &gt;&lt;70B clase) |
| **Temperatura** | T = 0.0 |
| **Formato de salida** | JSON: `{"safe": bool, "reason": str, "line_of_error": int}` |

### Dependientes

| Variable | Tipo |
|----------|------|
| MSR (recall en clase unsafe) | Proporción |
| Precision, Recall, F1 (safe/unsafe) | Proporción / derivadas |
| FNR, FPR | Proporción |
| Latencia de inferencia (ms) | Continua |
| Disagreement Rate (DR) entre verificadores A y B | Proporción |
| Self-Healing Success Rate (compilación tras reparación) | Proporción |

### Controladas

- Benchmark y versión (commit fijado)
- GPU de evaluación: NVIDIA A100 80GB, CUDA 12.2
- Timeout de API: 10 s
- Versiones de modelo pinneadas (no endpoints `latest`)
- Mismo prompt / system prompt (versionado en repo)
- Ground truth: etiquetado por ≥3 anotadores + regla de mayoría; reportar κ

---

## 1.4 Diseño experimental

**Tipo:** **Within-subjects** — cada kernel del benchmark se evalúa bajo todas las configuraciones relevantes.

**Unidad de análisis:**

- Clasificación: un par (kernel, configuración) → etiqueta predicha vs ground truth.
- Latencia: una medición por (kernel, configuración, repetición).

**Tamaño muestral planeado:**

- **N = 120** kernels (60 safe / 60 unsafe), salvo que el benchmark real tenga otro tamaño — **actualizar N antes de correr**.
- **R repeticiones de latencia:** R = 5 por (kernel, modelo) para estimar variabilidad de red; clasificación con T=0.0 se reporta en 1 repetición salvo empate técnico.

**Comparaciones principales:** ver tabla en 1.8 (mínimo 6 contrastes confirmatorios).

**Justificación within-subjects:** controla complejidad del kernel; permite McNemar y pruebas pareadas; mayor poder que between-subjects.

**Sesgos controlados:**

- Orden de llamadas a APIs aleatorizado por kernel.
- Evaluador automático ciego al método (solo recibe JSON).
- Pin de versiones de modelo.

---

## 1.5 Métricas

### Memory Safety Rate (MSR)

| | |
|--|--|
| **Definición** | Proporción de kernels **unsafe** correctamente identificados como `unsafe`. |
| **Fórmula** | MSR = TP / (TP + FN) *(recall en clase positiva unsafe)* |
| **Interpretación** | Mayor MSR → mejor detección de riesgo real. |
| **Limitación** | **No penaliza falsos positivos**; usar junto con F1 y FPR. |

### F1-Score (clasificación binaria safe/unsafe)

| | |
|--|--|
| **Fórmula** | F1 = 2PR/(P+R) |
| **Interpretación** | Balance precisión–sensibilidad. |
| **Limitación** | Sensible al balance de clases; mitigado con benchmark 50/50. |

### Falsos negativos (FNR) y falsos positivos (FPR)

Reportar ambos para no optimizar solo sensibilidad.

### Latencia (ms)

Tiempo desde envío del prompt hasta JSON válido. Reportar **mediana**, IQR y percentil 95.

### Disagreement Rate (DR)

Proporción de kernels donde Llama y Claude difieren. **No indica quién acierta** — usar con análisis de veto.

### Self-Healing Success Rate

Kernels que compilan tras ≤3 reintentos / kernels que fallaron compilación inicial.

**Limitación:** compilar ≠ semántica correcta.

### Métricas del curso no aplicables (justificación)

- **Speedup:** no aplica; el sistema no mide aceleración de kernels, sino **detección pre-compilación**.
- **Tasa de compilación cruda:** se reporta como métrica secundaria vía Self-Healing, no como métrica principal.

---

## 1.6 Tamaño del efecto esperado

| Contraste | Efecto esperado | Medida |
|-----------|-----------------|--------|
| MSR frontera vs local | **Mediano-grande** | φ ≈ 0.35–0.45 (McNemar) *o* Δ MSR ≈ 10–15 pp |
| FNR dual vs solo Llama | **Mediano** | reducción FNR ≈ 8–12 pp |
| Latencia frontera vs local | **Grande (práctico)** | Δ mediana ≈ 800–1500 ms |
| Claude vs GPT-4o | **Pequeño** | φ ≈ 0.15–0.25 |

*Nota:* El borrador original esperaba Cohen’s **d ≥ 0.8** pero usaba **McNemar** (φ). En esta revisión se unifica la expectativa en **φ ≈ 0.4** para comparaciones MSR, coherente con pruebas de proporciones pareadas.

---

## 1.7 Análisis de poder

| Parámetro | Valor |
|-----------|-------|
| α | 0.05 (familiar); α ajustada en 1.8 |
| Poder | 1 − β = 0.80 |
| Prueba principal MSR | McNemar (pareada) |
| Efecto | φ = 0.40 (mediano) |

**Cálculo (G*Power / aproximación):** para McNemar con φ = 0.40 y α = 0.05 bilateral, se requieren del orden de **50–65** pares discordantes o ~**80–100** observaciones totales según proporción marginal. Con **N = 120** kernels y ~50 % unsafe, el diseño es **suficiente** para MSR si el efecto real no es muy pequeño.

**Limitación:** si el benchmark real tiene N &lt; 80, recalcular poder **antes** de ejecutar.

---

## 1.8 Plan estadístico

### Comparaciones confirmatorias (pre-registradas)

| ID | Comparación | Métrica | Prueba |
|----|-------------|---------|--------|
| C1 | Claude vs Llama | MSR | McNemar |
| C2 | GPT-4o vs DeepSeek | MSR | McNemar |
| C3 | Triton Heal dual vs solo Llama | FNR | McNemar |
| C4 | Triton Heal dual vs solo Llama | F1 | McNemar sobre acierto binario por kernel |
| C5 | Frontera vs Local | Latencia mediana por kernel | **Wilcoxon signed-rank** en diferencias pareadas (mediana frontera − mediana local por kernel) |
| C6 | 4 configuraciones | F1 por kernel | **Friedman** + post-hoc **Wilcoxon** con Holm |

**Exploratorias (no para conclusión principal sin corrección):** MSR por categoría (matmul, softmax, etc.); subconjunto SRAM.

### Regla de decisión

Rechazar H₀ si:

1. **p &lt; α ajustada** (Holm-Bonferroni sobre C1–C6), y  
2. **Relevancia práctica:** Δ MSR o (FNR dual − FNR solo Llama) ≥ **5 pp** en la dirección esperada.

La significancia sin relevancia práctica **no** sustenta adopción del sistema.

### Intervalos de confianza

- Proporciones (MSR, FNR, DR): **Wilson 95 %**
- Latencia: **bootstrap pareado** B = 2000 sobre kernels
- Diferencia de medias pareadas MSR: IC 95 % para Δ proporción (método de Newcombe o bootstrap)

### Corrección por comparaciones múltiples

**Holm-Bonferroni** sobre las 6 comparaciones confirmatorias (preferido frente a Bonferroni estricto por mayor poder).

*El borrador original usaba α = 0.01 global (Bonferroni 0.05/5); aquí se mantiene Holm sobre 6 pruebas — reportar ambos si el profesor exige Bonferroni.*

### Violaciones de supuestos

- Si discordancias McNemar &lt; 25: usar **prueba exacta de McNemar**.
- Latencia: si empates abundantes, reportar **perm test** pareado como robustez.

---

## 1.9 Reproducibilidad experimental

| Componente | Especificación |
|------------|----------------|
| **GPU** | NVIDIA A100 80GB SXM4, CUDA 12.2 |
| **CPU / RAM** | [completar host real] |
| **SO** | Ubuntu 22.04 LTS (o el usado realmente) |
| **Software** | Python 3.10+, Triton 2.1.0, PyTorch 2.1.2, [completar] |
| **GPT-4o** | `gpt-4o-2024-05-13`, openai==1.12.0 |
| **Claude 3.5 Sonnet** | `claude-3-5-sonnet-20241022`, anthropic==0.18.1 |
| **Llama-3.1-8B-Instruct** | Meta-Llama-3.1-8B-Instruct, vLLM 0.4.2, BF16 |
| **DeepSeek-Coder-V2-Instruct** | deepseek-ai/DeepSeek-Coder-V2-Instruct, Ollama 0.1.38, Q4_K_M |
| **Inferencia** | T=0.0, max_tokens=512, top_p=1.0 |
| **Benchmark** | Triton Kernel Safety Benchmark v1.0 — **120 kernels (60/60)** — verificar existencia |
| **Repo** | `[URL real del equipo]` commit `[hash real]` |
| **Config** | `experiment_config.json`, seed=42 |

**⚠️ Acción:** Reemplazar URLs/commits de ejemplo si el repo aún no existe.

---

## 1.10 Amenazas a la validez (pre-registro)

### Validez interna

- Deriva de modelos en APIs → mitigación: pin de versión.
- Latencia de red en frontera → 5 repeticiones; reportar p95.
- Contaminación del benchmark con datos de entrenamiento → ~30 % kernels propios no publicados.
- **Confound en PRI-3:** frontera = API + red; local = GPU — no solo “tamaño del modelo”.

### Validez externa

- Benchmark limitado a 5 familias de kernels Triton.
- Solo A100; Hopper (H100) cambia límites SRAM ~15–20 %.

### Validez de constructo

- MSR ≠ seguridad total en producción.
- Ground truth humano con κ reportado; casos borderline de tiling.
- T=0.0 no refleja diversidad en producción.

---

# PARTE II — RESULTADOS Y ANÁLISIS

> **⚠️ IMPORTANTE:** Las tablas numéricas del PDF original (MSR 94.2 %, χ² = 21.4, etc.) parecen **simuladas o anticipadas**. Para la entrega académica:
>
> - Si **aún no hay experimentos:** deje la Parte II como plantilla con “[Pendiente]” o ejecute el pipeline y sustituya con datos reales.
> - Si **ya hay experimentos:** pegue aquí los CSV/logs y actualice números; no reutilice cifras inventadas.

## 2.1 Resultados descriptivos

### Tabla D1 — Resumen por configuración *(completar con datos reales)*

| Configuración | N eval. | JSON válidos | MSR (%) | F1 | FNR (%) | FPR (%) | Latencia mediana (ms) | IQR |
|---------------|---------|--------------|---------|-----|---------|---------|------------------------|-----|
| Triton Heal (dual) | | | | | | | | |
| Solo Llama-3.1-8B | | | | | | | | |
| Solo Claude 3.5 | | | | | | | | |
| Solo GPT-4o | | | | | | | | |
| Solo DeepSeek-V2 | | | | | | | | |

**Estadísticos obligatorios:** n total, éxitos de parsing JSON, media/DE o mediana/IQR, min, max por métrica.

**κ inter-anotador:** [valor] sobre [n] kernels disputados.

---

## 2.2 Visualización de datos

| Figura | Tipo | Contenido |
|--------|------|-----------|
| **Fig 1** | Barras + IC 95 % | MSR y F1 por configuración |
| **Fig 2** | Boxplot (eje log opcional) | Latencia por modelo (mostrar outliers de rate limit) |
| **Fig 3** | Barras por categoría | MSR en unsafe por familia (matmul, softmax, flash-attn, …) |
| **Fig 4** *(recomendada)* | Barras pareadas | FNR: dual vs solo Llama (PRI-2) |

Cada figura: título, ejes, unidades, leyenda, párrafo de interpretación.

---

## 2.3 Inferencia estadística

### Tabla I1 — Pruebas confirmatorias *(plantilla)*

| Comparación | Prueba | Estadístico | p (cruda) | p (Holm) | Tamaño efecto | IC 95 % | ¿Rechaza H₀? | ¿Relevancia práctica? |
|-------------|--------|-------------|-----------|----------|---------------|---------|--------------|------------------------|
| C1 Claude vs Llama (MSR) | McNemar | | | | φ = | | | Δ ≥ 5 pp? |
| C2 GPT-4o vs DeepSeek (MSR) | McNemar | | | | φ = | | | |
| C3 Dual vs solo Llama (FNR) | McNemar | | | | φ = | | | |
| C5 Latencia frontera vs local | Wilcoxon | | | | r = | | | Δ ms |
| C6 F1 global | Friedman | | | | W = | | | |

**Interpretación guiada:**

- ¿Se rechazó H₀ en C1/C2?
- ¿PRI-2 (C3) muestra reducción de FNR con efecto práctico?
- ¿La latencia (C5) invalida el despliegue o es aceptable en pre-compilación?

---

## 2.4 Comparación crítica

Discutir:

- Uniformidad de la brecha (¿solo flash-attention / SRAM?).
- Consistencia bajo 5 repeticiones de latencia vs clasificación determinista (T=0).
- Si DeepSeek cierra la brecha con Llama en kernels “sintácticos”.

---

## 2.5 Amenazas observadas a la validez

Documentar lo que **realmente ocurrió** (no copiar 1.10):

| Amenaza | Tipo | Evidencia | Impacto |
|---------|------|-----------|---------|
| Rate limiting GPT-4o | Interna | n picos &gt;3000 ms | Latencia p95 subestimada |
| 14 kernels con desempate | Interna | κ=… | MSR en tiling borderline |
| … | | | |

---

## 2.6 Discusión

- ¿H₁₁ soportada con φ moderado vs expectativa grande?
- ¿H₁₂ (veto) validada con C3 y análisis de desacuerdos?
- ¿PRI-3: trade-off latencia vs costo de fallo GPU?
- Limitaciones: N, hardware único, métricas proxy.

---

## 2.7 Conclusión

Responder explícitamente:

1. ¿Se rechazó H₀ en cada contraste principal?
2. ¿Evidencia suficiente tras Holm y relevancia práctica?
3. ¿El **método propuesto (Triton Heal)** justifica adopción frente a solo SLM?
4. ¿Reproducible por otro equipo con el repo/commit documentado?
5. ¿Cuánto explica la diferencia el **tamaño del modelo** vs la **arquitectura**? *(requiere ablation: dual vs solo frontera)*

---

## Apéndice A — Checklist de revisión del borrador original

| Aspecto | Estado borrador PDF | Acción |
|---------|---------------------|--------|
| Alineación “SLM vs baseline” | Parcial (compara 4 modelos sueltos) | Enfatizar Triton Heal como método propuesto |
| PRI-2 con prueba estadística | Solo narrativa | Añadido C3 McNemar FNR |
| Mann-Whitney latencia | **Incorrecto** para pareados | Cambiado a Wilcoxon |
| Friedman sobre F1 | Unidad de análisis ambigua | Aclarar: F1 por kernel o score global |
| Parte II con números | Parecen simulados | Reemplazar con datos reales o marcar pendiente |
| MSR solo | Omite FPR | Reportar F1, FPR, FNR |
| Fecha Mayo 2025 | Desactualizada | 2026 |
| Repo ejemplo | Puede no existir | URL/commit reales |

---

## Apéndice B — Texto sugerido para 1.1 (copiar al Word)

*El párrafo de tu captura de pantalla está bien redactado. Solo añade al final:*

> **Método propuesto del equipo:** la arquitectura Triton Heal, que combina verificación local con SLM (Llama-3.1-8B-Instruct) y veto jerárquico con un modelo de frontera (Claude 3.5 Sonnet), frente a baselines que usan un único verificador (local o de frontera) y frente a un segundo SLM local (DeepSeek-Coder-V2-Instruct).

---

*Fin del documento revisado.*
