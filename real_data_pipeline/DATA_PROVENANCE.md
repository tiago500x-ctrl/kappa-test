# Proveniência dos dados

| Conjunto | Release/DOI | URL | Licença | SHA256 | Data de acesso |
|---|---|---|---|---|---|
| IceCube (eventos de alerta reais) | ICECAT-1, DOI 10.7910/DVN/SCRUCD, v4 (2023-11-09) | https://doi.org/10.7910/DVN/SCRUCD, arquivo `IceCube_Gold_Bronze_Tracks.tab` | CC0 1.0 (domínio público) | `c4db13b85cf172b90af91692f467eb35538f85a5e8f12859281de4d1d35332ee` | 2026-10-07 |
| Plasma/FRB | PENDENTE | PENDENTE | PENDENTE | PENDENTE | PENDENTE |

Arquivo salvo em `real_data_pipeline/data/real_samples/IceCube_Gold_Bronze_Tracks.tab`.

## ICECAT-1 — o que é e o que NÃO é

**O que é:** catálogo real de 348 eventos individuais de alerta (tracks de múon de alta energia, >~100 TeV) observados pelo IceCube entre 2011–2023, cada um com probabilidade individual de origem astrofísica (`SIGNAL`) calculada pela própria colaboração. Licença CC0, citação: IceCube Collaboration, *IceCat-1: The IceCube Event Catalog of Alert Tracks*, [arXiv:2304.01174](https://arxiv.org/abs/2304.01174).

**O que NÃO é:** não é um catálogo de contagens por fonte fixa (o formato `counts_obs`/`counts_signal_pred`/`counts_background_pred` que `src/inference.py::fit_model` espera). É uma amostra **enviesada** (triggered) de eventos individuais de alta confiança, não um point-source search binado. Usar isso diretamente no modelo de atenuação exigiria repensar a estatística (ex: usar `SIGNAL` como peso por evento, ou agrupar eventos em bins espaciais) — não é um "encaixe" direto no pipeline atual.

**Visualizações descritivas** (sem ajuste de κ, só para conferir os dados reais) estão na wiki, página [[Real-Data-Analysis]].

## Transformações obrigatórias (antes de qualquer ajuste de κ com isso)

Documentar cortes, unidades, correções de DM, tratamento do redshift, exposição, área efetiva, seleção de eventos e exclusões. Especificamente para o ICECAT-1: decidir se/como usar `SIGNAL` e `FAR` como pesos, e como lidar com a ausência de redshift (a maioria dos eventos de alerta não tem contraparte eletromagnética confirmada).
