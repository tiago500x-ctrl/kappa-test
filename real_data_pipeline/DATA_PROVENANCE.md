# Proveniência dos dados

| Conjunto | Release/DOI | URL | Licença | SHA256 | Data de acesso |
|---|---|---|---|---|---|
| IceCube (eventos de alerta reais) | ICECAT-1, DOI 10.7910/DVN/SCRUCD, v4 (2023-11-09) | https://doi.org/10.7910/DVN/SCRUCD, arquivo `IceCube_Gold_Bronze_Tracks.tab` | CC0 1.0 (domínio público) | `c4db13b85cf172b90af91692f467eb35538f85a5e8f12859281de4d1d35332ee` | 2026-10-07 |
| Plasma/FRB | CHIME/FRB Catalog 1 (Amiri et al. 2021, ApJS 257, 59; erratum 2023), via VizieR J/ApJS/257/59/table2 | https://doi.org/10.26093/cds/vizier.22570059 | CDS/VizieR (dados de pesquisa, uso acadêmico padrão) | `cd07b9a41cd266f314ed76221fe824d884f45757f77e24b0702148c457ece538` | 2026-10-07 |

Arquivos salvos em `real_data_pipeline/data/real_samples/IceCube_Gold_Bronze_Tracks.tab` e `CHIME_FRB_Catalog1.csv`.

## ICECAT-1 — o que é e o que NÃO é

**O que é:** catálogo real de 348 eventos individuais de alerta (tracks de múon de alta energia, >~100 TeV) observados pelo IceCube entre 2011–2023, cada um com probabilidade individual de origem astrofísica (`SIGNAL`) calculada pela própria colaboração. Licença CC0, citação: IceCube Collaboration, *IceCat-1: The IceCube Event Catalog of Alert Tracks*, [arXiv:2304.01174](https://arxiv.org/abs/2304.01174).

**O que NÃO é:** não é um catálogo de contagens por fonte fixa (o formato `counts_obs`/`counts_signal_pred`/`counts_background_pred` que `src/inference.py::fit_model` espera). É uma amostra **enviesada** (triggered) de eventos individuais de alta confiança, não um point-source search binado. Usar isso diretamente no modelo de atenuação exigiria repensar a estatística (ex: usar `SIGNAL` como peso por evento, ou agrupar eventos em bins espaciais) — não é um "encaixe" direto no pipeline atual.

**Visualizações descritivas** (sem ajuste de κ, só para conferir os dados reais) estão na wiki, página [[Real-Data-Analysis]].

## CHIME/FRB Catalog 1 — o que é e o que NÃO é

**O que é:** 600 bursts de 536 FRBs (incluindo repetidores) detectados pelo CHIME/FRB entre 2018–2019, com RA/DEC, DM total medida, e **DM galáctica estimada por dois modelos independentes** (NE2001 e YMW16) — isso permite calcular um "excesso de DM" (DM total menos DM galáctica), usado aqui como proxy de `Cp` extragaláctico.

**O que NÃO é:** não tem redshift do hospedeiro para a grande maioria dos bursts (localização seria necessária, a maior parte do Catalog 1 não tem). O "excesso de DM" mistura contribuição do meio intergaláctico com a galáxia hospedeira do FRB — não separa as duas. E principalmente: **cruzar por vizinho mais próximo não implica associação física** — com 600 FRBs espalhados pelo céu (~41253 deg²), a separação angular esperada por puro acaso já é de alguns graus (ver análise em [[Real-Data-Analysis]]).

## Transformações obrigatórias (antes de qualquer ajuste de κ com isso)

Documentar cortes, unidades, correções de DM, tratamento do redshift, exposição, área efetiva, seleção de eventos e exclusões. Especificamente: decidir se/como usar `SIGNAL`/`FAR` do IceCube como pesos; como lidar com ausência de redshift em ambos os catálogos; e que o cruzamento atual (nearest-neighbor angular, sem filtro de redshift) é só uma primeira exploração, não uma associação física validada (ver issue #21).
