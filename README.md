# kappa-test

> **Status científico:** hipótese fenomenológica exploratória, não estabelecida. Resultados atuais são 100% sintéticos (validação do método) — ver [wiki](https://github.com/tiago500x-ctrl/kappa-test/wiki) para detalhes, gráficos e o que falta para uma análise com dados reais.

Teste de razão de verossimilhança (Poisson) para uma atenuação exponencial
de contagens de neutrinos por fonte em função de uma "coluna de plasma":

$$\mu_i(\kappa) = \mu_{0,i}\, e^{-\kappa\, C_{p,i}}, \qquad \kappa \ge 0$$

- **H0:** κ = 0 (sem atenuação)
- **H1:** κ > 0

## Estatística

$$TS = 2\,[\ln L(\hat\kappa) - \ln L(0)]$$

Como κ = 0 está na borda do espaço de parâmetros, sob H0 o TS segue
assintoticamente a mistura ½χ²₀ + ½χ²₁ (Chernoff 1954; Self & Liang 1987).
O p-value é unilateral, `p = ½ · P(χ²₁ > TS)`, e a significância é `σ = √TS`.

## Uso

```bash
pip install -r requirements.txt
python kappa_test.py seus_dados.csv
```

O CSV precisa das colunas:

| coluna          | descrição                                  |
|-----------------|--------------------------------------------|
| `counts_obs`    | contagens observadas por fonte (≥ 0)       |
| `counts_pred`   | contagens esperadas sem atenuação (> 0)    |
| `plasma_column` | coluna de plasma ao longo da linha de visada (≥ 0) |

### Dados sintéticos

```bash
python make_synthetic.py --kappa 0.05 -o sintetico.csv
python kappa_test.py sintetico.csv
```

### Testes

```bash
pytest -q
```

## Limitações

- `counts_pred` é tratado como conhecido exatamente; incertezas sistemáticas
  no fluxo previsto (nuisance parameters) não são incluídas.
- A aproximação assintótica pode falhar com poucas fontes ou contagens baixas;
  nesses casos, calibre o TS com pseudo-experimentos.
- Ao usar dados públicos do IceCube, siga os termos de uso e a forma de
  citação do respectivo data release.

## `real_data_pipeline/` — pipeline completa para dados reais

Versão mais completa, com crossmatch angular+redshift, bootstrap, calibração
nula empírica (10.000 pseudo-experimentos, resolução de p-value ~1e-4),
estudo de injeção (viés/cobertura/poder estatístico) e checagem de
consistência por energia/hemisfério. Ver `real_data_pipeline/README.md` e a
página [Resultados](https://github.com/tiago500x-ctrl/kappa-test/wiki/Results)
da wiki para os números e gráficos completos (ainda sintéticos).

## Dados reais coletados

Uma primeira amostra **real** do IceCube foi coletada: **ICECAT-1** (IceCube
Event Catalog of Alert Tracks), 348 eventos de alerta reais (2011–2023),
licença CC0, DOI [10.7910/DVN/SCRUCD](https://doi.org/10.7910/DVN/SCRUCD).
Salvo em `real_data_pipeline/data/real_samples/`, com proveniência completa
em `real_data_pipeline/DATA_PROVENANCE.md`. Gráficos descritivos (mapa
celeste, distribuição de energia, SIGNAL vs FAR) na página
[Real-Data-Analysis](https://github.com/tiago500x-ctrl/kappa-test/wiki/Real-Data-Analysis)
da wiki.

**Importante:** é uma amostra de eventos individuais de alta confiança
(triggered), não um catálogo de contagens por fonte fixa — ainda não foi
ajustada no modelo de κ (ver limitações na página da wiki e na issue
[#1](https://github.com/tiago500x-ctrl/kappa-test/issues/1)). O lado do
plasma (FRB/DM) continua sem dados reais.

## Ferramentas externas avaliadas

- **[PISA](https://github.com/icecube/pisa)** (framework oficial do IceCube
  para matrizes de resposta/oscilação): instala e importa com sucesso neste
  ambiente após `apt install build-essential python3-dev` — ver issue
  [#17](https://github.com/tiago500x-ctrl/kappa-test/issues/17) (fechada).
  Avaliação de uso efetivo no pipeline ainda pendente.
- **[GraphNeT](https://github.com/graphnet-team/graphnet)** (GNNs para
  reconstrução de eventos): instalação bloqueada neste ambiente — exige
  `numpy<2.0` (incompatível com o numpy 2.x já em uso) e um venv isolado. Ver
  issue [#18](https://github.com/tiago500x-ctrl/kappa-test/issues/18) (aberta)
  para o diagnóstico completo e os comandos para destravar.

## Licença

MIT — veja [LICENSE](LICENSE).
