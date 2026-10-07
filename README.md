# kappa-test

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
- Nenhum dado real está incluído neste repositório. Ao usar dados públicos do
  IceCube, siga os termos de uso e a forma de citação do respectivo data release.

## Licença

MIT — veja [LICENSE](LICENSE).
