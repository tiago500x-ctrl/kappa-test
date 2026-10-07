# Kappa Test v2

Pipeline experimental para investigar uma possível atenuação adicional de neutrinos correlacionada com a coluna de plasma ao longo da linha de visada.

> **Aviso científico:** este projeto implementa uma hipótese fenomenológica. Os resultados de demonstração usam dados sintéticos e não constituem evidência de nova física nem resultados oficiais da colaboração IceCube.

## 1. Modelo analisado

Para cada fonte ou bin `i`, o número observado de eventos é modelado por:

```text
N_i ~ Poisson(mu_i)
```

com:

```text
mu_i = b_i + A * s_i * eta_eff * exp(-kappa * eta_plasma * C_i)
```

Onde:

- `N_i`: contagem observada;
- `b_i`: fundo previsto;
- `s_i`: sinal previsto sem a interação adicional;
- `A`: normalização global do sinal;
- `C_i`: coluna de plasma escalada em unidades de `10^21 cm^-2`;
- `kappa`: coeficiente efetivo de atenuação;
- `eta_eff`: parâmetro nuisance da eficiência;
- `eta_plasma`: parâmetro nuisance da escala de plasma.

As hipóteses testadas são:

- `H0: kappa = 0` — sem atenuação adicional;
- `H1: kappa > 0` — atenuação adicional correlacionada com plasma.

## 2. Requisitos

- Python 3.10 ou superior;
- `pip` atualizado;
- Git opcional;
- Linux, macOS ou Windows com PowerShell/WSL.

Dependências principais:

- NumPy;
- pandas;
- SciPy;
- Matplotlib;
- Astropy;
- PyYAML;
- Requests;
- pytest.

## 3. Instalação

### 3.1 Obter o projeto

Extraia `kappa-test-v2.zip` ou clone o repositório:

```bash
git clone https://github.com/tiago500x-ctrl/kappa-test.git
cd kappa-test/real_data_pipeline
```

### 3.2 Criar ambiente virtual

#### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### Windows PowerShell

```powershell
py -m venv .venv
.venv\Scripts\Activate.ps1
```

### 3.3 Instalar dependências

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 3.4 Verificar a instalação

```bash
pytest -q
```

Os testes verificam, entre outros pontos:

- recuperação aproximada de um `kappa` injetado;
- comportamento sob `H0`;
- consistência entre `Z` e `sqrt(TS)`;
- rejeição de crossmatches incompatíveis em redshift;
- execução do bootstrap;
- execução da calibração nula.

## 4. Estrutura do projeto

```text
kappa-test-v2/
├── config.yaml
├── requirements.txt
├── run_pipeline.py
├── README.md
├── DATA_PROVENANCE.md
├── data/
│   ├── raw/
│   └── processed/
├── outputs/
├── scripts/
│   └── generate_synthetic.py
├── src/
│   ├── __init__.py
│   ├── crossmatch.py
│   ├── inference.py
│   ├── io_utils.py
│   ├── reporting.py
│   ├── robustness.py
│   └── validate.py
└── tests/
    ├── test_crossmatch.py
    ├── test_inference.py
    └── test_robustness.py
```

## 5. Execução rápida com dados sintéticos

O gerador cria catálogos controlados com `kappa = 0.06` injetado.

```bash
python scripts/generate_synthetic.py
```

Serão criados:

```text
data/raw/icecube.csv
data/raw/plasma.csv
```

Execute o pipeline sem download:

```bash
python run_pipeline.py --config config.yaml --skip-download
```

> O bootstrap e a calibração nula podem demorar, pois cada réplica realiza novos ajustes de máxima verossimilhança.

## 6. Arquivo de configuração

O comportamento do pipeline é controlado por `config.yaml`.

### Caminhos

```yaml
paths:
  icecube_raw: data/raw/icecube.csv
  plasma_raw: data/raw/plasma.csv
  merged: data/processed/merged.csv
  results: outputs/results.json
  profile_plot: outputs/profile_likelihood.png
  bootstrap_csv: outputs/bootstrap_kappa.csv
  null_csv: outputs/null_calibration.csv
```

### Crossmatch

```yaml
matching:
  max_radius_deg: 2.0
  max_redshift_difference: 0.15
  require_plasma_behind_source: false
```

- `max_radius_deg`: separação angular máxima;
- `max_redshift_difference`: diferença máxima permitida em redshift;
- `require_plasma_behind_source`: filtro geométrico opcional; deve ser validado fisicamente antes do uso.

### Inferência e robustez

```yaml
model:
  kappa_max: 2.0
  efficiency_uncertainty: 0.10
  plasma_scale_uncertainty: 0.20
  bootstrap_samples: 500
  null_simulations: 1000
  random_seed: 42
  energy_bins_tev: [10, 100, 1000, 1000000]
```

Para um teste rápido, reduza temporariamente:

```yaml
bootstrap_samples: 20
null_simulations: 50
```

Para resultados finais, aumente o número de pseudoexperimentos. Valores-p muito pequenos exigem muito mais simulações; com 1.000 réplicas, a resolução mínima é aproximadamente `1/1001`.

## 7. Formato dos dados

### 7.1 Catálogo de neutrinos

O CSV padrão deve conter:

| Coluna | Unidade | Descrição |
|---|---:|---|
| `ra_deg` | graus | ascensão reta |
| `dec_deg` | graus | declinação |
| `redshift` | adimensional | redshift da fonte associada |
| `energy_tev` | TeV | energia representativa/reconstruída |
| `counts_obs` | eventos | contagem observada |
| `counts_signal_pred` | eventos | sinal previsto |
| `counts_background_pred` | eventos | fundo previsto |

### 7.2 Catálogo de plasma/FRB

| Coluna | Unidade | Descrição |
|---|---:|---|
| `ra_deg` | graus | ascensão reta |
| `dec_deg` | graus | declinação |
| `redshift` | adimensional | redshift do traçador |
| `dm_pc_cm3` | pc cm^-3 | medida de dispersão |

A conversão inicial usada é:

```text
C_e = DM * 3.085677581e18 cm^-2
```

Essa conversão é dimensional e não remove automaticamente as contribuições da Via Láctea, halo, hospedeira e ambiente local.

## 8. Uso com dados reais

### 8.1 Uso com arquivos locais

1. Coloque os catálogos em `data/raw/`.
2. Ajuste os caminhos e nomes das colunas em `config.yaml`.
3. Registre a proveniência em `DATA_PROVENANCE.md`.
4. Execute:

```bash
python run_pipeline.py --config config.yaml --skip-download
```

### 8.2 Download automático

Substitua os placeholders por URLs diretas:

```yaml
sources:
  icecube_url: "https://exemplo.org/icecube.csv"
  plasma_url: "https://exemplo.org/plasma.csv"
```

Depois execute:

```bash
python run_pipeline.py --config config.yaml
```

O downloader atual espera arquivos CSV diretos. Arquivos ZIP, páginas HTML, autenticação e APIs exigem adaptadores específicos.

## 9. Saídas

### `data/processed/merged.csv`

Catálogo após validação e crossmatch, incluindo:

- separação angular;
- redshift do traçador de plasma;
- DM;
- coluna de plasma em `cm^-2`;
- coluna de plasma escalada.

### `outputs/results.json`

Inclui:

- `kappa_hat`;
- `TS`;
- significância assintótica;
- valor-p assintótico unilateral;
- intervalo de confiança perfilado;
- intervalo bootstrap;
- valor-p empírico sob `H0`;
- ajustes por energia e hemisfério;
- parâmetros nuisance ajustados.

### `outputs/bootstrap_kappa.csv`

Uma linha por reamostragem, contendo `kappa_hat` e `TS`.

### `outputs/null_calibration.csv`

Distribuição empírica de `TS` e `kappa_hat` em pseudoexperimentos gerados sob `H0`.

### `outputs/profile_likelihood.png`

Perfil de `Delta(-2 ln L)` para `kappa`, com limiar aproximado de 95%.

## 10. Interpretação

- `kappa_hat` próximo de zero e valor-p alto: sem evidência de atenuação adicional;
- `kappa_hat > 0` com valor-p pequeno: efeito candidato, ainda sujeito a sistemáticas;
- inconsistência entre subgrupos de energia ou hemisfério: possível viés/modelagem inadequada;
- resultado sintético positivo: valida somente o pipeline, não a hipótese física.

Não reivindique descoberta usando apenas `Z = sqrt(TS)`. Priorize a calibração empírica e análise sistemática.

## 11. Limitações científicas

1. O crossmatch por RA, DEC e redshift não reconstrói uma linha de visada 3D contínua.
2. Um FRB próximo angularmente é apenas um proxy esparso de DM.
3. A modelagem não corrige automaticamente DM galáctica, halo ou hospedeira.
4. A energia não aparece como `E^-gamma` explícito; assume-se que `counts_signal_pred` incorpora o modelo espectral e a resposta.
5. A resposta oficial do detector, resolução angular/energética e função de seleção devem ser incluídas numa análise real.
6. Nuisances globais simples não representam covariâncias completas.
7. O bootstrap não substitui calibração paramétrica, estudos de cobertura ou validação cega.

## 12. Solução de problemas

### `Menos de 10 correspondências válidas`

- aumente com cautela `max_radius_deg` ou `max_redshift_difference`;
- verifique unidades e cobertura dos catálogos;
- não aumente tolerâncias apenas para obter significância.

### `Falha de convergência`

- verifique valores nulos ou extremos;
- reduza `kappa_max`;
- inspecione sinal e fundo previstos;
- execute o ajuste em uma subamostra.

### Execução muito lenta

Reduza `bootstrap_samples` e `null_simulations` durante desenvolvimento.

### Importação de `src` falha

Execute os comandos a partir da raiz do projeto e confirme que o ambiente virtual está ativo.

## 13. Reprodutibilidade

- preserve `random_seed`;
- registre versões com `pip freeze > environment-lock.txt`;
- registre DOI, URL, licença e SHA256 dos dados;
- não sobrescreva resultados sem versionamento;
- use commits/tags para cada análise final.

## 14. Desenvolvimento

Execute os testes:

```bash
pytest -q
```

Execute um teste específico:

```bash
pytest -q tests/test_inference.py
```

Antes de enviar alterações:

```bash
python -m compileall src scripts tests
pytest -q
```

## 15. Próximos passos

- reconstrução contínua/3D da coluna de plasma;
- integração com liberações públicas documentadas;
- resposta em energia e direção;
- nuisances correlacionados;
- calibração com maior número de pseudoexperimentos;
- testes cegos e reprodução independente.

## 16. Licença e citação

Defina uma licença no repositório antes da distribuição. Para uso científico, cite também as liberações originais dos dados e softwares empregados.
