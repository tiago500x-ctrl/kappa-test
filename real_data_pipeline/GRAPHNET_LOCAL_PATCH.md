# Patch local do GraphNeT (não é um PR upstream)

Documenta a instalação do [GraphNeT](https://github.com/graphnet-team/graphnet) (commit `f7caa15`, 2026-10-02) neste ambiente (Python 3.14, sem GPU), com `DynEdge` rodando um forward pass completo.

**Isto não foi submetido ao GraphNeT.** É um patch local, num venv isolado (`/home/tiago/.venv-graphnet`, fora do repositório), só para avaliar se o GraphNeT seria utilizável aqui — ver issue [#18](https://github.com/tiago500x-ctrl/kappa-test/issues/18) para a decisão completa sobre uso no projeto.

## O que foi necessário

1. **Sistema:** `build-essential`, `python3-dev`, `python3.14-venv` (compilação de numpy/extensões).
2. **Venv isolado** fora do repo, em disco com espaço suficiente — `torch`+CUDA somam vários GB (o `/tmp` do WSL é tmpfs limitado a ~4GB, insuficiente).
3. **`torch==2.14.1`** em vez do `2.7.0` fixado pelo GraphNeT (não existe wheel de 2.7.0 para Python 3.14 — só a partir de 2.9.0).
4. **`torch_geometric` sem `pyg_lib`/`torch_scatter`/`torch_sparse`** (nenhum tem wheel para essa combinação) — ver `graphnet_local_patch.diff`:
   - `torch_scatter.scatter_{min,max,sum,mean}` → `torch_geometric.utils.scatter(..., reduce=...)`, em 6 arquivos (`dynedge.py`, `dynedge_jinst.py`, `dynedge_kaggle_tito.py`, `particlenet.py`, `models/utils.py`, `components/grit_layers.py`) — migração mecânica, mesma que [a issue do GraphNeT](https://github.com/graphnet-team/graphnet) já propunha para abandonar essas libs deprecadas.
   - `torch_sparse.coalesce` → `torch_geometric.utils.coalesce` (`components/embedding.py`).
   - `torch_sparse` em `add_full_rrwp` (GRIT): import tornado preguiçoso (função não usada pelo DynEdge).
   - **`knn_graph`** (`components/edge_convolutions.py`): sem substituto nativo no `torch_geometric` sem `pyg_lib`. Escrito um fallback força-bruta O(n²) em PyTorch puro (`torch.cdist` + `topk`), usado automaticamente só quando `pyg_lib` não está disponível (`torch_geometric.typing.WITH_KNN == False`) — não afeta ambientes onde `pyg_lib` existe.

## Resultado

```python
from graphnet.models.gnn.dynedge import DynEdge
m = DynEdge(nb_inputs=6, global_pooling_schemes=['min','max','mean','sum'])
# ... forward pass com grafo sintético de 50 nós:
out = m(data)  # shape (1, 128), grad_fn presente — backprop funcional
```

Forward pass completo funcionando. Não testado: treinamento real, outros modelos do GraphNeT além de `DynEdge`/`DynEdgeJINST`/`DynEdgeTITO`/`ParticleNeT` (`icemix`, `RNN_tito`, `convnet` não foram exercitados).

## Limitações do patch

- O fallback de `knn_graph` é O(n²) — inviável para eventos reais do IceCube com milhares de hits por evento sem reescrever para batched/aproximado.
- Não testado com dados reais do IceCube (nem MC nem eventos reais) — só com tensores aleatórios sintéticos, para confirmar que a arquitetura roda.
- `torchscale`, `fairscale`, `timm`, `torchvision` foram instalados como dependências transitivas de `torchscale==0.2.0` (exigido pelo `graphnet` base), mas não são relevantes para `DynEdge`.

## Arquivos

- `graphnet_local_patch.diff` — diff completo contra o commit `f7caa15` do GraphNeT.
- Kernel Jupyter registrado: `graphnet (torch 2.14 + torch_geometric, patch local)`.
