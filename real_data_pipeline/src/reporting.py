from pathlib import Path
import json, matplotlib.pyplot as plt

def save_outputs(result,grid,vals,result_path,plot_path):
    Path(result_path).parent.mkdir(parents=True,exist_ok=True); Path(plot_path).parent.mkdir(parents=True,exist_ok=True)
    Path(result_path).write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding="utf-8")
    fig,ax=plt.subplots(figsize=(8,5)); ax.plot(grid,vals); ax.axhline(3.841458820694124,color="gray",ls="--",label="Limiar 95%")
    ax.axvline(result["kappa_hat"],color="red",ls=":",label=f'κ̂={result["kappa_hat"]:.4g}')
    ax.set(xlabel="κ por 10²¹ cm⁻²",ylabel="Δ(-2 ln L)",title="Perfil da verossimilhança para κ"); ax.legend(); fig.tight_layout(); fig.savefig(plot_path,dpi=180); plt.close(fig)
