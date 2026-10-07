def require_columns(df, columns, label):
    missing=[c for c in columns if c not in df.columns]
    if missing: raise ValueError(f"{label}: colunas ausentes: {missing}")
    if df[list(columns)].isna().any().any(): raise ValueError(f"{label}: valores ausentes")

def validate_icecube(df,c):
    require_columns(df,c.values(),"IceCube")
    if (df[c["counts_obs"]]<0).any() or (df[c["signal_pred"]]<0).any() or (df[c["background_pred"]]<0).any(): raise ValueError("Contagens/previsões negativas")
    if (df[c["energy"]]<=0).any(): raise ValueError("Energia deve ser positiva")
    if not df[c["ra"]].between(0,360).all() or not df[c["dec"]].between(-90,90).all(): raise ValueError("Coordenadas inválidas")

def validate_plasma(df,c):
    require_columns(df,c.values(),"Plasma")
    if (df[c["dm"]]<0).any(): raise ValueError("DM negativa")
    if not df[c["ra"]].between(0,360).all() or not df[c["dec"]].between(-90,90).all(): raise ValueError("Coordenadas inválidas")
