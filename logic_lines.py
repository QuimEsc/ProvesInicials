import pandas as pd


_MIN_EXPANDING = 10

_COMPRA_QUANTILS = [
    ("c1", 0.30),
    ("c2", 0.20),
    ("c3", 0.10),
    ("c4", 0.05),
]

_VENDA_QUANTILS = [
    ("v1", 0.60),
    ("v2", 0.70),
    ("v3", 0.80),
    ("v4", 0.90),
]


def _series_to_chart(s: pd.Series, nom: str) -> pd.DataFrame:
    out = s.dropna().reset_index()
    if out.empty:
        return pd.DataFrame({"time": pd.Series(dtype="str"), nom: pd.Series(dtype="float64")})

    time_col = out.columns[0]
    value_col = out.columns[1]

    out[time_col] = pd.to_datetime(out[time_col]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={time_col: "time", value_col: nom})
    return out[["time", nom]]


def get_lines(df: pd.DataFrame, rang_c: int = 800, rang_v: int = 800) -> dict:
    d = df[["high", "low"]].copy()

    # 1. Calculamos los extremos absolutos de la ventana (Max de Highs, Min de Lows)
    d["Max_C"] = d["high"].rolling(window=int(rang_c), min_periods=1).max()
    d["Min_V"] = d["low"].rolling(window=int(rang_v), min_periods=1).min()

    # 2. Cruzamos las variables para capturar la volatilidad extrema (tu nueva propuesta)
    d["Var_Compra"] = d["low"] / d["Max_C"]
    d["Var_Venda"] = d["high"] / d["Min_V"]

    resultats = {}

    for nom, q in _COMPRA_QUANTILS:
        serie = d["Var_Compra"].expanding(min_periods=_MIN_EXPANDING).quantile(q) * d["Max_C"]
        resultats[nom] = _series_to_chart(serie, nom)

    for nom, q in _VENDA_QUANTILS:
        serie = d["Var_Venda"].expanding(min_periods=_MIN_EXPANDING).quantile(q) * d["Min_V"]
        resultats[nom] = _series_to_chart(serie, nom)

    return resultats