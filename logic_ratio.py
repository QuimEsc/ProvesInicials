import numpy as np
import pandas as pd

from data_manager import get_data


def get_ratio_data(base_ticker: str, quote_ticker: str) -> pd.DataFrame:
    base = get_data(base_ticker).copy()
    quote = get_data(quote_ticker).copy()

    if base.empty or quote.empty:
        raise ValueError("No hi ha dades suficients per al càlcul del ratio.")

    base = base[["open", "high", "low", "close"]].copy()
    quote = quote[["open", "high", "low", "close"]].copy().add_suffix("_Q")

    df = base.join(quote, how="left")

    df["close_Q"] = pd.to_numeric(df["close_Q"], errors="coerce").ffill().bfill()
    for col in ["open_Q", "high_Q", "low_Q"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df["close_Q"])

    ratio = pd.DataFrame(index=df.index)

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio["open"] = df["open"] / df["open_Q"]
        ratio["high"] = df["high"] / df["high_Q"]
        ratio["low"] = df["low"] / df["low_Q"]
        ratio["close"] = df["close"] / df["close_Q"]

    ratio = ratio.replace([np.inf, -np.inf], np.nan).dropna(how="all")
    ratio = ratio.ffill().bfill()

    valid_close = ratio["close"].dropna()
    if valid_close.empty:
        raise ValueError("No s'ha pogut normalitzar el ratio perquè no hi ha dades vàlides.")

    base0 = float(valid_close.iloc[0])
    if abs(base0) < 1e-12:
        raise ValueError("El valor inicial del ratio és zero o invàlid.")

    ratio = (ratio / base0) * 100.0
    ratio = ratio.ffill().bfill().dropna()

    out = ratio.reset_index()
    time_col = out.columns[0]
    out[time_col] = pd.to_datetime(out[time_col]).dt.strftime("%Y-%m-%d")
    out = out.rename(columns={time_col: "time"})
    return out[["time", "open", "high", "low", "close"]]