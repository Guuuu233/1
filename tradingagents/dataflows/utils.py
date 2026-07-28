import os
import json
import pandas as pd
from datetime import date, timedelta, datetime
from typing import Annotated

SavePathType = Annotated[str, "File path to save data. If None, data is not saved."]

def save_output(data: pd.DataFrame, tag: str, save_path: SavePathType = None) -> None:
    if save_path:
        data.to_csv(save_path)
        print(f"{tag} saved to {save_path}")


def get_current_date():
    return date.today().strftime("%Y-%m-%d")


def decorate_all_methods(decorator):
    def class_decorator(cls):
        for attr_name, attr_value in cls.__dict__.items():
            if callable(attr_value):
                setattr(cls, attr_name, decorator(attr_value))
        return cls

    return class_decorator


def get_next_weekday(date):

    if not isinstance(date, datetime):
        date = datetime.strptime(date, "%Y-%m-%d")

    if date.weekday() >= 5:
        days_to_add = 7 - date.weekday()
        next_weekday = date + timedelta(days=days_to_add)
        return next_weekday
    else:
        return date


def take_latest(df: "pd.DataFrame", date_col: str, n: int = 1) -> "pd.DataFrame":
    """Return the latest ``n`` rows by ``date_col`` regardless of input order.

    External vendors may return ascending, descending, or unsorted frames.
    Always convert to datetime, sort descending, then take the first ``n`` rows.
    Rows with unparseable dates are dropped.

    Selection order is newest-first. For LLM-facing text, pass the result through
    :func:`chronological` so "later rows = later events".
    """
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    if date_col not in df.columns:
        raise KeyError(
            f"take_latest: column {date_col!r} not in DataFrame columns {list(df.columns)}"
        )
    if n is None or int(n) <= 0:
        raise ValueError(f"take_latest: n must be positive, got {n!r}")

    out = df.copy()
    sort_key = "__take_latest_dt"
    out[sort_key] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[sort_key])
    if out.empty:
        return out.drop(columns=[sort_key], errors="ignore")
    out = out.sort_values(sort_key, ascending=False, kind="mergesort")
    out = out.drop(columns=[sort_key])
    return out.head(int(n)).reset_index(drop=True)


def chronological(df: "pd.DataFrame", date_col: str) -> "pd.DataFrame":
    """Sort rows ascending by ``date_col`` for prompt/display (later = more recent)."""
    import pandas as pd

    if df is None or getattr(df, "empty", True):
        return df
    if date_col not in df.columns:
        raise KeyError(
            f"chronological: column {date_col!r} not in DataFrame columns {list(df.columns)}"
        )
    out = df.copy()
    sort_key = "__chrono_dt"
    out[sort_key] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[sort_key])
    if out.empty:
        return out.drop(columns=[sort_key], errors="ignore")
    out = out.sort_values(sort_key, ascending=True, kind="mergesort")
    out = out.drop(columns=[sort_key])
    return out.reset_index(drop=True)
