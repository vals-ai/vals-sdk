import pandas as pd


# This is a default code snippet for a custom metric
def custom_metric_function(df: pd.DataFrame) -> float:
    if df.empty or "eval" not in df:
        return 0.0

    total = len(df)

    passed = (df["eval"] > 0.5).sum()

    return (passed / total) * 100
