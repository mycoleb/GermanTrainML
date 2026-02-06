import numpy as np
import pandas as pd

def to_float32(X):
    """
    Convert pandas/numpy inputs to float32 without lambdas (pickle-safe).
    Handles arrow/string dtypes by coercing non-numeric to NaN.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X = X.apply(pd.to_numeric, errors="coerce")
        return X.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(X, dtype=np.float32)
