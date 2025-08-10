import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency, pearsonr

def cramers_v(x, y) -> float:
    """
    Compute Cramér's V statistic for categorical-categorical association.

    Args:
        x (pd.Series): First categorical variable.
        y (pd.Series): Second categorical variable.

    Returns:
        float: Cramér's V statistic.

    """
    table = pd.crosstab(x, y)
    chi2 = chi2_contingency(table)[0]
    n = table.sum().sum()
    phi2 = chi2 / n
    r,k = table.shape
    return np.sqrt(phi2 / min(k-1, r-1)) if min(k-1,r-1)>0 else 0.0

def associate(df, var1, var2) -> dict:
    """
    Simple association measure between two variables:
    - If both categorical -> Cramer's V
    - If both numeric -> Pearson r
    - If mixed -> point-biserial (approx using pearson on codes)

    Args:
        df (pd.DataFrame): The DataFrame containing the variables.
        var1 (str): The name of the first variable.
        var2 (str): The name of the second variable.

    Returns:
        dict: A dictionary containing the association measure and its p-value.
    """
    x = df[var1]
    y = df[var2]
    if (pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)) and (pd.api.types.is_numeric_dtype(y) and not pd.api.types.is_bool_dtype(y)):
        r, p = pearsonr(x.dropna(), y.dropna())
        return {"type":"numeric","pearson_r": r, "pvalue": p}
    elif not pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_numeric_dtype(y):
        return {"type":"categorical","cramers_v": cramers_v(x,y)}
    else:
        # mixed: encode categorical and compute pearson
        if pd.api.types.is_numeric_dtype(x):
            y_enc = pd.Categorical(y).codes
            r, p = pearsonr(x.dropna(), y_enc[:len(x.dropna())])
        else:
            x_enc = pd.Categorical(x).codes
            r, p = pearsonr(x_enc[:len(y.dropna())], y.dropna())
        return {"type":"mixed","pearson_r": r, "pvalue": p}
