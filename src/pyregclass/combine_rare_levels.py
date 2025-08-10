import pandas as pd
from typing import Union

def combine_rare_levels(x: Union[pd.Series, list], threshold: int = 20, newname: str = "Combined") -> pd.Series:
    """
    Combine rare categories in a pandas Series or list.

    Args:
        x (Union[pd.Series, list]): The input data.
        threshold (int, optional): The frequency threshold for combining categories. Defaults to 20.
        newname (str, optional): The name for the combined category. Defaults to "Combined".

    Returns:
        pd.Series: The modified Series with rare categories combined.

    Usage Example: 
        >>> import pandas as pd
        >>> f = pd.Series(["A", "A", 'B', 'C'])
        >>> combine_rare_levels(f, threshold=1, newname="Rare")
    """
    s = pd.Series(x).astype(object)
    counts = s.value_counts(dropna=False)
    rare = list(counts[counts <= threshold].index)
    return s.replace(rare, newname)
