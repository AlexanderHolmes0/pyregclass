import numpy as np
import statsmodels.api as sm
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor

def vif(mod : sm.regression.linear_model.RegressionResultsWrapper):
    """
    Calculate VIF using statsmodels' variance_inflation_factor
    
    Parameters:
        mod: statsmodels regression results object
    
    Returns:
        DataFrame with VIF values for each predictor
    """
    # Get feature matrix X without constant
    X = mod.model.exog
    var_names = mod.model.exog_names
    
    # Remove constant if it exists
    if 'const' in var_names:
        X = X[:, 1:]
        var_names = var_names[1:]
    
    # Calculate VIF for each feature
    vif_data = pd.DataFrame()
    vif_data["Variable"] = var_names
    vif_data["VIF"] = [variance_inflation_factor(X, i) for i in range(X.shape[1])]
    
    return vif_data