import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from scipy import stats
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

def check_regression(model):
    """
    Regression diagnostics for a fitted statsmodels OLS model.

    Parameters
    ----------
    model : statsmodels.regression.linear_model.RegressionResults
        A fitted OLS model from statsmodels.
    """
    residuals = model.resid
    fitted = model.fittedvalues
    standardized_residuals = residuals / np.std(residuals)

    # 1. Residuals vs Fitted
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 2, 1)
    plt.scatter(fitted, residuals)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Fitted')

    # 2. Normal Q-Q
    plt.subplot(2, 2, 2)
    sm.qqplot(residuals, line='45', fit=True, ax=plt.gca())
    plt.title('Normal Q-Q')

    # 3. Scale-Location
    plt.subplot(2, 2, 3)
    plt.scatter(fitted, np.sqrt(np.abs(standardized_residuals)))
    plt.xlabel('Fitted values')
    plt.ylabel('Sqrt(|Standardized residuals|)')
    plt.title('Scale-Location')

    # 4. Residuals vs Leverage
    plt.subplot(2, 2, 4)
    sm.graphics.influence_plot(model, ax=plt.gca(), criterion="cooks")

    plt.tight_layout()
    plt.show()

    # Statistical tests
    print("Shapiro-Wilk test for normality:", stats.shapiro(residuals))
    bp_test = het_breuschpagan(residuals, model.model.exog)
    print("Breusch-Pagan test (LM, p-value, F, p-value):", bp_test)
    print("Durbin-Watson statistic:", durbin_watson(residuals))