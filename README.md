
# regclass_py

This is a lightweight Python package containing simplified ports of ~20 utilities
inspired by the R package `regclass`. Implementations are intentionally minimal
and educational; they are NOT full feature-complete clones.

Dependencies:
- numpy
- pandas
- scipy
- statsmodels
- matplotlib

Install:
```bash
pip install -r requirements.txt
# or
pip install numpy pandas scipy statsmodels matplotlib
```

Usage:
```python
import pandas as pd
from regclass_py import combine_rare_levels, check_regression
# ...
```
