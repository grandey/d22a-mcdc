"""
d23a:
    Functions that support the analysis contained in the d23a-mcdc repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from watermark import watermark


# Matplotlib settings
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelcolor'] = '0.15'
plt.rcParams['mathtext.default'] = 'default'
plt.rcParams['savefig.dpi'] = 300

# Seaborn style
sns.set_style('whitegrid')


# Watermark, including versions of dependencies

def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,scipy,seaborn,statsmodels,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)
