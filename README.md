# d22a-mcdc: Analysis Code for "Monte Carlo Drift Correction – Quantifying the Drift Uncertainty of Global Climate Models"

[![DOI](https://zenodo.org/badge/521571893.svg)](https://zenodo.org/badge/latestdoi/521571893)

## Usage guidelines
This repository accompanies the following manuscript:

Grandey, B. S., Koh, Z. Y., Samanta, D., Horton, B. P., Dauwels, J., and Chew, L. Y. (2013),  **Monte Carlo Drift Correction – Quantifying the Drift Uncertainty of Global Climate Models**, _EGUsphere [preprint]_, https://doi.org/10.5194/egusphere-2022-1515.

The [manuscript](https://doi.org/10.5194/egusphere-2022-1515) serves as the primary reference.
The [Zenodo archive](https://doi.org/10.5281/zenodo.7488334) of this repository serves as a secondary reference.

The [**`data/`**](data/) folder contains post-processed _CMIP6_ climate model data.
Users of these data should note the [CMIP6 Terms of Use](https://pcmdi.llnl.gov/CMIP6/TermsOfUse/TermsOfUse6-2.html).

## Workflow

### Setup
To create a _conda_ environment with the necessary software dependencies, use the [**`environment.yml`**](environment.yml) file:

```
conda env create --file environment.yml
conda activate d22a-mcdc
```

The analysis has been performed within this environment on _macOS 12_.

_Windows_ users may need to remove the references to _Climate Data Operators_ (`cdo`, `python-cdo`) in [`environment.yml`](environment.yml).
CDO is required by [`data_d22a.ipynb`](data_d22a.ipynb), but not by [`mcdc_analysis_d22a.ipynb`](mcdc_analysis_d22a.ipynb).

### Preparation of data
_CMIP6_ climate model data have been downloaded, post-processed, and prepared as follows:

1. Data have been **downloaded** from the _Earth System Grid Federation (ESGF)_ using the _ESGF PyClient_ and _Globus_ (see [**p22b-esgf-globus v0.2.0**](https://github.com/grandey/p22b-esgf-globus/tree/v0.2.0)).

2. Data have been **post-processed** using _Climate Data Operators (CDO)_.
This includes the following steps:
(i) calculate annual means,
(ii) multiply each flux variable with the corresponding grid cell area, then
(iii) sum globally
(see [**p22c-esgf-processing v0.2.0**](https://github.com/grandey/p22c-esgf-processing/tree/v0.2.0)).

3. Data have then been prepared for further analysis using [**`data_d22a.ipynb`**](data_d22a.ipynb) (in this repository).

The NetCDF files produced by [`data_d22a.ipynb`](data_d22a.ipynb) can be found in [**`data/`**](data/).

### Analysis
Analysis of the data in [`data/`](data/) is performed using [**`mcdc_analysis_d22a.ipynb`**](mcdc_analysis_d22a.ipynb).

[`mcdc_analysis_d22a.ipynb`](mcdc_analysis_d22a.ipynb) contains the Monte Carlo Drift Correction (MCDC) functions, and it writes both figures (in [**`figs_d22a/`**](figs_d22a/)) and tables (in [**`tables_d22a/`**](tables_d22a/)).

Are the trend-method MCDC results sensitive to the choice of estimation method?
The analysis has been re-run using a robust linear model with Huber’s T norm (instead of ordinary least squares): [**`rlm/`**](rlm/).
The results are similar.

## Author
[Benjamin S. Grandey](https://grandey.github.io) (_Nanyang Technological University_), in collaboration with Zhi Yang Koh, Dhrubajyoti Samanta, Benjamin P. Horton, Justin Dauwels, and Lock Yue Chew.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).

We acknowledge the World Climate Research Programme, which, through its Working Group on Coupled Modelling, coordinated and promoted CMIP6. We thank the climate modeling groups for producing and making available their model output, the Earth System Grid Federation (ESGF) for archiving the data and providing access, and the multiple funding agencies who support CMIP6 and ESGF.
