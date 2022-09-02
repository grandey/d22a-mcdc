# d22a-mcdc

## Description
Code and analysis for "**_Monte Carlo Drift Correction - Quantifying the Drift Uncertainty of Global Climate Models_**".

## Workflow

### Setup
To create a _conda_ environment with the necessary software dependencies, use the [**`environment.yml`**](environment.yml) file:

```
conda env create --file environment.yml
```

### Preparation of data
_CMIP6_ climate model data have been downloaded, post-processed, and prepared as follows:

1. Data have been **downloaded** from the _Earth System Grid Federation (ESGF)_ using the _ESGF PyClient_ and _Globus_ (see [**p22b-esgf-globus v0.1.0**](https://github.com/grandey/p22b-esgf-globus/tree/v0.1.0)).

2. Data have been **post-processed** using _Climate Data Operators (CDO)_, including
(i) regridding two-dimensional variables to a regular longitude-latitude grid (using first-order conservative remapping for unstructured grids, using bicubic interpolation for other grids),
(ii) setting missing data to zero before and after regridding (because the sea-surface heat flux is not defined over land),
(iii) calculation of annual means, and
(iv) calculation of area-weighted global means
(see [**p22c-esgf-processing v0.1.0**](https://github.com/grandey/p22c-esgf-processing/tree/v0.1.0)).

3. Data have then been prepared for further analysis by [**`data_d22a.ipynb`**](data_d22a.ipynb) (in this repository).

The NetCDF files produced by [`data_d22a.ipynb`](data_d22a.ipynb) can be found in [**`data/`**](data/).

### Analysis
Analysis of the data in [`data/`](data/) is performed by [**`mcdc_analysis_d22a.ipynb`**](mcdc_analysis_d22a.ipynb).

[`mcdc_analysis_d22a.ipynb`](mcdc_analysis_d22a.ipynb) contains the Monte Carlo Drift Correction functions, and it writes both figures (in [**`figs_d22a/`**](figs_d22a/)) and tables (in [**`tables_d22a/`**](tables_d22a/)).

## Author
B. S. Grandey (_Nanyang Technological University_), in collaboration with co-authors.

B.S.G. is responsible for the contributions contained in this repository.
Co-author contributions to the other aspects of the research project are listed in the manuscript.

## Acknowledgements
This Research/Project is supported by the National Research Foundation, Singapore, and National Environment Agency, Singapore under the National Sea Level Programme Funding Initiative (Award No. USS-IF-2020-3).

We acknowledge the World Climate Research Programme, which, through its Working Group on Coupled Modelling, coordinated and promoted CMIP6. We thank the climate modeling groups for producing and making available their model output, the Earth System Grid Federation (ESGF) for archiving the data and providing access, and the multiple funding agencies who support CMIP6 and ESGF.
