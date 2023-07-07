"""
d23a:
    Functions that support the analysis contained in the d23a-mcdc repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""

from functools import cache
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
import statsmodels.api as sm
from watermark import watermark
import xarray as xr


# Matplotlib settings
plt.rcParams['axes.titlesize'] = 'x-large'
plt.rcParams['axes.titleweight'] = 'bold'
plt.rcParams['axes.labelsize'] = 'large'
plt.rcParams['axes.labelweight'] = 'bold'
plt.rcParams['axes.labelcolor'] = '0.15'
plt.rcParams['mathtext.default'] = 'default'
plt.rcParams['savefig.dpi'] = 300


# Constants
IN_BASE = pathlib.Path.cwd() / 'data'  # base directory of input data produced by data_d22a.ipynb

SYMBOLS_DICT = {'Ep': "$E \prime$",  # TOA radiative flux
                'E': "$\Delta E$",  # change in Earth system energy
                'Hp': "$H \prime$",  # sea-surface heat flux
                'H': "$\Delta H$",  # change in ocean heat content
                'Z': "$\Delta Z$",  # change in thermosteric sea level
                'eta': "$\eta$",  # fraction of excess energy absorbed by the ocean
                'eps': "$\epsilon$",  # ocean expansion efficiency of heat
                }

UNITS_DICT = {'Ep': "W m$^{-2}$",
              'E': "YJ",
              'Hp': "W m$^{-2}$",
              'H': "YJ",
              'Z': "mm",
              'eta': "unitless",
              'eps': "mm YJ$^{-1}$",
              }

SAMPLE_N = 100  # number of drift samples to drawn

RNG = np.random.default_rng(12345)  # random number generator


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,scipy,seaborn,statsmodels,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


@cache
def get_calendar_days_branch(esm='UKESM1-0-LL_r1i1p1f2'):
    """Get calendar, number of days in year, and parent branch year for historical simulation of an ESM."""
    # Read historical Dataset
    in_fns = sorted(IN_BASE.glob(f'*/zostoga/{esm}/*_{esm}_historical.mergetime.nc'))
    if len(in_fns) != 1:
        print(f'et_calendar_ndays_branch({esm}): {len(in_fns)} files found for zostoga; using {in_fns[0]}')
    in_ds = xr.open_dataset(in_fns[0])
    # Calendar
    calendar = in_ds['time'].attrs['calendar']
    # Number of days in year
    if '_day' in calendar:
        days_in_yr = int(calendar[0:3])
    elif 'gregorian' in calendar:
        days_in_yr = 365.25
    else:
        print(f'get_calendar_days_branch({esm}): calendar {calendar} not recognised; assuming 365.25 days in year')
        days_in_yr = 365.25
    # Branch year
    parent_time_units = in_ds.attrs['parent_time_units']
    branch_time_in_parent = in_ds.attrs['branch_time_in_parent']
    if parent_time_units[0:10] == 'days since':
        branch_yr = round(int(parent_time_units[11:15]) + float(branch_time_in_parent) / days_in_yr)
    else:
        print(f'get_calendar_days_branch({esm}): parent_time_units {parent_time_units} not recognised')
        branch_yr = None
    return calendar, days_in_yr, branch_yr


@cache
def get_cmip6_df(esm=True, scenario=True):
    """Get CMIP6 ESM data, including reading data and basic processing.

    Parameters
    ----------
    esm : str, tuple, or True
        ESM variant or tuple of ESM variants. Default is True, which corresponds to all available ESMs.
    scenario : str, tuple, or True
        Scenario / experiment or tuple of scenarios. Default is True, which corresponds to control, historical,
        and Tier 1 SSPs.

    Returns
    -------
    cmip6_df : DataFrame
        A DataFrame containing Ep, E, Hp, H, and Z.

    Notes
    -----
    - E, H, and Z are referenced to the 1995-2014 mean.
    """
    # Create DataFrame to hold data
    col_list = ['ESM', 'Scenario', 'Year', 'Ep', 'E', 'Hp', 'H', 'Z', 'convert_Wm2yr_YJ']
    cmip6_df = pd.DataFrame(columns=col_list)
    # If esm is True, update esm to include all available ESMs (based on zostoga availability)
    if esm is True:
        esm = tuple(sorted([d.name for d in IN_BASE.glob(f'regrid_missto0_yearmean_mergetime/zostoga/[!.]*_r*')]))
    # If scenario is True, update scenario to include control, historical, and Tier 1 SSPs
    if scenario is True:
        scenario = ('piControl', 'historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')
    # If esm is tuple, call recursively
    if isinstance(esm, tuple):
        for esm1 in esm:
            temp_df = get_cmip6_df(esm=esm1, scenario=scenario)
            cmip6_df = pd.concat([cmip6_df, temp_df], ignore_index=True)
    # If scenarios is tuple, call recursively
    elif isinstance(scenario, tuple):
        for scenario1 in scenario:
            temp_df = get_cmip6_df(esm=esm, scenario=scenario1)
            cmip6_df = pd.concat([cmip6_df, temp_df], ignore_index=True)
    # Otherwise, proceed to read uncorrected data for single ESM and single scenario
    else:
        # Read input data for variables of interest
        in_da_list = []
        for cmip6_var in ['rsdt', 'rsut', 'rlut', 'hfds', 'hfcorr', 'zostoga']:
            try:
                in_fns = sorted(IN_BASE.glob(f'*/{cmip6_var}/{esm}/*_{esm}_{scenario}.mergetime.nc'))
                if len(in_fns) != 1:
                    print(f'get_cmip6_df({esm}, {scenario}): {len(in_fns)} files found for {cmip6_var}; '
                          f'using {in_fns[0].name}')
                in_da = xr.open_dataset(in_fns[0])[cmip6_var].squeeze()  # read data and drop degenerate dimensions
                in_da['time'] = (in_da.time // 1e4).astype(int)  # convert time units to year
                if (in_da**2).sum() == 0:  # skip if data are all zero
                    print(f'get_cmip6_df({esm}, {scenario}): ignoring {cmip6_var}, which is always zero')
                elif any(in_da.isnull().data):  # are NaNs present?
                    print(f'get_cmip6_df({esm}, {scenario}): ignoring {cmip6_var}, which has missing data')
                else:
                    in_da_list.append(in_da)
            except IndexError:
                if cmip6_var != 'hfcorr':
                    print(f'get_cmip6_df({esm}, {scenario}): no {cmip6_var} data found')
        # Merge DataArrays of different into single Dataset, using common years
        in_ds = xr.merge(in_da_list, join='inner')
        # Shift control according to branch year (derived from historical)
        if scenario == 'piControl':
            _, _, branch_yr = get_calendar_days_branch(esm=esm)
            if branch_yr != 1850:
                old_start_yr = in_ds.time.data[0]
                in_ds['time'] = in_ds.time - branch_yr + 1850
                print(f'get_cmip6_df({esm}, {scenario}): '
                      f'shifted start year from {old_start_yr} to {in_ds.time.data[0]}')
        # Limit SSPs to 2100 for consistency
        if ('ssp' in scenario) and (in_ds.time[-1] > 2100):
            in_ds = in_ds.where(in_ds.time <= 2100, drop=True)
        # If there are gaps, then use period before gap
        intervals = in_ds.time.data[1:] - in_ds.time.data[:-1]
        if np.any(intervals != 1):
            gap_i = int(np.where(intervals != 1)[0][0])  # identify first gap
            gap_yr = in_ds.time.data[gap_i]  # final year before gap
            len_old = len(in_ds.time)
            in_ds = in_ds.where(in_ds.time <= gap_yr, drop=True)  # limit data
            len_new = len(in_ds.time)
            print(f'get_cmip6_df({esm}, {scenario}): '
                  f'gap after {gap_yr}; using period before gap; length {len_old} yr -> {len_new} yr')
        # Locate and correct any discontinuities > 20 mm (0.020 m) in zostoga time series
        diffs = in_ds['zostoga'].data[1:] - in_ds['zostoga'].data[:-1]
        if np.any(np.abs(diffs) > 0.020):
            for dis_i in np.where(np.abs(diffs) > 0.020)[0]:  # loop over locations of discontinuities
                dis_yr = in_ds.time.data[dis_i+1]  # year of discontinuity
                old_val = in_ds['zostoga'].data[dis_i+1]
                new_val = in_ds['zostoga'].data[dis_i] + diffs[dis_i-1]  # extrapolate using previous year's diff
                correction = new_val - old_val
                in_ds['zostoga'].data[dis_i+1:] += correction
                print(f'get_cmip6_df({esm} {scenario}): '
                      f'shifted yr-{dis_yr} zostoga from {old_val*1e3:0.1f} to {new_val*1e3:0.1f} mm')
        # Area or earth and conversion factor for W m-2 yr -> YJ
        area_df = pd.read_csv(IN_BASE / 'area_earth.csv')  # get area of earth...
        area_earth = area_df.loc[area_df['source_member'] == esm]['area_earth'].values[0]  # ... for this ESM
        _, days_in_yr, _ = get_calendar_days_branch(esm=esm)
        convert_Wm2yr_YJ = area_earth * days_in_yr * 24 * 60 * 60 / 1e24  # convert W m-2 yr -> YJ
        # Map cmip6 variables to Ep, E, Hp, H, and Z
        Ep = (in_ds['rsdt'] - in_ds['rsut'] - in_ds['rlut']).rename('Ep') / area_earth  # W -> W m-2
        try:
            Hp = (in_ds['hfds'] + in_ds['hfcorr']).rename('Hp') / area_earth
            print(f'get_cmip6_df({esm}, {scenario}): applied flux correction when calculating Hp')
        except KeyError:
            Hp = in_ds['hfds'].rename('Hp') / area_earth
        E = Ep.cumsum().rename('E') * convert_Wm2yr_YJ  # W m-2 yr -> JY
        H = Hp.cumsum().rename('H') * convert_Wm2yr_YJ
        Z = in_ds['zostoga'].rename('Z') * 1e3  # m -> mm
        # Reference non-flux data 1995-2014 mean
        for da in [E, H, Z]:
            da -= da.sel(time=slice(1995, 2014)).mean()
        # Save to DataFrame
        cmip6_df['Year'] = Ep.time
        cmip6_df['Ep'] = Ep.data
        cmip6_df['E'] = E.data
        cmip6_df['Hp'] = Hp.data
        cmip6_df['H'] = H.data
        cmip6_df['Z'] = Z.data
        cmip6_df['ESM'] = esm
        cmip6_df['Scenario'] = scenario
        cmip6_df['convert_Wm2yr_YJ'] = convert_Wm2yr_YJ
    return cmip6_df


@cache
def sample_drift(esm='UKESM1-0-LL_r1i1p1f2', variable='E', degree=1, sample_n=SAMPLE_N, plot=False):
    """Sample drift of a control simulation, using OLS with HAC. Returns samples as DataArray."""
    # Get control time series and convert to DataArray
    pi_da = get_cmip6_df(esm=esm, scenario='piControl').set_index('Year')[variable].to_xarray()
    # If degree is an integer, then use a polynomial of that degree
    if isinstance(degree, int):
        # Predictors; t refers to year/time dimension, k to order of polynomial term
        x_tk = np.stack([(pi_da.Year - 2005)**k for k in range(degree+1)]).transpose()  # use 2005 as base year
        # OLS fit to full control time series, with heteroskedasticity-autocorrelation robust covariance
        sm_reg = sm.OLS(pi_da.data, x_tk).fit().get_robustcov_results(cov_type='HAC', maxlags=100)
        # Sample parameters and standard error, assuming Gaussian distribution; n refers to sample draw
        params_nk = RNG.normal(loc=sm_reg.params, scale=sm_reg.bse, size=(sample_n, degree+1))
        # Sample drift using the parameter samples
        drift_nt = (params_nk[:, np.newaxis, :] * x_tk[np.newaxis, :, :]).sum(axis=2)  # array
        drift_da = xr.DataArray(drift_nt, dims=['Draw', 'Year'],
                                coords={'Draw': range(sample_n), 'Year': pi_da.Year})  # DataArray
    # If degree is 'int.-bias', apply integrated-bias method
    elif degree == 'int.-bias':
        # Call recursively to get bias samples of corresponding flux variable
        bias_da = sample_drift(esm=esm, variable=f'{variable}p', degree=0, sample_n=sample_n)
        # Cumulatively integrate bias samples
        convert_Wm2yr_YJ = get_cmip6_df(esm=esm, scenario='piControl')['convert_Wm2yr_YJ'][0]
        drift_da = bias_da.cumsum(dim='Year') * convert_Wm2yr_YJ
    # If degree is 'agnostic', apply agnostic method using polynomials of degree 1-3
    elif degree == 'agnostic':
        # Create list to hold drift_da for each degree
        drift_da_list = []
        # Sample using different degrees (1-3), calling recursively
        for deg in range(1, 4):
            sub_n = sample_n // 3  # number of (sub)samples corresponding to this polynomial fit
            if deg == 3:  # total number of samples should correspond to sample_n
                sub_n = sample_n - 2 * sub_n
            temp_da = sample_drift(esm=esm, variable=variable, degree=deg, sample_n=sub_n)
            temp_da = temp_da.assign_coords({'Draw': (sub_n * (deg - 1) + np.arange(sub_n))})  # shift Draw coordinate
            drift_da_list.append(temp_da)
        # Concatenate the drift samples
        drift_da = xr.concat(drift_da_list, dim='Draw')
    # If degree != 0, reference to 1995-2014 mean
    if degree != 0:
        drift_da -= drift_da.sel(Year=slice(1995, 2014)).mean(dim='Year')
    # Plot?
    if plot:
        pi_da.plot(color='0.2', label=f'{esm} control')
        for i in range(sample_n):
            if i == 0:
                label = f'Drift samples (n = {sample_n}; degree = {degree})'
            else:
                label = None
            drift_da.isel(Draw=i).plot(color='r', alpha=10/sample_n, label=label)
        drift_da.mean(dim='Draw').plot(color='blue', label='Mean of samples')
        plt.legend()
        plt.ylabel(f'{variable} ({UNITS_DICT[variable]})')
        plt.show()
    return drift_da


@cache
def sample_corrected(esm='UKESM1-0-LL_r1i1p1f2', variable='E', degree=1, scenario='historical',
                     sample_n=SAMPLE_N, plot=False):
    """Apply MCDC to get drift corrected samples. Returns samples as DataArray."""
    # Get uncorrected time series for scenario and convert to DataArray
    uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[variable].to_xarray()
    # Get drift samples
    drift_da = sample_drift(esm=esm, variable=variable, degree=degree, sample_n=sample_n, plot=False)
    # Apply drift correction
    corr_da = uncorr_da - drift_da
    # If degree != 0, reference to 1995-2014 mean
    if degree != 0:
        corr_da -= corr_da.sel(Year=slice(1995, 2014)).mean(dim='Year')
    # Plot?
    if plot:
        uncorr_da.plot(color='0.2', label=f'{esm} {scenario}')
        for i in range(sample_n):
            if i == 0:
                label = f'Corrected samples (n = {sample_n}; degree = {degree})'
            else:
                label = None
            corr_da.isel(Draw=i).plot(color='r', alpha=10/sample_n, label=label)
        corr_da.mean(dim='Draw').plot(color='blue', label='Mean of samples')
        plt.legend()
        plt.ylabel(f'{variable} ({UNITS_DICT[variable]})')
        plt.show()
    return corr_da
