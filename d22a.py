"""
d23a:
    Functions that support the analysis contained in the d23a-mcdc repository.

Author:
    Benjamin S. Grandey, 2022-2023.
"""

from functools import cache
import itertools
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pathlib
from scipy import stats
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

REF_YRS = [1850, 1859]  # reference period; start year is used for origin when fitting drift
REF_STR = '1850s'

SCENARIO_DICT = {'piControl': 'Control', 'historical': 'Historical',  # names of scenarios
                 'ssp126': 'SSP1-2.6', 'ssp245': 'SSP2-4.5',
                 'ssp370': 'SSP3-7.0', 'ssp585': 'SSP5-8.5'}
SCENARIO_C_DICT = {'piControl': '0.5', 'historical': 'darkblue',  # colours to use when plotting
                   'ssp126': 'lavender', 'ssp245': 'greenyellow',
                   'ssp370': 'darkorange', 'ssp585': 'darkred'}

DEF_ESM = 'UKESM1-0-LL_r1i1p1f2'  # default ESM

SAMPLE_N = 1500  # number of drift samples to draw

RNG = np.random.default_rng(12345)  # random number generator

FIG_DIR = pathlib.Path.cwd() / 'figs_d22a'  # directory in which to save figures
F_NUM = itertools.count(1)  # main figures counter
S_NUM = itertools.count(1)  # supplementary figures counter
O_NUM = itertools.count(1)  # other figures counter
TABLE_DIR = pathlib.Path.cwd() / 'tables_d22a'  # directory in which to save tables
T_NUM = itertools.count(1)  # main tables counter
ST_NUM = itertools.count(1)  # supplementary tables counter
OT_NUM = itertools.count(1)  # other tables counter


def get_watermark():
    """Return watermark string, including versions of dependencies."""
    packages = 'matplotlib,numpy,pandas,scipy,statsmodels,xarray'
    return watermark(machine=True, conda=True, python=True, packages=packages)


@cache
def get_calendar_days_branch(esm=DEF_ESM):
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
    """
    # Create DataFrame to hold data
    col_list = ['ESM', 'Scenario', 'Year', 'Ep', 'E', 'Hp', 'H', 'Z', 'convert_Wm2yr_YJ']
    cmip6_df = pd.DataFrame(columns=col_list)
    # If esm is True, update esm to include all available ESMs (based on zostoga availability)
    if esm is True:
        esm = tuple(sorted([d.name for d in IN_BASE.glob(f'regrid_missto0_yearmean_mergetime/zostoga/[!.]*_r*')]))
    # If scenario is True, update scenario to include control, historical, and Tier 1 SSPs
    if scenario is True:
        scenario = tuple(SCENARIO_DICT.keys())
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
        # Reference non-flux data reference period
        for da in [E, H, Z]:
            da -= da.sel(time=slice(REF_YRS[0], REF_YRS[1])).mean()
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
def get_esm_info_df():
    """DataFrame containing ESM, variant, control length, calendar, and further info URL."""
    # Columns
    columns = ['Model', 'Variant', 'Control length (yr)', 'Calendar', 'Further information URL']
    # Initialize DataFrame
    info_df = pd.DataFrame(columns=columns)
    info_df = info_df.set_index('Model')
    # Loop over ESMs
    esms = get_cmip6_df(esm=True, scenario=True)['ESM'].unique()
    for esm in esms:
        # Model and variant
        model, variant = esm.split('_')
        # Length of available control data
        control_len = len(get_cmip6_df(esm=esm, scenario='piControl')['Z'])
        # Read control Dataset for Z - used for calendar and further info URL
        in_fns = sorted(IN_BASE.glob(f'*/zostoga/{esm}/*_{esm}_piControl.mergetime.nc'))
        if len(in_fns) != 1:
            print(f'get_esm_info_df(): {len(in_fns)} files found for {esm} piControl zostoga; using {in_fns[0]}')
        in_ds = xr.open_dataset(in_fns[0])
        # Calendar
        calendar = in_ds['time'].attrs['calendar']
        calendar = calendar.replace('_', ' ')  # replace '_' with space
        # Further info URL
        further_info_url = in_ds.attrs['further_info_url']
        # Save to DataFrame
        info_df.loc[model] = {'Variant': variant, 'Control length (yr)': control_len,
                              'Calendar': calendar,
                              'Further information URL': further_info_url}
    return info_df


def get_esm_info_tex():
    """Latex table containing ESM, variant, control length, calendar, and further info URL."""
    # Get DataFrame
    info_df = get_esm_info_df()
    # Caption
    caption = ("Coupled Model Intercomparison Project Phase 6 (CMIP6) models analysed in this study. "
               "``Control length'' refers to the time series length of the pre-industrial control simulation data. "
               "The further information URLs also correspond to the control simulations. ")
    # Convert DataFrame to Latex
    tex_str = info_df.style.to_latex(environment='table*', position='t', position_float='centering',
                                     column_format='ccccc', multicol_align='c', hrules=True,
                                     caption=caption)
    # Manually reformat column titles
    tex_str = tex_str.replace('\n & Variant & Control length (yr) & Calendar & Further information URL',
                              '\nModel & Variant & Control length (yr) & Calendar & Further information URL')
    tex_str = tex_str.replace('\nModel &  &  &  &  \\\\\n', '\n')
    return tex_str


@cache
def sample_drift(esm=DEF_ESM, variable='E', degree='agnostic', sample_n=SAMPLE_N, plot=False):
    """Sample drift of a control simulation, using OLS with HAC. Returns samples as DataArray."""
    # Get control time series and convert to DataArray
    pi_da = get_cmip6_df(esm=esm, scenario='piControl').set_index('Year')[variable].to_xarray()
    # If degree is 'linear', then integer equivalent is 1
    if degree == 'linear':
        degree = 1
    # If degree is an integer, then use a polynomial of that degree
    if isinstance(degree, int):
        # Predictors; t refers to year/time dimension, k to order of polynomial term
        x_tk = np.stack([(pi_da.Year - REF_YRS[0])**k for k in range(degree+1)]).transpose()  # ref to REF_YRS[0]
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
    # If degree != 0, reference to reference period
    if degree != 0:
        drift_da -= drift_da.sel(Year=slice(REF_YRS[0], REF_YRS[1])).mean(dim='Year')
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
def sample_corrected(esm=DEF_ESM, variable='E', degree='agnostic', scenario='historical',
                     sample_n=SAMPLE_N, plot=False):
    """Apply MCDC to get drift corrected samples. Returns samples as DataArray."""
    # Get uncorrected time series for scenario and convert to DataArray
    uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[variable].to_xarray()
    # Get drift samples
    drift_da = sample_drift(esm=esm, variable=variable, degree=degree, sample_n=sample_n, plot=False)
    # Apply drift correction
    corr_da = uncorr_da - drift_da
    # If degree != 0, reference to reference period
    if degree != 0:
        corr_da -= corr_da.sel(Year=slice(REF_YRS[0], REF_YRS[1])).mean(dim='Year')
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


@cache
def sample_target_decade(esm=DEF_ESM, variable='E', degree='agnostic', scenario='historical',
                         target_decade='2000s', sample_n=SAMPLE_N, plot=False):
    """Return decadal-mean drift-corrected samples as DataArray."""
    # Get time series samples
    corr_da = sample_corrected(esm=esm, variable=variable, degree=degree, scenario=scenario,
                               sample_n=sample_n, plot=False)
    # Calculate decadal mean, relative to reference period
    target_start = int(target_decade[0:4])
    target_end = target_start + 9
    decadal_da = (corr_da.sel(Year=slice(target_start, target_end)).mean(dim='Year') -
                  corr_da.sel(Year=slice(REF_YRS[0], REF_YRS[1])).mean(dim='Year'))
    # Plot?
    if plot:
        decadal_da.plot.hist(label=f'{esm} {scenario} (n = {SAMPLE_N}; degree = {degree})')
        plt.legend()
        plt.xlabel(f'{variable} ({UNITS_DICT[variable]}; {target_decade})')
        plt.show()
    return decadal_da


@cache
def sample_eta_eps(esm=DEF_ESM, eta_or_eps='eta', degree='agnostic', scenario='historical',
                   sample_n=SAMPLE_N, plot=False):
    """Using drift-corrected samples, return samples of eta or epsilon coefficients as a DataArray."""
    # Variables to use in calculation
    if eta_or_eps == 'eta':  # eta: H = eta * E
        x_var = 'E'
        y_var = 'H'
    elif eta_or_eps == 'eps':  # epsilon: Z = eps * H
        x_var = 'H'
        y_var = 'Z'
    else:
        raise ValueError(f'sample_eta_eps(): eta_or_eps of {eta_or_eps} not recognised.')
    # Get drift-corrected time-series samples
    x_da = sample_corrected(esm=esm, variable=x_var, degree=degree, scenario=scenario, sample_n=sample_n)
    y_da = sample_corrected(esm=esm, variable=y_var, degree=degree, scenario=scenario, sample_n=sample_n)
    # For SSPs, use data from 2015-2100 only
    if 'ssp' in scenario:
        x_da = x_da.sel(Year=slice(2015, 2100))
        y_da = y_da.sel(Year=slice(2015, 2100))
    # DataArray to hold coefficients (either eta or eps; initialized as zero)
    coeff_da = x_da.mean(dim='Year')  # remove Year dimension
    coeff_da.data[:] = 0.
    # Loop over samples and estimate coefficients using OLS with a free intercept
    for i in range(sample_n):
        x_in = sm.add_constant(x_da.isel(Draw=i).data, prepend=True)
        sm_reg = sm.OLS(y_da.isel(Draw=i).data, x_in).fit()
        coeff_da.data[i] = sm_reg.params[1]
    # Plot?
    if plot:
        coeff_da.plot.hist(label=f'{esm} {scenario} (n = {SAMPLE_N}; degree = {degree})')
        plt.legend()
        plt.xlabel(f'{eta_or_eps} ({UNITS_DICT[eta_or_eps]})')
        plt.show()
    return coeff_da


@cache
def calc_drift_uncertainty(esm=DEF_ESM, variable='E', degree='agnostic', scenario='ssp585',
                           target_decade='2050s', sample_n=SAMPLE_N):
    """Calculate drift uncertainty using 2nd-98th percentiles of drift-corrected samples."""
    # Get data
    if variable in ['eta', 'eps']:
        data_da = sample_eta_eps(esm=esm, eta_or_eps=variable, degree=degree, scenario=scenario,
                                 sample_n=sample_n, plot=False)
    else:
        data_da = sample_target_decade(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                       target_decade=target_decade, sample_n=sample_n, plot=False)
    # Calculate uncertainty using 2nd-98th percentiles of samples
    perc_data = np.percentile(data_da, [2, 98])
    uncertainty = perc_data.max() - perc_data.min()
    return uncertainty


@cache
def calc_scenario_uncertainty(esm=DEF_ESM, variable='E', degree='agnostic', target_decade='2050s', sample_n=SAMPLE_N):
    """Calculate scenario uncertainty using the inter-scenario range across SSPs."""
    # Calculate mean for each SSP
    mean_s = np.zeros(4)  # array to hold mean for each SSP
    for s, scenario in enumerate(['ssp126', 'ssp245', 'ssp370', 'ssp585']):
        # Get data
        if variable in ['eta', 'eps']:
            data_da = sample_eta_eps(esm=esm, eta_or_eps=variable, degree=degree, scenario=scenario,
                                     sample_n=sample_n, plot=False)
        else:
            data_da = sample_target_decade(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                           target_decade=target_decade, sample_n=sample_n, plot=False)
        # Calculate mean
        mean_s[s] = data_da.mean()
    # Calculate uncertainty using inter-scenario range
    uncertainty = mean_s.max() - mean_s.min()
    return uncertainty


@cache
def calc_model_uncertainty(variable='E', degree='agnostic', scenario='ssp585', target_decade='2050s',
                           sample_n=SAMPLE_N):
    """Calculate model uncertainty using the inter-model range across ensemble."""
    # Calculate mean for each ESM
    esms = get_cmip6_df(esm=True, scenario=True)['ESM'].unique()  # list of ESMs
    mean_e = np.zeros(len(esms))  # array to hold mean for each ESM
    for e, esm in enumerate(esms):
        # Get data
        if variable in ['eta', 'eps']:
            data_da = sample_eta_eps(esm=esm, eta_or_eps=variable, degree=degree, scenario=scenario,
                                     sample_n=sample_n, plot=False)
        else:
            data_da = sample_target_decade(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                           target_decade=target_decade, sample_n=sample_n, plot=False)
        # Calculate mean
        mean_e[e] = data_da.mean()
    # Calculate uncertainty using inter-model range
    uncertainty = mean_e.max() - mean_e.min()
    return uncertainty


@cache
def get_detailed_df(variable='E', target_decade='2050s', sample_n=SAMPLE_N):
    """Detailed DataFrame showing drift, model, and scenario uncertainty."""
    # Lists of ESMs and scenarios
    esms = get_cmip6_df(esm=True, scenario=True)['ESM'].unique()
    if variable in ['eta', 'eps']:
        scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    elif target_decade == '2000s':
        scenarios = ['historical',]
    else:
        scenarios = ['ssp126', 'ssp245', 'ssp370', 'ssp585']
    # MCDC methods to use depend on variable
    if variable in ['E', 'H', 'eta']:
        degrees = ('int.-bias', 'linear', 'agnostic')
    else:
        degrees = ('linear', 'agnostic')
    # Column index
    column_tuples = []
    for degree in degrees:
        column_tuples.append(('Drift uncertainty', degree.capitalize()))
    column_tuples.append(('Other uncertainty', 'Model'))
    if target_decade != '2000s':
        column_tuples.append(('Other uncertainty', 'Scenario'))
    column_index = pd.MultiIndex.from_tuples(column_tuples)
    # Row index
    index_list = [esm.split('_')[0] for esm in esms]
    index_list += [SCENARIO_DICT[scenario] for scenario in scenarios]
    index_list += ['Min', 'Median', 'Max']
    index = pd.Index(index_list)
    # Initialize DataFrame
    detailed_df = pd.DataFrame(columns=column_index, index=index)
    # Loop over columns
    for column in column_index:
        # Drift uncertainty columns
        if column[0] == 'Drift uncertainty':
            degree = column[1].lower()
            for esm in esms:   # loop over ESMs
                uncertainty_s = np.zeros(len(scenarios))  # array to hold drift uncertainty for each scenario
                for s, scenario in enumerate(scenarios):  # calc drift uncertainty for each scenario
                    uncertainty_s[s] = calc_drift_uncertainty(esm=esm, variable=variable, degree=degree,
                                                              scenario=scenario, target_decade=target_decade,
                                                              sample_n=sample_n)
                uncertainty = uncertainty_s.mean()  # mean across scenarios
                detailed_df.loc[esm.split('_')[0], column] = uncertainty  # save drift uncertainty
        # Scenario uncertainty column
        elif column[1] == 'Scenario':
            for esm in esms:   # loop over ESMs; use agnostic method for scenario uncertainty
                uncertainty = calc_scenario_uncertainty(esm=esm, variable=variable, degree='agnostic',
                                                        target_decade=target_decade, sample_n=sample_n)
                detailed_df.loc[esm.split('_')[0], column] = uncertainty  # save scenario uncertainty
        # Model uncertainty column
        elif column[1] == 'Model':
            for scenario in scenarios:  # loop over scenarios; use agnostic method for model uncertainty
                uncertainty = calc_model_uncertainty(variable=variable, degree='agnostic', scenario=scenario,
                                                     target_decade=target_decade, sample_n=sample_n)
                detailed_df.loc[SCENARIO_DICT[scenario], column] = uncertainty  # save model uncertainty
    # Calculate min, median, and max of each column
    detailed_df.loc['Min'] = detailed_df.min(axis=0, skipna=True)
    detailed_df.loc['Median'] = detailed_df.median(axis=0, skipna=True)
    detailed_df.loc['Max'] = detailed_df.max(axis=0, skipna=True)
    # Round to specific number of decimal places
    if variable in ['Z', 'eps']:
        detailed_df = detailed_df.astype('float64').round(0)
    elif variable in ['E', 'H']:
        detailed_df = detailed_df.astype('float64').round(2)
    else:
        detailed_df = detailed_df.astype('float64').round(2)
    return detailed_df


def get_detailed_tex(variable='E', target_decade='2050s', sample_n=SAMPLE_N):
    """Detailed Latex table showing drift, model, and scenario uncertainty."""
    # Get DataFrame
    detailed_df = get_detailed_df(variable=variable, target_decade=target_decade, sample_n=sample_n)
    # Caption
    if variable in ['eta', 'eps']:
        variable_str = SYMBOLS_DICT[variable]
    else:
        variable_str = f'{SYMBOLS_DICT[variable]} ({target_decade}, relative to {REF_STR})'
    if target_decade == '2000s':
        caption = (
            f'Sources of uncertainty in {variable_str}. '
            'For each drift-correction method and model, '
            '\emph{drift uncertainty} corresponds to the 2nd--98th inter-percentile range. '
            '\emph{Model uncertainty} corresponds to the inter-model range: '
            '(i) for each model, calculate the mean of the agnostic-method drift-corrected data, '
            'then (ii) calculate the inter-model range. '
            'The final three rows contain summary statistics: the minimum, median, and maximum of each column.')
    else:
        caption = (
            f'Sources of uncertainty in {variable_str}. '
            'For each drift-correction method and model, '
            '\emph{drift uncertainty} corresponds to the 2nd--98th inter-percentile range: '
            '(i) for each projection scenario, '
            'calculate the 2nd--98th inter-percentile range of the drift-corrected data, '
            'then (ii) calculate the mean of this inter-percentile range by averaging across the projection scenarios. '
            'For each projection scenario, \emph{model uncertainty} corresponds to the inter-model range: '
            '(i) for each model, calculate the mean of the agnostic-method drift-corrected data, '
            'then (ii) calculate the inter-model range. '
            'For each model, \emph{scenario uncertainty} corresponds to the inter-scenario range: '
            '(i) for each projection scenario, calculate the mean of the agnostic-method drift-corrected data, '
            'then (ii) calculate the inter-scenario range. '
            'The final three rows contain summary statistics: the minimum, median, and maximum of each column.')
    # Column format, number of columns, and decimal places formatter
    if variable in ['Z', 'eps']:
        column_format = 'c|rr|r'
        n_cols = 4
        formatter = '{:.0f}'
    elif variable in ['E', 'H']:
        column_format = 'c|rrr|r'
        n_cols = 5
        formatter = '{:.2f}'
    else:
        column_format = 'c|rrr|r'
        n_cols = 5
        formatter = '{:.2f}'
    if target_decade != '2000s':
        column_format += 'r'
        n_cols += 1
    # Convert DataFrame to Latex
    tex_str = detailed_df.style.format(formatter=formatter, na_rep='').to_latex(
            environment='table*', position='t', position_float='centering',
            column_format=column_format, multicol_align='c|', hrules=True, caption=caption)
    # Manually reformat column titles etc
    if variable in ['eta', 'eps']:
        title_str = f'Sources of uncertainty in {SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})'
    else:
        title_str = f'Sources of uncertainty in {SYMBOLS_DICT[variable]} ({target_decade}; {UNITS_DICT[variable]})'
    tex_str = tex_str.replace('toprule\n',  # add title row, containing units
                              'toprule\n\multicolumn{%d}{c}{%s} \\\\ \n\midrule\n' % (n_cols, title_str))
    tex_str = tex_str.replace('\n & \multicolumn',  # add column title for first column
                              '\nModel or scenario & \multicolumn')
    tex_str = tex_str.replace('\multicolumn{2}{c|}{Other uncertainty}',  # remove line after "Other uncertainty"
                              '\multicolumn{2}{c}{Other uncertainty}')
    tex_str = tex_str.replace('\nMin',  # insert line before summary statistics
                              '\n\midrule\nMin')
    return tex_str


@cache
def get_summary_df(variables=('E', 'H', 'Z', 'eta', 'eps'), target_decade='2050s', sample_n=SAMPLE_N):
    """Summary DataFrame showing drift, model, and scenario uncertainty for multiple variables."""
    # Create empty summary DataFrame, with index that includes scenario uncertainty
    index = get_detailed_df(variable='eta', target_decade=None, sample_n=sample_n).columns
    summary_df = pd.DataFrame(index=index)
    # Loop over variables
    for variable in variables:
        # Get detailed DataFrame
        if variable in ['eta', 'eps']:
            detailed_df = get_detailed_df(variable=variable, target_decade=None, sample_n=sample_n)
        else:
            detailed_df = get_detailed_df(variable=variable, target_decade=target_decade, sample_n=sample_n)
        # List and Series of formatted median and range
        zipped_stats = zip(detailed_df.loc['Median'], detailed_df.loc['Min'], detailed_df.loc['Max'])
        if variable in ['Z', 'eps']:
            summary_list = [f'{a:.0f} ({b:.0f}–{c:.0f})' for a, b, c in zipped_stats]
        elif variable in ['E', 'H']:
            summary_list = [f'{a:.2f} ({b:.2f}–{c:.2f})' for a, b, c in zipped_stats]
        else:
            summary_list = [f'{a:.2f} ({b:.2f}–{c:.2f})' for a, b, c in zipped_stats]
        summary_ser = pd.Series(summary_list, index=detailed_df.columns)
        # Remove redundant range for model uncertainty if only historical scenario has been used
        if (variable not in ['eta', 'eps']) and (target_decade == '2000s'):
            summary_ser[('Other uncertainty', 'Model')] = summary_ser[('Other uncertainty', 'Model')].split(' ')[0]
        # Save mean and range to new column of summary DataFrame
        if variable in ['eta', 'eps']:
            summary_df[f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})'] = summary_ser
        else:
            summary_df[f'{SYMBOLS_DICT[variable]} ({target_decade}; {UNITS_DICT[variable]})'] = summary_ser
    return summary_df


def get_summary_tex(variables=('E', 'H', 'Z', 'eta', 'eps'), target_decade='2050s', sample_n=SAMPLE_N):
    """Summary Latex table Detailed DataFrame showing drift, model, and scenario uncertainty."""
    # Get DataFrame
    summary_df = get_summary_df(variables=variables, target_decade=target_decade, sample_n=sample_n)
    # Caption
    caption = (f'CMIP6 ensemble median and range (minimum–maximum) for different sources of uncertainty. '
               'For each drift-correction method, \emph{drift uncertainty} corresponds to '
               'the 2nd--98th inter-percentile range of the drift-corrected data. '
               '\emph{Model uncertainty} corresponds to from the inter-model range. '
               '\emph{Scenario uncertainty} corresponds to the inter-scenario range. '
               'The ensemble statistics shown here correspond to the summary statistics shown in Tables~S2--S6. '
               'For further details, see Tables~S2--S6.')
    # Convert DataFrame to Latex
    tex_str = summary_df.style.format(na_rep='').to_latex(
            environment='table*', position='t', position_float='centering',
            column_format=('cc'+'|c'*len(variables)), hrules=True, multirow_align='c', caption=caption)
    # Manually insert horizontal line between "Drift uncertainty" and "Other uncertainty"
    tex_str = tex_str.replace('\multirow[c]{2}{*}{Other uncertainty}',
                              '\midrule\n\multirow[c]{2}{*}{Other uncertainty}')
    return tex_str


def plot_uncorrected_timeseries(esm=DEF_ESM, variable='Ep', scenarios=('piControl', 'historical'),
                                title=None, legend=True, label_mean=True, ax=None):
    """Plot uncorrected time series for variable and scenario(s)."""
    # Create figure if ax is None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Loop over scenarios (in reverse)
    for scenario in scenarios[::-1]:
        # Get uncorrected time series for scenario and convert to DataArray
        uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[variable].to_xarray()
        # For SSPs, use data from 2015-2100
        if 'ssp' in scenario:  # for SSPs
            uncorr_da = uncorr_da.sel(Year=slice(2015, 2100))
        # If mean is to be included in label, calculate mean
        if label_mean:
            m = uncorr_da.mean()  # mean
            sem = stats.sem(uncorr_da)  # standard error (not accounting for autocorrelation here)
            label = f'{SCENARIO_DICT[scenario]} (uncorrected; {m:.3f} $\pm$ {sem:.3f} {UNITS_DICT[variable]})'
        else:
            label = f'{SCENARIO_DICT[scenario]} (uncorrected)'
        # Limit control data before plotting (so that y-axis range is appropriate)?
        if scenario == 'piControl' and 'historical' in scenarios:
            uncorr_da = uncorr_da.sel(Year=slice(1850, 2014))
        # Plot time series
        ax.plot(uncorr_da.Year, uncorr_da, label=label, color=SCENARIO_C_DICT[scenario], alpha=1.0, linewidth=1.0)
    # Axis ticks
    ax.set_xticks(np.arange(1850, 2101, 50))
    ax.minorticks_on()
    ax.tick_params(axis='both', left=True, top=False, right=True, bottom=True)
    # Limit x-axis and time series to 1850-2100 or 1850-2014?
    if bool({'ssp126', 'ssp245', 'ssp370', 'ssp585'} & set(scenarios)):
        ax.set_xlim([1850, 2100])
    else:
        ax.set_xlim([1850, 2014])
    # Labels, legend etc
    ax.set_xlabel(r'Year')
    ax.set_ylabel(f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})')
    if title:
        ax.set_title(title)
    if legend:
        ax.legend(fontsize='small')
    return ax


def legend_min_alpha_linewidth(leg):
    """Set min alpha and min linewidth in legend."""
    for lh in leg.legendHandles:
        try:  # set min alpha in legend
            if lh.get_alpha() < 0.5:
                lh.set_alpha(0.5)
        except TypeError:
            pass
        try: # set min linewidth in legend
            if lh.get_linewidth() < 0.5:
                lh.set_linewidth(0.5)
        except TypeError:
            pass
    return leg


def plot_control_with_drift(esm=DEF_ESM, variable='E', degree='agnostic', sample_n=SAMPLE_N, title=None, legend=True,
                            ax=None):
    """Plot uncorrected control time series with drift samples."""
    # Create figure if ax=None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Get uncorrected control time series
    pi_da = get_cmip6_df(esm=esm, scenario='piControl').set_index('Year')[variable].to_xarray()
    # Plot piControl time series
    ax.plot(pi_da.Year, pi_da, label='Control time series (uncorrected)', color='black', alpha=0.5, linewidth=0.7)
    # Get drift samples
    drift_da = sample_drift(esm=esm, variable=variable, degree=degree, sample_n=sample_n, plot=False)
    # Plot drift samples
    for i in range(sample_n):
        if i == 0:
            label = f'Drift samples ({degree} method; n={sample_n})'
        else:
            label = None
        ax.plot(drift_da.Year, drift_da.isel(Draw=i), color='purple', alpha=10./sample_n, label=label)
    # x-axis ticks and range
    ax.set_xticks(np.arange(0, pi_da.Year[-1], 200))
    ax.minorticks_on()
    ax.tick_params(axis='both', left=True, top=False, right=True, bottom=True)
    ax.set_xlim(pi_da.Year[0], pi_da.Year[-1])
    # Labels, legend etc
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})')
    if title:
        ax.set_title(title)
    if legend:
        leg = ax.legend(fontsize='small')
        leg = legend_min_alpha_linewidth(leg)
    return ax


def plot_corrected_timeseries(esm=DEF_ESM, variable='E', degree='agnostic', scenarios=('piControl', 'historical'),
                              sample_n=SAMPLE_N, plot_uncorrected=False, title=None, legend=True, ax=None):
    """Plot drift-corrected time series for variable and scenario(s)."""
    # Create figure if ax=None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 3))
    # Loop over scenarios (in reverse)
    for scenario in scenarios[::-1]:
        # Also plot original uncorrected time series?
        if plot_uncorrected:
            # Get uncorrected time series for scenario and convert to DataArray
            uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[variable].to_xarray()
            # For SSPs, show from 2015
            if 'ssp' in scenario:  # for SSPs
                uncorr_da = uncorr_da.sel(Year=slice(2015, 2100))
            # Plot time series
            label = f'{SCENARIO_DICT[scenario]} (uncorrected)'
            ax.plot(uncorr_da.Year, uncorr_da, label=label, color='k', linestyle='--', alpha=1.0, linewidth=1.0)
        # Get drift corrected time series samples
        corr_da = sample_corrected(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                   sample_n=sample_n, plot=False)
        # For SSPs, show from 2015
        if 'ssp' in scenario:  # for SSPs
            corr_da = corr_da.sel(Year=slice(2015, 2100))
        # Plot corrected samples
        for i in range(sample_n):
            if i == 0:
                label = f'{SCENARIO_DICT[scenario]} ({degree}; n={sample_n})'
            else:
                label = None
            ax.plot(corr_da.Year, corr_da.isel(Draw=i), color=SCENARIO_C_DICT[scenario], alpha=10./sample_n, label=label)
    # x-axis ticks and range
    if 'piControl' in scenarios:
        ax.set_xticks(np.arange(0, 3000, 200))
    else:
        ax.set_xticks(np.arange(1850, 2100, 50))
    ax.minorticks_on()
    ax.tick_params(axis='both', left=True, top=False, right=True, bottom=True)
    if bool({'ssp126', 'ssp245', 'ssp370', 'ssp585'} & set(scenarios)):
        ax.set_xlim([1850, 2100])
    elif 'historical' in scenarios:
        ax.set_xlim([1850, 2014])
    else:
        ax.set_xlim([corr_da.Year[0], corr_da.Year[-1]])
    # Labels, legend etc
    ax.set_xlabel('Year')
    ax.set_ylabel(f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})')
    if title:
        ax.set_title(title)
    if legend:
        leg = ax.legend(fontsize='small')
        leg = legend_min_alpha_linewidth(leg)
    return ax


def scatter_line_rel(esm=DEF_ESM, x_var='E', y_var='H', scenarios=True,
                     plot_uncorrected=False, degree='agnostic', sample_n=SAMPLE_N, plot_largest_intercept=False,
                     title=None, legend=True, ax=None):
    """Scatter (uncorrected) and/or line (corrected) plot of y_var vs x_var."""
    # If scenarios is True, update scenarios to include Tier 1 SSPs
    if scenarios is True:
        scenarios = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
    # Create figure if ax=None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # Show zero
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    # Loop over scenarios (in reverse)
    for scenario in scenarios[::-1]:
        # Plot uncorrected data?
        if plot_uncorrected:
            # Get uncorrected data
            x_uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[x_var].to_xarray()
            y_uncorr_da = get_cmip6_df(esm=esm, scenario=scenario).set_index('Year')[y_var].to_xarray()
            # For SSPs, show from 2015
            if 'ssp' in scenario:
                x_uncorr_da = x_uncorr_da.sel(Year=slice(2015, 2100))
                y_uncorr_da = y_uncorr_da.sel(Year=slice(2015, 2100))
            # Plot uncorrected data
            ax.plot(x_uncorr_da, y_uncorr_da, label=f'{SCENARIO_DICT[scenario]} (uncorrected)',
                    color=SCENARIO_C_DICT[scenario], linestyle='-', linewidth=0.2, marker='.', alpha=0.5)
        # Plot drift-corrected samples?
        if degree and sample_n:
            # Get drift-corrected data for x_var
            x_corr_da = sample_corrected(esm=esm, variable=x_var, degree=degree, scenario=scenario,
                                         sample_n=sample_n, plot=False)
            y_corr_da = sample_corrected(esm=esm, variable=y_var, degree=degree, scenario=scenario,
                                         sample_n=sample_n, plot=False)
            # For SSPs, show from 2015
            if 'ssp' in scenario:
                x_corr_da = x_corr_da.sel(Year=slice(2015, 2100))
                y_corr_da = y_corr_da.sel(Year=slice(2015, 2100))
            # Intialize max and min intercept encountered as zero
            max_intercept = 0
            min_intercept = 0
            # Loop over drift-corrected samples
            for i in range(sample_n):
                # Label for plotting
                if i == 0:  # label only once
                    label = f'{SCENARIO_DICT[scenario]} ({degree}; n={sample_n})'
                else:
                    label = None
                # Plot drift-corrected data
                ax.plot(x_corr_da.isel(Draw=i), y_corr_da.isel(Draw=i), label=label,
                        color=SCENARIO_C_DICT[scenario], alpha=0.1, linewidth=0.05)
                # Does this sample have the max or min intercept encountered so far?
                sign_change_idxs = np.where(x_corr_da.isel(Draw=i).data[:-1] * x_corr_da.isel(Draw=i).data[1:] < 0)[0]
                try:
                    final_intercept = y_corr_da.isel(Draw=i).data[sign_change_idxs[-1]]
                except IndexError:
                    final_intercept = 0
                if final_intercept > max_intercept:
                    max_intercept = final_intercept
                    max_intercept_i = i
                elif final_intercept < min_intercept:
                    min_intercept = final_intercept
                    min_intercept_i = i
            # Re-plot sample with max/min intercept?
            if plot_largest_intercept:
                ax.plot(x_corr_da.isel(Draw=max_intercept_i), y_corr_da.isel(Draw=max_intercept_i),
                        label='Max intercept',
                        color='magenta', alpha=0.5, linewidth=1, linestyle='--')
                ax.plot(x_corr_da.isel(Draw=min_intercept_i), y_corr_da.isel(Draw=min_intercept_i),
                        label='Min intercept',
                        color='cyan', alpha=0.5, linewidth=1, linestyle='--')
    # Axis ticks
    ax.minorticks_on()
    # x & y axis labels
    ax.set_xlabel(f'{SYMBOLS_DICT[x_var]} ({UNITS_DICT[x_var]})')
    ax.set_ylabel(f'{SYMBOLS_DICT[y_var]} ({UNITS_DICT[y_var]})')
    # Legend, title etc
    if title:
        ax.set_title(title)
    if legend:
        leg = ax.legend(fontsize='small')
        leg = legend_min_alpha_linewidth(leg)
    return ax


def histogram_of_variable(esm=DEF_ESM, variable='Z', degree='agnostic', scenarios=True,
                          target_decade='2050s',  # target_decade not relevant if variable is eta or eps
                          sample_n=SAMPLE_N, title=None, legend=True, ax=None):
    """Plot histogram of (i) E/H/Z for a target decade or (ii) eta/eps coefficient."""
    # Create figure if ax=None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # If scenarios is True, update scenarios to include historical and/or Tier 1 SSPs
    if scenarios is True:
        if variable in ['eta', 'eps']:
            scenarios = ('historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')
        elif int(target_decade[0:4]) <= 2000:
            scenarios = ('historical',)
        else:
            scenarios = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
    # Loop over scenarios (in reverse)
    for scenario in scenarios[::-1]:
        # Get data for plotting
        if variable in ['eta', 'eps']:
            data_da = sample_eta_eps(esm=esm, eta_or_eps=variable, degree=degree, scenario=scenario,
                                     sample_n=sample_n, plot=False)
        else:
            data_da = sample_target_decade(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                           target_decade=target_decade, sample_n=sample_n, plot=False)
        # Legend label (including mean and standard deviation) and bin width depend on variable
        if variable == 'eta':
            bin_width = 0.001
            mean_std_str = f'{data_da.mean():0.3f} $\pm$ {data_da.std():0.3f}'
        elif variable in ['E', 'H']:
            bin_width = 0.01
            mean_std_str = f'{data_da.mean():0.2f} $\pm$ {data_da.std():0.2f} ({UNITS_DICT[variable]})'
        else:
            bin_width = 0.1
            mean_std_str = f'{data_da.mean():0.1f} $\pm$ {data_da.std():0.1f} ({UNITS_DICT[variable]})'
        label = f'{SCENARIO_DICT[scenario]} ({mean_std_str})'
        # Plot histogram
        bins = np.arange(data_da.min()-bin_width/2, data_da.max()+bin_width, bin_width)
        c = SCENARIO_C_DICT[scenario]
        ax.hist(data_da, bins=bins, density=False, histtype='step', color=c, label=label)  # outer edges
        ax.hist(data_da, bins=bins, density=False, color=c, alpha=0.1)  # inner shading (transparent)
    # Axis ticks and labels
    ax.minorticks_on()
    ax.set_xlabel(f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})')
    ax.set_ylabel(f'Count')
    # Title & legend
    if title:
        ax.set_title(title)
    if legend:
        ax.set_ylim([0, ax.get_ylim()[1]*1.6])  # extend ylim so more room for legend
        ax.legend(fontsize='small', loc='upper left')
    return ax


def boxplot_of_variable(esm=DEF_ESM, variable='E',
                        target_decade='2050s',  # target_decade not relevant if variable is eta or eps
                        degrees=True, scenarios=True, sample_n=SAMPLE_N, title=None, ax=None):
    """Plot box (25-75) and whisker (2-98) plots of (i) E/H/Z for a target decade or (ii) eta/eps coefficient."""
    # Create figure if ax=None
    if not ax:
        fig, ax = plt.subplots(1, 1, figsize=(4.5, 4.5))
    # If degrees is True, update degrees to include int.-bias (if relevant), linear, and agnostic method
    if degrees is True:
        if variable in ['E', 'H', 'eta']:
            degrees = ('int.-bias', 'linear', 'agnostic')
        else:
            degrees = ('linear', 'agnostic')
    # If scenarios is True, update scenarios to include historical and/or Tier 1 SSPs
    if scenarios is True:
        if variable in ['eta', 'eps']:
            scenarios = ('historical', 'ssp126', 'ssp245', 'ssp370', 'ssp585')
        elif int(target_decade[0:4]) <= 2000:
            scenarios = ('historical',)
        else:
            scenarios = ('ssp126', 'ssp245', 'ssp370', 'ssp585')
    # List to hold DataArrays, tick labels
    data_da_list = []
    tick_label_list = []
    # Loop over methods (degree) and scenarios and get data
    for i, degree in enumerate(degrees):
        for scenario in scenarios:
            if variable in ['eta', 'eps']:
                data_da = sample_eta_eps(esm=esm, eta_or_eps=variable, degree=degree, scenario=scenario,
                                         sample_n=sample_n, plot=False)
            else:
                data_da = sample_target_decade(esm=esm, variable=variable, degree=degree, scenario=scenario,
                                               target_decade=target_decade, sample_n=sample_n, plot=False)
            data_da_list.append(data_da)
            tick_label_list.append(SCENARIO_DICT[scenario])
    # For eta, plot eta = 1 line
    if variable == 'eta':
        ax.axhline(1., color='0.9')
    # Plot data: median, 25-75, 2-98, and outliers
    ax.boxplot(data_da_list, whis=[2, 98], sym='.', flierprops={'markersize': 1})
    # Annotate segments of figure
    for i, degree in enumerate(degrees):
        xlim = ax.get_xlim()
        seg_bnds = (np.array([i, i+1.]) / len(degrees)) * (xlim[1] - xlim[0]) + xlim[0]  # segment bounds
        if i > 0:
            ax.axvline(seg_bnds[0], color='0.8')
        ax.text(seg_bnds.mean(), ax.get_ylim()[1], f'{degree.capitalize()}', ha='center', va='bottom',
                fontsize='large')
    # Configure axes etc
    ylim = list(ax.get_ylim())
    ylim[1] += 0.1 * (ylim[1] - ylim[0])  # stretch y-lim to accommodate annotation above
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', left=True, top=False, right=True, bottom=True)
    ax.set_xticklabels(tick_label_list, rotation=90)
    if variable in ['eta', 'eps']:
        ax.set_ylabel(f'{SYMBOLS_DICT[variable]} ({UNITS_DICT[variable]})')
    else:
        ax.set_ylabel(f'{SYMBOLS_DICT[variable]} ({target_decade}; {UNITS_DICT[variable]})')
    if title:
        ax.set_title(title)
    return ax


def composite_problem_of_drift(esm=DEF_ESM):
    """Demonstrate problem of drift by showing uncorrected time series and relationships."""
    # Configure plot and subplots
    fig = plt.figure(figsize=(15, 10.5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.03, hspace=0, wspace=0)
    spec = fig.add_gridspec(6, 3)
    axs = []  # list of axes
    for r in range(2):  # 1st column (a, b)
        ax = fig.add_subplot(spec[2*r:2*r+2, 0])
        axs.append(ax)
    for r in range(3):  # 2nd column (c, d, e)
        ax = fig.add_subplot(spec[2*r:2*r+2, 1])
        axs.append(ax)
    for r in range(2):  # 3rd column (f, g)
        ax = fig.add_subplot(spec[3*r:3*r+3, 2])
        axs.append(ax)
    # 1st & 2nd columns
    for i, variable in enumerate(['Ep', 'Hp', 'E', 'H', 'Z']):
        ax = axs[i]
        if i < 2:
            label_mean = True
        else:
            label_mean = False
        plot_uncorrected_timeseries(esm=esm, variable=variable, scenarios=('piControl', 'historical'),
                                    title=f'({chr(97+i)}) {SYMBOLS_DICT[variable]} time series',
                                    legend=True, label_mean=label_mean, ax=ax)
    # 3rd column
    for i, x_var, y_var in zip(range(5, 7), ['E', 'H'], ['H', 'Z']):
        ax = axs[i]
        scatter_line_rel(esm=esm, x_var=x_var, y_var=y_var, scenarios=('historical',),
                         plot_uncorrected=True, degree=None, sample_n=None, plot_largest_intercept=False,
                         title=f'({chr(97+i)}) {SYMBOLS_DICT[y_var]} vs {SYMBOLS_DICT[x_var]}',
                         legend=True, ax=ax)
    # Main title
    fig.suptitle((f'Uncorrected time series and {"–".join([SYMBOLS_DICT[variable] for variable in ("E", "H", "Z")])} '
                  f'relationships for the {esm.split("_")[0]} control & historical simulations, '
                  f'demonstrating the problem of drift\n'),
                 fontsize='x-large')
    return fig


def composite_compare_methods_timeseries(esm=DEF_ESM, variable='E', degrees=True, sample_n=SAMPLE_N):
    """Compare int.-bias, linear, and agnostic methods (ie degrees) of MCDC by plotting time series."""
    # If degrees is True, update degrees to include int.-bias (if integrated flux), linear, and agnostic
    if degrees is True:
        if variable in ['E', 'H']:
            degrees = ('int.-bias', 'linear', 'agnostic')
        else:
            degrees = ('linear', 'agnostic')
    # Configure plot and subplots
    fig, axs = plt.subplots(len(degrees), 3, figsize=(18, 4*len(degrees)), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.1, hspace=0, wspace=0)
    # Loop over methods, which correspond to rows
    for j, degree in enumerate(degrees):
        # 1st column: drift samples
        title = f'({chr(97+j*4)}) {degree.capitalize()}-method drift samples'
        plot_control_with_drift(esm=esm, variable=variable, degree=degree,
                                sample_n=sample_n, title=title, legend=True, ax=axs[j, 0])
        # 2nd column: corrected control time series
        title = f'({chr(97+j*4+1)}) {degree.capitalize()}-corrected control'
        plot_corrected_timeseries(esm=esm, variable=variable, degree=degree, scenarios=('piControl',),
                                  sample_n=sample_n, plot_uncorrected=True, title=title, legend=True, ax=axs[j, 1])
        # 3rd column: corrected historical time series
        title = f'({chr(97+j*4+2)}) {degree.capitalize()}-corrected historical'
        plot_corrected_timeseries(esm=esm, variable=variable, degree=degree, scenarios=('historical',),
                                  sample_n=sample_n, plot_uncorrected=True, title=title, legend=True, ax=axs[j, 2])
    # Share y limits within column
    for i in range(3):  # columns
        for j in range(1, len(degrees)):  # rows (2nd onwards)
            axs[j, i].sharey(axs[j, i])
    # Main title
    fig.suptitle((f'Different methods of MCDC applied to {SYMBOLS_DICT[variable]}, '
                  f'using the {esm.split("_")[0]} control & historical simulations\n'),
                  fontsize='xx-large')
    return fig


def composite_boxplots(esm=DEF_ESM, variables=('E', 'H', 'Z'), target_decade='2000s',
                       degrees=True, scenarios=True, sample_n=SAMPLE_N):
    """Fig showing boxplots of drift-corrected results for multiple variables from one ESM."""
    # Configure plot and subplots
    fig, axs = plt.subplots(1, len(variables), figsize=(5*len(variables), 5), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.2, h_pad=0.1, hspace=0, wspace=0)
    # Loop over variables
    for i, variable in enumerate(variables):
        # Plot subplot
        if variable in ['eta', 'eps']:
            title = f'({chr(97+i)}) {SYMBOLS_DICT[variable]}'
        else:
            title = f'({chr(97+i)}) {SYMBOLS_DICT[variable]} ({target_decade})'
        _ = boxplot_of_variable(esm=esm, variable=variable, target_decade=target_decade,
                                degrees=degrees, scenarios=scenarios, sample_n=sample_n, title=title, ax=axs[i])
    # Main title depends on number of scenarios shown
    scenarios_shown = set([s.get_text() for s in axs[0].get_xticklabels()])  # set of unique scenarios in 1st subplot
    if len(scenarios_shown) == 1:
        scen = scenarios_shown.pop()
        if scen == 'Historical':
            scen = 'historical'
        suptitle = f'Drift-corrected results with drift uncertainty for the {esm.split("_")[0]} {scen} simulation'
        [ax.set_xticklabels([]) for ax in axs]  # also hide x-axis tick labels if only one scenario
    elif scenarios_shown == {'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'}:
        suptitle = f'Drift-corrected results with drift uncertainty for the {esm.split("_")[0]} projection simulations'
    else:
        suptitle = f'Drift-corrected results with drift uncertainty for the {esm.split("_")[0]} simulations'
    fig.suptitle(suptitle, fontsize='x-large')
    return fig


def ensemble_boxplots(esms=True, variable='E', target_decade='2000s', degrees=True, scenarios=True, sample_n=SAMPLE_N):
    """Fig showing boxplots of drift-corrected results for the CMIP6 ensemble."""
    # If esms is True, use all available ESMs
    if esms is True:
        esms = get_cmip6_df(esm=True, scenario=True)['ESM'].unique()
    # Configure subplots
    ncols = 4
    nrows = math.ceil(len(esms) / ncols)
    fig, axs = plt.subplots(nrows, ncols, figsize=(4.5*ncols, 4.5*nrows), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.15, h_pad=0.15, hspace=0, wspace=0)
    # Loop over ESMs
    for i, esm in enumerate(esms):
        # Plot subplot for this ESM
        title = f'({chr(97+i)}) {esm.split("_")[0]}'
        _ = boxplot_of_variable(esm=esm, variable=variable, target_decade=target_decade,
                                degrees=degrees, scenarios=scenarios, sample_n=sample_n, title=title,
                                ax=axs.flatten()[i])
    # Main title depends on variable and number of scenarios shown
    if variable in ['eta', 'eps']:
        suptitle = f'Drift-corrected {SYMBOLS_DICT[variable]} with drift uncertainty'
    else:
        suptitle = f'Drift-corrected {SYMBOLS_DICT[variable]} ({target_decade}) with drift uncertainty'
    scenarios_shown = set([s.get_text() for s in axs.flatten()[0].get_xticklabels()])  # unique scenarios in 1st subplot
    if len(scenarios_shown) == 1:
        scen = scenarios_shown.pop()
        if scen == 'Historical':
            scen = 'historical'
        suptitle = f'{suptitle} for the CMIP6 {scen} simulations'
        [ax.set_xticklabels([]) for ax in axs.flatten()]  # also hide x-axis tick labels if only one scenario
    elif scenarios_shown == {'SSP1-2.6', 'SSP2-4.5', 'SSP3-7.0', 'SSP5-8.5'}:
        suptitle = f'{suptitle} for the CMIP6 projection simulations'
    else:
        suptitle = f'{suptitle} for the CMIP6 simulations'
    fig.suptitle(suptitle, fontsize='xx-large')
    return fig


def composite_rel_eta_eps_demo(esm=DEF_ESM, degree='agnostic', sample_n=SAMPLE_N):
    """Fig showing E-H-Z relationships and eta & eps boxplots for one ESM."""
    # Configure subplots
    fig, axs = plt.subplots(2, 3, figsize=(14, 10), constrained_layout=True)
    fig.set_constrained_layout_pads(w_pad=0.1, h_pad=0.2, hspace=0, wspace=0)
    # Loop over rows
    for j, eta_or_eps, x_var, y_var in [(0, 'eta', 'E', 'H'), (1, 'eps', 'H', 'Z')]:
        # Cols 1 & 2: plot relationships for (i) historical and (ii) projections, for specified degree/method of MCDC
        for i, scenarios in enumerate([('historical',), True]):
            title = f'({chr(97+3*j+i)}) {SYMBOLS_DICT[y_var]} vs {SYMBOLS_DICT[x_var]}'
            if scenarios == ('historical',):
                title = f'{title} (historical; {degree})'
                plot_largest_intercept = True
            else:
                title = f'{title} (projections; {degree})'
                plot_largest_intercept = False
            _ = scatter_line_rel(esm=esm, x_var=x_var, y_var=y_var, scenarios=scenarios,
                                 plot_uncorrected=False, degree=degree, sample_n=sample_n,
                                 plot_largest_intercept=plot_largest_intercept, title=title, legend=True, ax=axs[j, i])
        # Col 3: boxplots of eta & eps for all degrees/methods of MCDC
        title = f'({chr(97+3*j+2)}) {SYMBOLS_DICT[eta_or_eps]} estimates'
        _ = boxplot_of_variable(esm=esm, variable=eta_or_eps, target_decade=None, degrees=True,
                                scenarios=scenarios, sample_n=sample_n, title=title, ax=axs[j, 2])
    # Main title
    suptitle = (f'Drift-corrected {"–".join([SYMBOLS_DICT[var] for var in ("E", "H", "Z")])} relationships '
                f'and {" & ".join([SYMBOLS_DICT[var] for var in ("eta", "eps")])} estimates for {esm.split("_")[0]}')
    fig.suptitle(suptitle, fontsize='xx-large')
    return fig


def name_save_fig(fig,
                  fso='f',  # figure type, either 'f' (main), 's' (supp), or 'o' (other)
                  exts=('pdf', 'png'),  # extension(s) to use
                  close=False):
    """Name & save a figure, and increase counter."""
    # Name based on counter, then update counter (in preparation for next figure)
    if fso == 'f':
        fig_name = f'fig{next(F_NUM):02}'
    elif fso == 's':
        fig_name = f's{next(S_NUM):02}'
    else:
        fig_name = f'o{next(O_NUM):02}'
    # File location based on extension(s)
    for ext in exts:
        # Get constrained layout pads (to preserve values after saving fig)
        w_pad, h_pad, _, _ = fig.get_constrained_layout_pads()
        # Sub-directory
        sub_dir = FIG_DIR.joinpath(f'{fso}_{ext}')
        sub_dir.mkdir(exist_ok=True)
        # Save
        fig_path = sub_dir.joinpath(f'{fig_name}.{ext}')
        fig.savefig(fig_path)
        # Print file name and size
        fig_size = fig_path.stat().st_size / 1024 / 1024  # bytes -> MB
        print(f'Written {fig_name}.{ext} ({fig_size:.2f} MB)')
        # Reset constrained layout pads to previous values
        fig.set_constrained_layout_pads(w_pad=w_pad, h_pad=h_pad)
        # Suppress output in notebook?
    if close:
        plt.close()
    return fig_name


def name_save_table(tex_str, fso='f'):
    """Name & save a Latex table, and increase counter."""
    # Name based on counter, then update counter (in preparation for next figure)
    if fso == 'f':
        table_name = f'table_{next(T_NUM):02}.tex'
    elif fso == 's':
        table_name = f'table_S{next(ST_NUM):02}.tex'
    else:
        table_name = f'table_O{next(OT_NUM):02}.tex'
    # Save
    table_path = TABLE_DIR.joinpath(table_name)
    with open(table_path, 'w') as f:
        f.write(tex_str)
    print(f'Written {table_name}')
    return table_name
