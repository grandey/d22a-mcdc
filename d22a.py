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
    - For SSP scenarios, data are referenced to the 2000-2010 mean. Otherwise, data are referenced to first 11 years.
    - Requires data produced by data_dTBD_epsilon.ipynb.
    """
    # Create DataFrame to hold data
    col_list = ['ESM', 'Scenario', 'Year', 'Ep', 'E', 'Hp', 'H', 'Z']
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
    return cmip6_df
