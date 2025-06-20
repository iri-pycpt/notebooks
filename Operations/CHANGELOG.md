# Release notes for PyCPT conda environment

## v2.10.2
- Fixed typo in Mac environment file again

## v2.10.1
- Fixed typo in Mac environment file

## v2.10.0
### New functionality
- PyCPT no longer requires model outputs to be available for all models for all years of the training period. If a model is missing for a given year, its climatology will be used.
- The notebook now displays a table of which model is available for which years.
### Data catalog changes
- `SEAS51c` corrects errors in `SEAS51b`
- Added `METEOFRANCE9`
- Removed non-working dataset `AER04`
### Other
- Now managing development environment with pixi, and building packages with pixi-build.

## v2.9.2
- Fix generate-forecasts-from-config, which was broken in 2.9.0.
- Make `root_mean_squared_error` available for MME as well as for individual models

## v2.9.1
- Bug fix: write forecast plots for different models to different files

## v2.9.0
### New functionality
- Add support for weather station data.
- New skill score `root_mean_squared_error`
- New configuration options `skillmask_threshold` and `drymask_threshold`. Setting them to None (the default) results in no mask being used.
- PCR (principal components regression), which was removed after it was found to be broken, has been fixed and reintroduced.
- New download args `target_first_year` and `target_final_year` allow hindcast period to be specified by referring to the years of the target season, instead of the years of the forecast initialization. This representation is more convenient when generating forecasts at multiple lead times for the same target season, particularly when some initializations are in the prior calendar year, e.g. Dec, Jan, and Feb initialization for MAM season.
### Data catalog changes
- Fix `GEFSv12.PRCP` hindcast URL, which was missing longitude constraint.
- Fix `CCSM4.TMIN` URL, which pointed to TMAX rather than TMIN.
- New datasets added to the catalog:
  - `SPEARb.SST` (corrects errors in `SPEAR.SST`)
  - `SEAS51b` (corrects errors in `SEAS51`)
  - `CanSIPSIC4.T2M`
### Other
- Dependencies have been updated. Notably, jupyter notebook was updated to version 7, which entails changes to the user interface, e.g. some menu items have moved around.
- The function `plot_mme_skill` now supports the same skill core names as `plot_skill`. Previously, they used different names for the same score, e.g. `2afc` vs. `two_alternative_forced_choice`. The old names are still accepted in order not to break existing notebooks.
- `transform_predictand` option has been marked as non-working. It has been found not to work properly in previous versions, and is not fixed in this release.
- The conda packages `cpt-dl`, `cpt-extras`, and `cpt-core` have been merged into the `pycpt` package, and are no longer packaged separately. The python package names (the ones that python code imports) have not changed. `cpt-bin` and `cpt-io` are still packaged separately.
- Fix generation of skill matrix png file in subseasonal version.
- Resolve spurious warnings that appeared in the notebook.

## v2.8.2
- cpt-dl dataset additions: OISST2p1, NNRP.UA, NNRP.VA, CESM1.PRCP

## v2.8.1
- Renamed generate-forecasts to generate-forecasts-from-config, resolving name collision with a script on Data Library servers.
- Added optional download arguments `target_first_year`, `target_final_year`. Whereas `first_year`/`final_year` refer to the year of the forecast initialization, `target_first_year`/`target_final_year` refer to the year of the midpoint of the target season. `first_year` and `final_year` are deprecated and may be removed in a future version.
- `cptio.enacts` package for loading and aggregating ENACTS data directly from files, without going through Ingrid.
- TSV files generated by `cptio` now format numbers to six significant digits, as Ingrid does, rather than six digits after the decimal point.

## v2.7.3
- Fixed failure in generate-forecasts when temp directory and output directory are on different filesystems

## v2.7.2
- Added generate-forecasts and upload-forecasts scripts. Jupyter Notebook is now only for experimentation, not for operational use.
- Removed pycpt-operational.py, replaced by generate-forecasts script.

## v2.6.2
- No backwards-incompatible changes. Notebooks designed for 2.5.1 and later can be used without modification.
- Added pycpt-operational.py

## v2.6.1
- No backwards-incompatible changes. Notebooks designed for 2.5.1 and later can be used without modification.
- `convert_np64_datetime` now works on Windows, even for dates prior to the epoch.
- If `lead_low` and `lead_high` are not specified, they will now be calculated automatically from `fdate` and `target`. If they are specified, they will be compared against the leads entailed by `fdate` and `target`, and an exception will be raised if the values are not consistent.
- Generated forecast file names now contain the forecast year, e.g. `MME_deterministic_forecast_2023.nc`.
- Local predictand file can now be netcdf or tsv. (Instructions for constructing a compatible netcdf file are still pending.)
- Bug fixed in UCSB.TMEAN data URL.
