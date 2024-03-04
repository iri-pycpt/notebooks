# Release notes for PyCPT conda environment

## PyCPT 2.7.2
- Added generate-forecasts and upload-forecasts scripts. Jupyter Notebook is now only for experimentation, not for operational use.
- Removed pycpt-operational.py, replaced by generate-forecasts script.

## PyCPT 2.6.2
- No backwards-incompatible changes. Notebooks designed for 2.5.1 and later can be used without modification.
- Added pycpt-operational.py

## PyCPT 2.6.1
- No backwards-incompatible changes. Notebooks designed for 2.5.1 and later can be used without modification.
- `convert_np64_datetime` now works on Windows, even for dates prior to the epoch.
- If `lead_low` and `lead_high` are not specified, they will now be calculated automatically from `fdate` and `target`. If they are specified, they will be compared against the leads entailed by `fdate` and `target`, and an exception will be raised if the values are not consistent.
- Generated forecast file names now contain the forecast year, e.g. `MME_deterministic_forecast_2023.nc`.
- Local predictand file can now be netcdf or tsv. (Instructions for constructing a compatible netcdf file are still pending.)
- Bug fixed in UCSB.TMEAN data URL.
