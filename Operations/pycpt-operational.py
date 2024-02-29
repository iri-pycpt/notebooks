#!/usr/bin/env python
# coding: utf-8

import pycpt
import packaging
min_version = '2.6.2'
assert packaging.version.parse(pycpt.__version__) >= packaging.version.parse(min_version), f'This notebook requires version {min_version} or higher of the pycpt library, but you have version {pycpt.__version__}. Please close the notebook, update your environment, and load the notebook again. See https://iri-pycpt.github.io/installation/'

import cptdl as dl
import datetime as dt
from pathlib import Path
import pycpt.automation


MOS = 'CCA'
predictand_name = 'UCSB.PRCP'
local_predictand_file = None
download_args = { 
    'first_year': 1992,
    'final_year': 2020,
    'target': 'Mar-May',
    'predictor_extent': {
        'west': 10, 
        'east': 90,
        'south': -20, 
        'north': 20,
      }, 
    'predictand_extent': {
        'west': 41.6,  
        'east': 43.6,
        'south': 10.9, 
        'north': 12.8,
      },
    'filetype': 'cptv10.tsv'
}
cpt_args = { 
    'transform_predictand': None,
    'tailoring': None,
    'cca_modes': (1,6),
    'x_eof_modes': (1,8),
    'y_eof_modes': (1,6),
    'validation': 'crossvalidation',
    'drymask': False,
    'scree': True,
    'crossvalidation_window': 3,
    'synchronous_predictors': True,
}

issue_months = [12, 1, 2]

ensemble = [ "SPEAR.PRCP", 'CanSIPSIC3.PRCP','CCSM4.PRCP','GEOSS2S.PRCP','CFSv2.PRCP']

dest_dir = Path.home() / "Desktop/PyCPT-DL" / "DJIBOUTI_MAM_2024"


# starts for which we need to generate the forecast by hand
skip_issue_dates = [
    # CanSIPS-IC3 hindcasts end 1 Nov 2021, forecasts start 1 Oct 2021
    dt.datetime(2020, 12, 1),
    dt.datetime(2021, 1, 1),
    dt.datetime(2021, 2, 1),
]

if __name__ == '__main__':
    Path(dest_dir).mkdir(parents=True, exist_ok=True)

    pycpt.automation.update_all(
        dest_dir,
        issue_months,
        skip_issue_dates,
        MOS,
        ensemble,
        predictand_name,
        local_predictand_file,
        download_args,
        cpt_args,
    )
