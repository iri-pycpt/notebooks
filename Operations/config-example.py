# This is a configuration file for the PyCPT scripts
# generate-forecasts and upload-forecasts.

import datetime as dt
from pathlib import Path


# The path where forecasts for all seasons (including this one) will
# be stored. This should NOT be the same as the case_dir used in the
# Jupyter Notebook. That one is for experimentation, this one is for
# operational forecasts generated after the forecast configuration has
# been finalized.
operational_forecasts_dir = Path.home() / "Desktop" / "Operational_PyCPT_Forecasts"

# The name of this season's forecast. Will be used as a subdirectory of the above.
forecast_name = "prcp-mam-v1"

MOS = 'CCA'

# The models to include in the MME. Do not list models that were
# evaluated in the Jupyter Notebook but not chosen for inclusion
ensemble = [
    'SPEAR.PRCP',
    'CanSIPSIC3.PRCP',
    'CCSM4.PRCP',
    'GEOSS2S.PRCP',
    'CFSv2.PRCP',
]

predictand_name = 'UCSB.PRCP'
local_predictand_file = None

# The list of months in which to issue forecasts (Jan = 1, Feb = 2,
# ...). Unlike in the Jupyter Notebook, you can configure multiple
# issue months here.
issue_months = [12, 1, 2]

download_args = { 
    'target_first_year': 1992,
    'target_final_year': 2020,
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
    'drymask_threshold': None,
    'skillmask_threshold': None,
    'crossvalidation_window': 3,
    'synchronous_predictors': True,
}

# Information about a remote server to which forecasts will be
# uploaded. You will have a chance to look at the forecasts before
# they are uploaded. If you don't plan to run 
remote_host = 'ftp.iri.columbia.edu'
remote_user = 'myuser'
remote_operational_forecasts_dir = '/myuser/operational-forecasts'
