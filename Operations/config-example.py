# This is a configuration file for the PyCPT scripts
# generate-forecasts and upload-forecasts. See in-line comments for
# customization instructions.

import datetime as dt
from pathlib import Path

######################################################################
#
# Replace these values with the ones from your PyCPT Jupyter
# Notebook. Their meaning is documented there.
#
######################################################################

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

######################################################################
#
# End of values copied directly from Jupyter Notebook
#
######################################################################


# The list of months in which to issue forecasts (Jan = 1, Feb = 2,
# ...). You can configure multiple issue months.
issue_months = [12, 1, 2]

# The models to include in the MME. Do not list models that were
# evaluated in the Jupyter Notebook but not chosen for inclusion
ensemble = [ "SPEAR.PRCP", 'CanSIPSIC3.PRCP','CCSM4.PRCP','GEOSS2S.PRCP','CFSv2.PRCP']

# The path where forecasts for all seasons (including this one) will
# be stored. This should NOT be the same as the case_dir used in the
# Jupyter Notebook. That one is for experimentation, this one is for
# operational forecasts generated after the forecast configuration has
# been finalized.
operational_forecasts_dir = Path.home() / "Desktop" / "Operational_PyCPT_Forecasts"

# The name of this season's forecast. Will be used as a subdirectory of the above.
forecast_name = "MAM_v1"

# Information about a remote server to which forecasts will be
# uploaded. You will have a chance to look at the forecasts before
# they are uploaded. If you don't plan to run 
remote_host = 'ftp.iri.columbia.edu'
remote_user = 'myuser'
remote_operational_forecasts_dir = '/myuser/operational-forecasts'

# Initializations for which we need to generate the forecast by hand,
# e.g. because one of the models isn't available.
skip_issue_dates = [
    # dt.datetime(2020, 12, 1),
]
