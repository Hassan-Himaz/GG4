from ZLDSParams import LDSParams
from pathlib import Path
def get_saved_params()->LDSParams:
    load_dir = Path.cwd() / "LDSParams_Saves"/'run_002_sim'
    return LDSParams.load(load_dir)
      