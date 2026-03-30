'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_jump_analysis.py
    ---------------------------------------------------------------------------

    This example shows how to run a kinematic analysis of countermovement
    jump data (CMJ) using COM-based segmentation and compute scalar metrics.
'''

import os
import sys

sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from ActivityAnalyses.jump_analysis import jump_analysis
from utils import get_trial_id, download_trial


# Base directory is the repository root (one level above Examples).
baseDir = os.path.join(os.getcwd(), "..")

# %% Paths.
dataFolder = os.path.join(baseDir, "Data")

# %% User-defined variables.
# This session/trial pair is used in the existing jump forceplate example.
session_id = "4b038322-470b-49ec-98e6-a8d75ff54792"
trial_name = "jump"
# trial_name = "jump_hemiparesis" # This simulates left side weakness as an example for lateralized impairments
# trial_name = "jump_noAir" # An example of segmentation when the participant is incapable of leaving the ground

# Lowpass filter frequency for kinematics data.
filter_frequency = 6

# Set to True to show the COM segmentation plot.
visualize_com_plots = True

scalar_names = {"rise_time", "flight_time", "jump_height", "max_com_vel_rise"}

# %% Run Jump analysis for the selected trial.
trial_id = get_trial_id(session_id, trial_name)
sessionDir = os.path.join(dataFolder, session_id)
trialName = download_trial(trial_id, sessionDir, session_id=session_id)

jump = jump_analysis(
    sessionDir,
    trialName,
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
)

if visualize_com_plots:
    # Re-run segmentation with plots enabled.
    jump.segment_jump(visualize=True)

scalars = jump.compute_scalars(scalar_names)

# %% Display scalars.
print()
print(f"Jump results: {trial_name}")
print("-" * 40)
for key, value in sorted(scalars.items()):
    v = value["value"]
    units = value["units"]
    rounded_value = round(v, 4)
    print(f"  {key}: {rounded_value} {units}")
print()

