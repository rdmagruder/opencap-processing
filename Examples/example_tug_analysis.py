'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_tug_analysis.py
    ---------------------------------------------------------------------------

    Copyright TBD

    Author(s): RD Magruder

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of Timed Up and Go (TUG)
    activity data and compute scalar metrics based on COM and torso motion.

'''

import os
import sys

sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from ActivityAnalyses.tug_analysis import tug_analysis
from utils import get_trial_id, download_trial

# Base directory is the repository root (one level above Examples).
baseDir = os.path.join(os.getcwd(), "..")

# %% Paths.
dataFolder = os.path.join(baseDir, "Data")

# %% User-defined variables.
# Session and trials for TUG example.
session_id = "c292a43d-a36e-4226-8660-c7df71b47793"
trial_names = ["tug_fast", "tug_slow", "tug_hemiparesis"]

# Lowpass filter frequency for kinematics data.
filter_frequency = 6

# Set to True to show COM Y segmentation plot for each trial.
visualize_com_plots = True

# Scalars to compute from the TUG analysis.
scalar_names = {
    "tug_time",
    "torso_orientation_liftoff",
    "torso_angular_velocity",

}

# %% Run TUG analysis for each trial.
sessionDir = os.path.join(dataFolder, session_id)
results = {}

for trial_name in trial_names:
    trial_id = get_trial_id(session_id, trial_name)
    trialName = download_trial(trial_id, sessionDir, session_id=session_id)

    tug = tug_analysis(
        sessionDir,
        trialName,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    )

    if visualize_com_plots:
        # Either re-run segmentation with visualize=True or use the helper.
        tug.segment_tug(visualize=True)
        # Alternatively:
        # tug.plot_tug_COM()

    scalars = tug.compute_scalars(scalar_names)
    results[trial_name] = {"tug": tug, "scalars": scalars}

# %% Display scalars for each trial.
print()
for trial_name in trial_names:
    print(f"TUG results: {trial_name}")
    print("-" * 40)
    for key, value in sorted(results[trial_name]["scalars"].items()):
        v = value["value"]
        units = value["units"]
        # Handle both scalar and tuple outputs for robustness.
        if isinstance(v, tuple):
            # If a tuple appears in future metrics, print elements separately.
            formatted = ", ".join([str(round(x, 2)) for x in v])
            print(f"  {key}: {formatted} {units}")
        else:
            rounded_value = round(v, 2)
            print(f"  {key}: {rounded_value} {units}")
    print()

