'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_toe_stand_analysis.py
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

    This example downloads toe-stand session trials, segments each trial from the
    mean calcaneus marker height (first toe stand through final landing), and
    prints scalar metrics (duration, mean heel height, mean ankle angle, COM path
    metrics). COM stability uses session metadata height_m when normalize_by_height
    is True; set it False for mean COM path speed (m/s).

'''


import os
import sys

sys.path.append("../")
sys.path.append("../ActivityAnalyses")
from ActivityAnalyses.toe_stand_analysis import toe_stand_analysis
from utils import get_trial_id, download_trial

baseDir = os.path.join(os.getcwd(), "..")

# %% Paths.
dataFolder = os.path.join(baseDir, "Data")

# %% User-defined variables.
session_id = "c5b65f0b-0743-438f-8157-5bac6e60174a"
trial_names = ["ToeStand", "ToeStand_lowBalance", "ToeStand_kneeBend"]
# ToeStand: reference toe-stand task
# ToeStand_lowBalance: lower balance / more sway during toe stand and stepping
# ToeStand_kneeBend: knee flexion during the task, commonly used compensatory strategy to lower COM and increase stability

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Set to True to show mean calcaneus Y segmentation plot for each trial.
visualize_calc_plots = True

# True: COM stability = (path/height_m)/duration using height_m from sessionMetadata.yaml
# False: COM stability = path/duration (m/s), no height normalization
normalize_by_height = False

scalar_names = {
    "standing_duration",
    "mean_heel_height",
    "mean_ankle_angle",
    "com_stability_normalized_velocity_3d",
    "com_stability_normalized_velocity_horizontal",
}

# %% Run toe stand analysis for each trial.
sessionDir = os.path.join(dataFolder, session_id)
results = {}

for trial_name in trial_names:
    trial_id = get_trial_id(session_id, trial_name)
    trialName = download_trial(trial_id, sessionDir, session_id=session_id)

    ts = toe_stand_analysis(
        sessionDir,
        trialName,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
        normalize_by_height=normalize_by_height,
    )

    if visualize_calc_plots:
        ts.segment_toe_stand(visualize=True)

    results[trial_name] = {"toe_stand": ts, "scalars": ts.compute_scalars(scalar_names)}

# %% Display scalars for each trial.
print()
for trial_name in trial_names:
    print(f"Toe stand results: {trial_name}")
    print("-" * 40)
    for key, value in sorted(results[trial_name]["scalars"].items()):
        if isinstance(value["value"], tuple):
            left_value, right_value = value["value"]
            left_rounded = round(left_value, 2)
            right_rounded = round(right_value, 2)
            print(
                f"  {key}: Left = {left_rounded} {value['units']}, "
                f"Right = {right_rounded} {value['units']}"
            )
        else:
            rounded_value = round(value["value"], 4)
            print(f"  {key}: {rounded_value} {value['units']}")
    print()
