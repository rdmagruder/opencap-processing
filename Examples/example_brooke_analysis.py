'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_brooke_analysis.py
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

    This example shows how to run a kinematic analysis of Brooke activity data
    and compute scalar metrics.

'''


import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")
from ActivityAnalyses.brooke_analysis import brooke_analysis
from utils import get_trial_id, download_trial

baseDir = os.path.join(os.getcwd(), '..')

# %% Paths.
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
session_id = '24db43d3-77c2-49c8-88cb-819f0ee06748'
trial_names = ['brooke', 'brooke_elbow_flexion', 'brooke_hemiparesis']
# 'brooke' is how a unimpaired person may perform the brooke test
# 'brooke_elbow_flexion' includes trunk compensations and bent elbows during ascent, which may be observed in people with upper extremity weakness
# 'brooke_hemiparesis' shows right arm impairment for lateralized impairment analysis

# Select lowpass filter frequency for kinematics data.
filter_frequency = 6

# Set to True to show wrist Y segmentation plot for each trial.
visualize_wrist_plots = True

scalar_names = {
    'max_elbow_flexion_ascent',
    'max_height_above_shoulder',
    'ascent_path_length',
    'trunk_lean_at_peak',
    'time_to_peak',
    'cycle_duration',
}

# %% Run Brooke analysis for each trial.
sessionDir = os.path.join(dataFolder, session_id)
results = {}

for trial_name in trial_names:
    trial_id = get_trial_id(session_id, trial_name)
    trialName = download_trial(trial_id, sessionDir, session_id=session_id)

    brooke = brooke_analysis(
        sessionDir, trialName,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    )

    if visualize_wrist_plots:
        brooke.segment_brooke(min_rest_interval_s=brooke.min_rest_interval_s, visualize=True)

    results[trial_name] = {'brooke': brooke, 'scalars': brooke.compute_scalars(scalar_names)}

# %% Display scalars for each trial.
print()
for trial_name in trial_names:
    print(f"Brooke results: {trial_name}")
    print("-" * 40)
    for key, value in sorted(results[trial_name]['scalars'].items()):
        if isinstance(value['value'], tuple):
            left_value, right_value = value['value']
            left_rounded = round(left_value, 2)
            right_rounded = round(right_value, 2)
            print(f"  {key}: Left = {left_rounded} {value['units']}, Right = {right_rounded} {value['units']}")
        else:
            rounded_value = round(value['value'], 2)
            print(f"  {key}: {rounded_value} {value['units']}")
    print()
