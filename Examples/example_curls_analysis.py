'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_curls_analysis.py
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

    This example shows first-phase curl segmentation using arm flexion
    coordinates (arm_flex_r, arm_flex_l), selecting the first prominent
    flexion peak and rest-flattening boundaries.
'''

import os
import sys

sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from ActivityAnalyses.curls_analysis import curls_analysis
from utils import get_trial_id, download_trial

baseDir = os.path.join(os.getcwd(), "..")

# %% Paths.
dataFolder = os.path.join(baseDir, "Data")

# %% User-defined variables.
session_id = "02269d40-f70f-4f57-a6ec-4473b2c3123f"
trial_names = ["curls", "curls_hemiparesis", "curls_momentum"]

filter_frequency = 6
visualize_curl_plot = True

scalar_names = {
    "curl_peak_flexion",
    "curl_excursion",
    "curl_duration",
}

# %% Run curls analysis for each trial.
sessionDir = os.path.join(dataFolder, session_id)
results = {}

for trial_name in trial_names:
    trial_id = get_trial_id(session_id, trial_name)
    trialName = download_trial(trial_id, sessionDir, session_id=session_id)

    curls = curls_analysis(
        sessionDir,
        trialName,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    )

    if visualize_curl_plot:
        curls.segment_curls(visualize=True)

    scalars = curls.compute_scalars(scalar_names)
    results[trial_name] = {"curls": curls, "scalars": scalars}

# %% Display segmentation events and scalars.
print()
for trial_name in trial_names:
    print(f"Curls results: {trial_name}")
    print("-" * 40)

    for key, value in sorted(results[trial_name]["scalars"].items()):
        rounded_value = round(value["value"], 2)
        print(f"  {key}: {rounded_value} {value['units']}")
    print()
