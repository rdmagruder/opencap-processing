'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_arm_rom_analysis.py
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

    This example shows how to run an arm range-of-motion analysis and compute
    normalized reachable workspace (m^2/m^2) and segmented trial duration. Each
    arm is normalized separately, so the left and right arm workspace scalars
    are also computed and summed.

'''

import os
import sys

sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from ActivityAnalyses.arm_rom_analysis import arm_rom_analysis
from utils import get_trial_id, download_trial

# Base directory is the repository root (one level above Examples).
baseDir = os.path.join(os.getcwd(), "..")

# %% Paths.
dataFolder = os.path.join(baseDir, "Data")

# %% User-defined variables.
session_id = "a221bc9f-47f7-4d6c-af2c-e8afa7801fb8"
trial_names = ["arm_rom", "arm_rom_hemiparesis", "arm_rom_deltWeakness"]

filter_frequency = 6

visualize_wrist_plots = True

scalar_names = {
    "arm_rom_reachable_workspace",
    "arm_rom_reachable_workspace_left",
    "arm_rom_reachable_workspace_right",
    "arm_rom_trial_time",
}

# %% Run arm ROM analysis for each trial.
sessionDir = os.path.join(dataFolder, session_id)
results = {}

for trial_name in trial_names:
    trial_id = get_trial_id(session_id, trial_name)
    trialName = download_trial(trial_id, sessionDir, session_id=session_id)

    arm_rom = arm_rom_analysis(
        sessionDir,
        trialName,
        lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    )

    if visualize_wrist_plots:
        arm_rom.plot_arm_rom_wrist_height()

    scalars = arm_rom.compute_scalars(scalar_names)
    results[trial_name] = {"arm_rom": arm_rom, "scalars": scalars}

# %% Display scalars for each trial.
print()
for trial_name in trial_names:
    print(f"Arm ROM results: {trial_name}")
    print("-" * 40)
    for key, value in sorted(results[trial_name]["scalars"].items()):
        v = value["value"]
        units = value["units"]
        if isinstance(v, tuple):
            formatted = ", ".join([str(round(x, 2)) for x in v])
            print(f"  {key}: {formatted} {units}")
        else:
            rounded_value = round(v, 2)
            print(f"  {key}: {rounded_value} {units}")
    print()
