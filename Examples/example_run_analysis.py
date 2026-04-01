'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_run_analysis.py
    ---------------------------------------------------------------------------
    Copyright 2023 Stanford University and the Authors

    Author(s): Scott Uhlrich

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0
    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.

    Please contact us for any questions: https://www.opencap.ai/#contact

    This example shows how to run a kinematic analysis of treadmill running data.
    Segmentation uses calcaneus heel-strike peaks in alternating R/L order
    (see run_analysis.segment_running). Scalar metrics match gait_analysis
    except DMU, which is walking-only.

'''

import os
import sys
sys.path.append("../")
sys.path.append("../ActivityAnalyses")

from run_analysis import run_analysis
from utils import get_trial_id, download_trial
from utilsPlotting import plot_dataframe_with_shading

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
session_id = '06a5f1db-b3c2-4613-9b3d-0998759891af'
trial_name = 'run'
# trial_name = 'run_hemiparesis' # run with slight right side weakness and asymmetry

scalar_names = ['gait_speed', 'stride_length', 'step_width', 'step_length_symmetry']

n_run_cycles = -1

filter_frequency = 6

# %% Run analysis.
trial_id = get_trial_id(session_id, trial_name)

sessionDir = os.path.join(dataFolder, session_id)

trialName = download_trial(trial_id, sessionDir, session_id=session_id)

run_r = run_analysis(
    sessionDir, trialName, leg='r',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_run_cycles=n_run_cycles,
    run_style='treadmill')
run_l = run_analysis(
    sessionDir, trialName, leg='l',
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency,
    n_run_cycles=n_run_cycles,
    run_style='treadmill')

runResults = {}
runResults['scalars_r'] = run_r.compute_scalars(scalar_names)
runResults['curves_r'] = run_r.get_coordinates_normalized_time()
runResults['scalars_l'] = run_l.compute_scalars(scalar_names)
runResults['curves_l'] = run_l.get_coordinates_normalized_time()

# %% Print scalar results.
print('\nRight foot run metrics:')
print('-----------------')
for key, value in runResults['scalars_r'].items():
    rounded_value = round(value['value'], 2)
    print(f"{key}: {rounded_value} {value['units']}")

print('\nLeft foot run metrics:')
print('-----------------')
for key, value in runResults['scalars_l'].items():
    rounded_value = round(value['value'], 2)
    print(f"{key}: {rounded_value} {value['units']}")


# %% Compare right and left time-normalized kinematics.
plot_dataframe_with_shading(
    [runResults['curves_r']['mean'], runResults['curves_l']['mean']],
    [runResults['curves_r']['sd'], runResults['curves_l']['sd']],
    leg=['r', 'l'],
    xlabel='% gait cycle',
    title='kinematics (m or deg)',
    legend_entries=['right', 'left'])
