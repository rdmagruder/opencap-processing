'''
    ---------------------------------------------------------------------------
    OpenCap processing: example_wbam_analysis.py
    ---------------------------------------------------------------------------

    Copyright 2026 University of Utah and the Authors
    
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

    This example computes whole-body angular momentum (WBAM) about the center
    of mass for a walking trial and plots pelvis translation, CoM trajectory,
    and WBAM time series.
    
'''

import os
import sys

sys.path.append("..")
sys.path.append("../ActivityAnalyses")

import matplotlib.pyplot as plt
from gait_analysis import gait_analysis
from utils import get_trial_id, download_trial

# %% Paths.
baseDir = os.path.join(os.getcwd(), '..')
dataFolder = os.path.join(baseDir, 'Data')

# %% User-defined variables.
# Default overground walking trial from example_gait_analysis.py.
session_id = 'b39b10d1-17c7-4976-b06c-a6aaf33fead2'
trial_name = 'gait_3'

# Lowpass filter frequency for kinematics data.
filter_frequency = 6

# %% Download data and compute kinematics.
trial_id = get_trial_id(session_id, trial_name)
sessionDir = os.path.join(dataFolder, session_id)
trialName = download_trial(trial_id, sessionDir, session_id=session_id)

gait = gait_analysis(
    sessionDir,
    trialName,
    lowpass_cutoff_frequency_for_coordinate_values=filter_frequency)

coordinates = gait.get_coordinate_values()
com_values = gait.get_center_of_mass_values(
    lowpass_cutoff_frequency=filter_frequency)
wbam_values = gait.get_whole_body_angular_momentum(
    lowpass_cutoff_frequency=filter_frequency)

time = coordinates['time']

# Gait events from marker-based segmentation (ipsilateral + contralateral).
gait_events = gait.get_gait_events()
if gait_events['ipsilateralLeg'] == 'r':
    r_heel_strike_times = gait_events['ipsilateralTime'][:, (0, 2)].flatten()
    r_toe_off_times = gait_events['ipsilateralTime'][:, 1]
    l_heel_strike_times = gait_events['contralateralTime'][:, 1]
    l_toe_off_times = gait_events['contralateralTime'][:, 0]
else:
    l_heel_strike_times = gait_events['ipsilateralTime'][:, (0, 2)].flatten()
    l_toe_off_times = gait_events['ipsilateralTime'][:, 1]
    r_heel_strike_times = gait_events['contralateralTime'][:, 1]
    r_toe_off_times = gait_events['contralateralTime'][:, 0]

# %% Plot pelvis x, CoM trajectory, and WBAM curves.
fig, axs = plt.subplots(3, 1, figsize=(8, 9), sharex=True)

axs[0].plot(time, coordinates['pelvis_tx'], linewidth=2, color='C0')
axs[0].set_ylabel('Pelvis x (m)')

for axis, color in zip(['x', 'y', 'z'], ['C0', 'C1', 'C2']):
    axs[1].plot(time, com_values[axis], linewidth=2, color=color, label=axis)
axs[1].set_ylabel('CoM position (m)')
axs[1].legend(fontsize=12)

for axis, color in zip(['x', 'y', 'z'], ['C0', 'C1', 'C2']):
    axs[2].plot(time, wbam_values[axis], linewidth=2, color=color, label=axis)

gait_event_styles = [
    (r_heel_strike_times, 'C3', 'R heel strike'),
    (r_toe_off_times, 'C4', 'R toe-off'),
    (l_heel_strike_times, 'C5', 'L heel strike'),
    (l_toe_off_times, 'C6', 'L toe-off'),
]
for event_times, color, label in gait_event_styles:
    for i, event_time in enumerate(event_times):
        axs[2].axvline(
            event_time,
            color=color,
            linestyle='--',
            linewidth=1.2,
            alpha=0.8,
            label=label if i == 0 else None)

axs[2].set_ylabel('WBAM (1/s, normalized by m·h²)')
axs[2].set_xlabel('Time (s)')
axs[2].legend(fontsize=10, ncol=2)

for ax in axs:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='major', labelsize=12)

fig.suptitle(f'Whole-body angular momentum: {trial_name}', fontsize=14)
fig.align_ylabels(axs)
fig.tight_layout()

plt.show()
