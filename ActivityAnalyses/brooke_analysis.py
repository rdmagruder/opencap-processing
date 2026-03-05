"""
    ---------------------------------------------------------------------------
    OpenCap processing: brooke_analysis.py
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
"""


import numpy as np
import matplotlib.pyplot as plt

from utilsKinematics import kinematics


BROOKE_MARKER_KEYS = ['r_lwrist_study', 'L_lwrist_study', 'r_mwrist_study', 'L_mwrist_study']

# For arm length: shoulder, elbow midpoint, wrist midpoint per side
ARM_LENGTH_MARKERS = {
    'r': {
        'shoulder': 'r_shoulder_study',
        'elbow_l': 'r_lelbow_study',
        'elbow_m': 'r_melbow_study',
        'wrist_l': 'r_lwrist_study',
        'wrist_m': 'r_mwrist_study',
    },
    'l': {
        'shoulder': 'L_shoulder_study',
        'elbow_l': 'L_lelbow_study',
        'elbow_m': 'L_melbow_study',
        'wrist_l': 'L_lwrist_study',
        'wrist_m': 'L_mwrist_study',
    },
}


class brooke_analysis(kinematics):
    """Kinematics-only analysis for the Brooke activity: lateral arm raise and return.
    Segments a single cycle using the two globally smallest wrist height minima at least 2 s apart.
    """

    def __init__(self, session_dir, trial_name,
                 lowpass_cutoff_frequency_for_coordinate_values=6,
                 min_rest_interval_s=2.0,
                 trimming_start=0, trimming_end=0):
        super().__init__(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values,
        )
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.min_rest_interval_s = min_rest_interval_s

        self.markerDict = self.get_marker_dict(
            session_dir, trial_name,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )
        self.coordinateValues = self.get_coordinate_values()

        if self.trimming_start > 0:
            self.idx_trim_start = np.where(
                np.round(self.markerDict['time'] - self.trimming_start, 6) <= 0
            )[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:, :]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = np.where(
                np.round(self.markerDict['time'], 6)
                <= np.round(self.markerDict['time'][-1] - self.trimming_end, 6)
            )[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end, :]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]

        self.body_angles = self.get_body_orientation(
            expressed_in='ground',
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )
        if self.trimming_start > 0:
            self.body_angles = self.body_angles.iloc[self.idx_trim_start:]
        if self.trimming_end > 0:
            self.body_angles = self.body_angles.iloc[:self.idx_trim_end]

        self.brookeEvents = self.segment_brooke(min_rest_interval_s=self.min_rest_interval_s)
        self.n_brooke_cycles = 1

        self._arm_length_cache = None

    def segment_brooke(self, min_rest_interval_s=2.0, height_threshold_m=0.02, visualize=False):
        """Segment one Brooke cycle: find mean peak index, then traverse backward and forward in time
        to find when arms get within height_threshold_m (default 2 cm) of the global minimum as start and end.
        """
        markers = self.markerDict['markers']
        for key in BROOKE_MARKER_KEYS:
            if key not in markers:
                raise ValueError(
                    f"Marker '{key}' not found in markerDict. Available: {list(markers.keys())}"
                )

        time_vec = self.markerDict['time']

        # Height signal: mean of four markers' global Y (column index 1)
        ys = np.stack([markers[k][:, 1] for k in BROOKE_MARKER_KEYS], axis=1)
        height_signal = np.mean(ys, axis=1)

        global_min_height = float(np.min(height_signal))
        peak_idx = int(np.argmax(height_signal))
        rest_threshold = global_min_height + height_threshold_m

        # Traverse backward from peak to find first frame where mean height is within 2 cm of global min
        start_idx = peak_idx
        for i in range(peak_idx - 1, -1, -1):
            if height_signal[i] <= rest_threshold:
                start_idx = i
                break

        # Traverse forward from peak to find first frame where mean height is within 2 cm of global min
        end_idx = peak_idx
        n_frames = len(height_signal)
        for i in range(peak_idx + 1, n_frames):
            if height_signal[i] <= rest_threshold:
                end_idx = i
                break

        start_time = float(time_vec[start_idx])
        end_time = float(time_vec[end_idx])

        brookeEvents = {
            'restIdx': np.array([start_idx, end_idx]),
            'restTime': np.array([start_time, end_time]),
            'startIdx': start_idx,
            'endIdx': end_idx,
            'startTime': start_time,
            'endTime': end_time,
        }

        if visualize:
            self._plot_brooke_segmentation(height_signal, time_vec, brookeEvents, ys)

        return brookeEvents

    def _plot_brooke_segmentation(self, height_signal, time_vec, brookeEvents, ys):
        """Plot mean wrist height and optional per-marker Y with rest points and cycle span."""
        plt.figure()
        plt.plot(time_vec, height_signal, 'k-', label='Mean wrist height')
        for i, key in enumerate(BROOKE_MARKER_KEYS):
            plt.plot(time_vec, ys[:, i], '--', alpha=0.6, label=key)
        r1, r2 = brookeEvents['restIdx'][0], brookeEvents['restIdx'][1]
        plt.plot(time_vec[r1], height_signal[r1], 'go', markersize=10, label='Rest 1')
        plt.plot(time_vec[r2], height_signal[r2], 'go', markersize=10, label='Rest 2')
        plt.axvspan(
            brookeEvents['startTime'], brookeEvents['endTime'],
            alpha=0.2, color='gray', label='Cycle'
        )
        plt.xlabel('Time [s]')
        plt.ylabel('Height [m]')
        plt.title('Brooke activity: wrist height and segment')
        plt.legend(loc='lower right', fontsize=8)
        plt.tight_layout()
        plt.show()

    def _height_signal_segment(self):
        """Mean wrist Y over the single cycle segment (startIdx to endIdx inclusive)."""
        start = self.brookeEvents['startIdx']
        end = self.brookeEvents['endIdx'] + 1
        ys = np.stack(
            [self.markerDict['markers'][k][start:end, 1] for k in BROOKE_MARKER_KEYS],
            axis=1,
        )
        return np.mean(ys, axis=1)

    def _peak_frame_global(self):
        """Global frame index where mean wrist height is maximum within the cycle."""
        start = self.brookeEvents['startIdx']
        height = self._height_signal_segment()
        local_peak = int(np.argmax(height))
        return start + local_peak

    def _arm_length(self):
        """Arm length per side (upper arm + forearm from markers). Returns {'r': float, 'l': float}."""
        if self._arm_length_cache is not None:
            return self._arm_length_cache
        markers = self.markerDict['markers']
        lengths = {}
        for side, keys in ARM_LENGTH_MARKERS.items():
            for k in keys.values():
                if k not in markers:
                    raise ValueError(
                        f"Marker '{k}' not found for arm length. Available: {list(markers.keys())}"
                    )
            sh = markers[keys['shoulder']]
            elb = (markers[keys['elbow_l']] + markers[keys['elbow_m']]) / 2
            wr = (markers[keys['wrist_l']] + markers[keys['wrist_m']]) / 2
            upper = np.mean(np.linalg.norm(elb - sh, axis=1))
            forearm = np.mean(np.linalg.norm(wr - elb, axis=1))
            lengths[side] = float(upper + forearm)
        self._arm_length_cache = lengths
        return lengths

    def compute_scalars(self, scalarNames, return_all=False):
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if entry.startswith('compute_')]

        if scalarNames is None:
            print('No scalars defined, these methods are available:')
            print(*[m.replace('compute_', '') for m in possibleMethods])
            return

        nonexistant = [
            entry for entry in scalarNames
            if 'compute_' + entry not in method_names
        ]
        if nonexistant:
            raise Exception(
                str(['compute_' + a for a in nonexistant])
                + ' does not exist in brooke_analysis class.'
            )

        scalarDict = {}
        for scalarName in scalarNames:
            fn = getattr(self, 'compute_' + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]['value'], scalarDict[scalarName]['units']) = fn(
                return_all=return_all
            )
        return scalarDict

    def compute_max_elbow_flexion_ascent(self, return_all=False):
        """Max elbow flexion during arm raise (until max height), left and right separately (deg)."""
        start = self.brookeEvents['startIdx']
        peak = self._peak_frame_global()
        ascent_slice = slice(start, peak + 1)
        max_r = float(np.max(self.coordinateValues['elbow_flex_r'].iloc[ascent_slice]))
        max_l = float(np.max(self.coordinateValues['elbow_flex_l'].iloc[ascent_slice]))
        value = (max_l, max_r)
        units = 'deg'
        if return_all:
            return [value], units
        return value, units

    def compute_max_height_above_shoulder(self, return_all=False):
        """Max height above shoulder over the cycle, normalized by arm length, left and right separately."""
        start = self.brookeEvents['startIdx']
        end = self.brookeEvents['endIdx'] + 1
        lengths = self._arm_length()
        markers = self.markerDict['markers']
        values_norm = []
        for side in ('l', 'r'):
            keys = ARM_LENGTH_MARKERS[side]
            sh_y = markers[keys['shoulder']][start:end, 1]
            w1_y = markers[keys['wrist_l']][start:end, 1]
            w2_y = markers[keys['wrist_m']][start:end, 1]
            wrist_y_max = np.maximum(w1_y, w2_y)
            height_above = wrist_y_max - sh_y
            max_above = float(np.max(height_above))
            values_norm.append(max_above / lengths[side])
        value = (values_norm[0], values_norm[1])
        units = 'arm lengths'
        if return_all:
            return [value], units
        return value, units

    def compute_ascent_path_length(self, return_all=False):
        """Path length of wrist during arm ascent (start to peak), normalized by arm length, per arm."""
        start = self.brookeEvents['startIdx']
        peak = self._peak_frame_global()
        lengths = self._arm_length()
        markers = self.markerDict['markers']
        values_norm = []
        for side in ('l', 'r'):
            keys = ARM_LENGTH_MARKERS[side]
            wrist_pos = (markers[keys['wrist_l']] + markers[keys['wrist_m']]) / 2
            seg = wrist_pos[start:peak + 1]
            diffs = np.diff(seg, axis=0)
            path = float(np.sum(np.linalg.norm(diffs, axis=1)))
            values_norm.append(path / lengths[side])
        value = (values_norm[0], values_norm[1])
        units = 'arm lengths'
        if return_all:
            return [value], units
        return value, units

    def compute_trunk_lean_at_peak(self, return_all=False):
        """Trunk lean in global frame at the point of highest arms (forward lean convention, deg)."""
        peak = self._peak_frame_global()
        torso_z = self.body_angles['torso_z']
        lean = float(np.degrees(torso_z.iloc[peak]) * -1)
        units = 'deg'
        if return_all:
            return [lean], units
        return lean, units

    def compute_time_to_peak(self, return_all=False):
        """Time from cycle start to frame of max height (s)."""
        start = self.brookeEvents['startIdx']
        end = self.brookeEvents['endIdx'] + 1
        height = self._height_signal_segment()
        time_vec = self.markerDict['time'][start:end]
        peak_idx = int(np.argmax(height))
        value = float(time_vec[peak_idx] - time_vec[0])
        units = 's'
        if return_all:
            return [value], units
        return value, units

    def compute_cycle_duration(self, return_all=False):
        """Cycle duration (s)."""
        value = float(self.brookeEvents['endTime'] - self.brookeEvents['startTime'])
        units = 's'
        if return_all:
            return [value], units
        return value, units
