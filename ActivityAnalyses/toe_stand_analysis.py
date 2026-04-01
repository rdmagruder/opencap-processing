"""
    ---------------------------------------------------------------------------
    OpenCap processing: toe_stand_analysis.py
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

import os

import numpy as np
import matplotlib.pyplot as plt

import utils
from utilsKinematics import kinematics


CALC_MARKER_KEYS = ["r_calc_study", "L_calc_study"]


class toe_stand_analysis(kinematics):
    """Kinematics-only analysis for toe stand: segment from mean calcaneus height.

    Segmentation uses the mean vertical (Y) position of r_calc_study and L_calc_study.
    Quiet standing height is estimated from the start of the trial. The segment begins
    at the first upward crossing from the ground band into lifted height (first toe stand)
    and ends at the last downward crossing from lifted height back into the ground band
    (final landing), so intervening falls and re-stands remain inside the segment.

    COM stability scalars use ``height_m`` from ``sessionMetadata.yaml`` in ``session_dir``
    when ``normalize_by_height`` is True (same source as ``processInputsOpenSimAD``).
    If ``normalize_by_height`` is False, COM metrics are mean path speed (path length /
    segment duration) in m/s.
    """

    def __init__(
        self,
        session_dir,
        trial_name,
        lowpass_cutoff_frequency_for_coordinate_values=6,
        trimming_start=0,
        trimming_end=0,
        normalize_by_height=True,
        body_height_m=None,
        baseline_window_s=0.25,
        lift_off_m=0.04,
        min_segment_frames=3,
    ):
        super().__init__(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values,
        )
        self._session_dir = session_dir
        self.trimming_start = trimming_start
        self.trimming_end = trimming_end
        self.normalize_by_height = normalize_by_height
        self.body_height_m = body_height_m
        self.baseline_window_s = baseline_window_s
        self.lift_off_m = lift_off_m
        self.min_segment_frames = min_segment_frames

        self.markerDict = self.get_marker_dict(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )
        self.coordinateValues = self.get_coordinate_values()

        if self.trimming_start > 0:
            self.idx_trim_start = np.where(
                np.round(self.markerDict["time"] - self.trimming_start, 6) <= 0
            )[0][-1]
            self.markerDict["time"] = self.markerDict["time"][self.idx_trim_start :]
            for marker in self.markerDict["markers"]:
                self.markerDict["markers"][marker] = self.markerDict["markers"][marker][
                    self.idx_trim_start :, :
                ]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start :]

        if self.trimming_end > 0:
            self.idx_trim_end = (
                np.where(
                    np.round(self.markerDict["time"], 6)
                    <= np.round(
                        self.markerDict["time"][-1] - self.trimming_end,
                        6,
                    )
                )[0][-1]
                + 1
            )
            self.markerDict["time"] = self.markerDict["time"][: self.idx_trim_end]
            for marker in self.markerDict["markers"]:
                self.markerDict["markers"][marker] = self.markerDict["markers"][marker][
                    : self.idx_trim_end, :
                ]
            self.coordinateValues = self.coordinateValues.iloc[: self.idx_trim_end]

        self._comValues = self.get_center_of_mass_values(
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values
        )
        if self.trimming_start > 0:
            self._comValues = self._comValues.iloc[self.idx_trim_start :]
        if self.trimming_end > 0:
            self._comValues = self._comValues.iloc[: self.idx_trim_end]

        self.toeStandEvents = self.segment_toe_stand()

    def _calc_height_signal(self):
        markers = self.markerDict["markers"]
        for key in CALC_MARKER_KEYS:
            if key not in markers:
                raise ValueError(
                    f"Marker '{key}' not found in markerDict. Available: {list(markers.keys())}"
                )
        r_y = markers["r_calc_study"][:, 1]
        l_y = markers["L_calc_study"][:, 1]
        return 0.5 * (r_y + l_y)

    def _height_m_for_com_normalization(self):
        """Standing height (m) from session metadata or explicit override.

        Matches ``processInputsOpenSimAD``: ``metadata['height_m']`` from
        ``sessionMetadata.yaml`` in the session folder. Used only when
        ``normalize_by_height`` is True.
        """
        if self.body_height_m is not None:
            h = float(self.body_height_m)
        else:
            meta_path = os.path.join(self._session_dir, "sessionMetadata.yaml")
            if not os.path.isfile(meta_path):
                raise ValueError(
                    "No sessionMetadata.yaml in session_dir; cannot read height_m. "
                    f"Expected: {meta_path}. "
                    "Pass body_height_m=... or add metadata from OpenCap."
                )
            metadata = utils.import_metadata(meta_path)
            if "height_m" not in metadata:
                raise ValueError(
                    "sessionMetadata.yaml has no 'height_m'. "
                    "Pass body_height_m=... explicitly."
                )
            h = float(metadata["height_m"])
        if h <= 0 or not np.isfinite(h):
            raise ValueError(
                "Invalid height_m for COM normalization; "
                "set body_height_m to a positive value (m)."
            )
        return h

    def segment_toe_stand(self, visualize=False):
        """Segment toe-stand episode using mean calcaneus Y vs quiet baseline.

        rest_height: median mean-calc Y over the first baseline_window_s.

        Start: first index i with height[i-1] <= ground_level and height[i] >= lift_level
        (first departure from quiet standing into toe stand).

        End: last index j > start with height[j-1] >= lift_level and height[j] <= ground_level
        (final landing). If none found, raises ValueError.
        """
        time_vec = self.markerDict["time"]
        height_signal = self._calc_height_signal()
        n_frames = len(height_signal)
        if n_frames < self.min_segment_frames + 2:
            raise ValueError("Trial too short for toe stand segmentation.")

        dt = float(np.mean(np.diff(time_vec)))
        if dt <= 0:
            raise ValueError("Invalid time step in marker data.")

        baseline_n = max(3, int(round(self.baseline_window_s / dt)))
        baseline_n = min(baseline_n, n_frames // 4 or n_frames)
        rest_height = float(np.median(height_signal[:baseline_n]))

        lift_level = rest_height + self.lift_off_m

        start_idx = None
        for i in range(1, n_frames):
            if height_signal[i] >= lift_level:
                start_idx = i
                break

        if start_idx is None:
            raise ValueError(
                "Could not detect first toe stand: no upward crossing from ground band "
                "to lift level. Try adjusting lift_off_m."
            )

        crossings_down = []
        for j in range(start_idx + 1, n_frames):
            if height_signal[j - 1] >= lift_level:
                crossings_down.append(j)

        if not crossings_down:
            raise ValueError(
                "Could not detect final landing: no downward crossing to ground band "
                "after first toe stand. Try adjusting thresholds or trimming the trial."
            )

        end_idx = int(crossings_down[-1])

        if end_idx <= start_idx:
            raise ValueError("Invalid segment: end index must be after start index.")

        if end_idx - start_idx + 1 < self.min_segment_frames:
            raise ValueError("Segment shorter than min_segment_frames.")

        start_time = float(time_vec[start_idx])
        end_time = float(time_vec[end_idx])

        toe_stand_events = {
            "startIdx": start_idx,
            "endIdx": end_idx,
            "startTime": start_time,
            "endTime": end_time,
            "restHeight": rest_height,
            "liftLevel": float(lift_level),
            "liftOffM": float(self.lift_off_m),
        }

        if visualize:
            self._plot_toe_stand_segmentation(
                height_signal, time_vec, toe_stand_events
            )

        return toe_stand_events

    def _plot_toe_stand_segmentation(self, height_signal, time_vec, events):
        plt.figure()
        plt.plot(time_vec, height_signal, "k-", label="Mean calcaneus Y")
        plt.axhline(events["restHeight"], color="gray", linestyle=":", label="Rest height")
        plt.axhline(events["liftLevel"], color="b", linestyle="--", alpha=0.7, label="Lift level")
        s, e = events["startIdx"], events["endIdx"]
        plt.plot(time_vec[s], height_signal[s], "go", markersize=10, label="Start (first stand)")
        plt.plot(time_vec[e], height_signal[e], "ro", markersize=10, label="End (final landing)")
        plt.axvspan(events["startTime"], events["endTime"], alpha=0.2, color="gray", label="Segment")
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.title("Toe stand: mean calcaneus height and segment")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    def _segment_slice(self):
        s = self.toeStandEvents["startIdx"]
        e = self.toeStandEvents["endIdx"] + 1
        return slice(s, e)

    def _com_path_metrics(self):
        sl = self._segment_slice()
        com = self._comValues.iloc[sl]
        xyz = com[["x", "y", "z"]].to_numpy(dtype=float)
        if len(xyz) < 2:
            return 0.0, 0.0
        d3 = np.diff(xyz, axis=0)
        path_3d = float(np.sum(np.linalg.norm(d3, axis=1)))
        dxz = np.diff(xyz[:, [0, 2]], axis=0)
        path_hz = float(np.sum(np.linalg.norm(dxz, axis=1)))
        return path_3d, path_hz

    def compute_scalars(self, scalarNames, return_all=False):
        method_names = [func for func in dir(self) if callable(getattr(self, func))]
        possibleMethods = [entry for entry in method_names if entry.startswith("compute_")]

        if scalarNames is None:
            print("No scalars defined, these methods are available:")
            print(*[m.replace("compute_", "") for m in possibleMethods])
            return

        nonexistant = [
            entry for entry in scalarNames if "compute_" + entry not in method_names
        ]
        if nonexistant:
            raise Exception(
                str(["compute_" + a for a in nonexistant])
                + " does not exist in toe_stand_analysis class."
            )

        scalarDict = {}
        for scalarName in scalarNames:
            fn = getattr(self, "compute_" + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]["value"], scalarDict[scalarName]["units"]) = fn(
                return_all=return_all
            )
        return scalarDict

    def compute_standing_duration(self, return_all=False):
        value = float(
            self.toeStandEvents["endTime"] - self.toeStandEvents["startTime"]
        )
        units = "s"
        if return_all:
            return [value], units
        return value, units

    def compute_mean_heel_height(self, return_all=False):
        sl = self._segment_slice()
        markers = self.markerDict["markers"]
        r_y = markers["r_calc_study"][sl, 1]
        l_y = markers["L_calc_study"][sl, 1]
        bilateral = 0.5 * (r_y + l_y)
        value = float(np.mean(bilateral))
        units = "m"
        if return_all:
            return [value], units
        return value, units

    def compute_mean_ankle_angle(self, return_all=False):
        sl = self._segment_slice()
        a_l = np.asarray(self.coordinateValues["ankle_angle_l"].iloc[sl], dtype=float)
        a_r = np.asarray(self.coordinateValues["ankle_angle_r"].iloc[sl], dtype=float)
        value = float(np.mean(0.5 * (a_l + a_r)))
        units = "deg"
        if return_all:
            return [value], units
        return value, units

    def compute_com_stability_normalized_velocity_3d(self, return_all=False):
        path_3d, _ = self._com_path_metrics()
        duration = float(
            self.toeStandEvents["endTime"] - self.toeStandEvents["startTime"]
        )
        if duration <= 0:
            raise ValueError("Non-positive segment duration.")
        if self.normalize_by_height:
            h = self._height_m_for_com_normalization()
            value = (path_3d / h) / duration
            units = "1/s"
        else:
            value = path_3d / duration
            units = "m/s"
        if return_all:
            return [value], units
        return value, units

    def compute_com_stability_normalized_velocity_horizontal(self, return_all=False):
        _, path_hz = self._com_path_metrics()
        duration = float(
            self.toeStandEvents["endTime"] - self.toeStandEvents["startTime"]
        )
        if duration <= 0:
            raise ValueError("Non-positive segment duration.")
        if self.normalize_by_height:
            h = self._height_m_for_com_normalization()
            value = (path_hz / h) / duration
            units = "1/s"
        else:
            value = path_hz / duration
            units = "m/s"
        if return_all:
            return [value], units
        return value, units
