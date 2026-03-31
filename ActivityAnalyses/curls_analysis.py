"""
    ---------------------------------------------------------------------------
    OpenCap processing: curls_analysis.py
    ---------------------------------------------------------------------------

    First-phase biceps curl segmentation using arm flexion coordinates.
    There is an expected second-phase for wrist RoM, however with the current
    model, the wrists are welded, and thus this portion is ignored.

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
from scipy import signal

from utilsKinematics import kinematics


class curls_analysis(kinematics):
    """Kinematics-only analysis for first-phase curls.

    Segmentation uses bilateral arm flexion (`arm_flex_r`, `arm_flex_l`), where
    0 deg is full extension and curl flexion increases toward shoulder touch
    (capped near 150 deg). The rep is defined by the first prominent flexion
    peak and left/right flattening-rest boundaries around that peak.
    """

    def __init__(
        self,
        session_dir,
        trial_name,
        lowpass_cutoff_frequency_for_coordinate_values=6,
        trimming_start=0,
        trimming_end=0,
    ):
        super().__init__(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values,
        )

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end

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
                    <= np.round(self.markerDict["time"][-1] - self.trimming_end, 6)
                )[0][-1]
                + 1
            )
            self.markerDict["time"] = self.markerDict["time"][: self.idx_trim_end]
            for marker in self.markerDict["markers"]:
                self.markerDict["markers"][marker] = self.markerDict["markers"][marker][
                    : self.idx_trim_end, :
                ]
            self.coordinateValues = self.coordinateValues.iloc[: self.idx_trim_end]

        self.curlsEvents = self.segment_curls()

    def _get_flexion_coordinates(self):
        """Return right/left flexion arrays and coordinate source label."""
        cols = self.coordinateValues.columns
        if "arm_flex_r" in cols and "arm_flex_l" in cols:
            return (
                self.coordinateValues["arm_flex_r"].to_numpy(dtype=float),
                self.coordinateValues["arm_flex_l"].to_numpy(dtype=float),
                "arm_flex",
            )
        if "elbow_flex_r" in cols and "elbow_flex_l" in cols:
            return (
                self.coordinateValues["elbow_flex_r"].to_numpy(dtype=float),
                self.coordinateValues["elbow_flex_l"].to_numpy(dtype=float),
                "elbow_flex",
            )

        raise ValueError(
            "Could not find a complete bilateral flexion pair. Expected either "
            "('arm_flex_r', 'arm_flex_l') or ('elbow_flex_r', 'elbow_flex_l')."
        )

    @staticmethod
    def _rest_flex_flat_pre_first_peak(
        flex_signal,
        first_peak_idx,
        dt,
        min_rest_duration_s=0.3,
        flat_window_s=0.4,
        max_baseline_flexion_deg=50.0,
    ):
        """Estimate rest flexion from flattest pre-peak window."""
        min_rest_frames = int(np.ceil(min_rest_duration_s / dt)) if dt > 0 else 1
        win = max(min_rest_frames, int(np.ceil(flat_window_s / dt)) if dt > 0 else 5)
        win = max(3, win)

        pre = flex_signal[: max(first_peak_idx, 1)]
        if len(pre) == 0:
            return float(np.min(flex_signal)), None, None
        if len(pre) < win:
            below = np.where(pre < max_baseline_flexion_deg)[0]
            if len(below) > 0:
                return float(np.mean(pre[below])), int(below[0]), int(below[-1])
            raise ValueError(
                f"No pre-peak samples below {max_baseline_flexion_deg} deg were found "
                "for baseline estimation."
            )

        best_i = 0
        best_var = np.inf
        found_valid_window = False
        for i in range(len(pre) - win + 1):
            seg = pre[i : i + win]
            # Baseline section must remain below the 50 deg constraint.
            if np.any(seg >= max_baseline_flexion_deg):
                continue
            v = float(np.var(seg))
            if v < best_var:
                best_var = v
                best_i = i
                found_valid_window = True

        if found_valid_window:
            rest_flex = float(np.mean(pre[best_i : best_i + win]))
            return rest_flex, best_i, best_i + win - 1

        below = np.where(pre < max_baseline_flexion_deg)[0]
        if len(below) > 0:
            rest_flex = float(np.mean(pre[below]))
            return rest_flex, int(below[0]), int(below[-1])

        raise ValueError(
            f"No pre-peak samples below {max_baseline_flexion_deg} deg were found "
            "for baseline estimation."
        )

    @staticmethod
    def _find_rest_crossing_with_duration(
        flex_signal,
        peak_idx,
        rest_threshold,
        min_rest_frames,
        direction="left",
    ):
        """Find boundary closest to peak where a sustained rest run exists."""
        n = len(flex_signal)
        if direction == "left":
            for i in range(peak_idx, min_rest_frames - 2, -1):
                run_start = i - min_rest_frames + 1
                seg = flex_signal[run_start : i + 1]
                if np.all(seg <= rest_threshold):
                    return i
            return 0

        for i in range(peak_idx, n - min_rest_frames + 1):
            seg = flex_signal[i : i + min_rest_frames]
            if np.all(seg <= rest_threshold):
                return i
        return n - 1

    def segment_curls(
        self,
        min_peak_prominence_deg=15.0,
        min_peak_distance_s=0.5,
        min_peak_flexion_deg=25.0,
        rest_threshold_deg=5.0,
        min_rest_duration_s=0.25,
        flat_window_s=0.4,
        visualize=False,
    ):
        """Segment first-phase curl from bilateral arm flexion.

        Signal conventions:
        - 0 deg: full extension (resting arms by side)
        - up to ~150 deg: peak flexion during shoulder-touch attempt
        """
        time_vec = self.markerDict["time"]
        dt = float(np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 0.0

        flex_r, flex_l, coord_source = self._get_flexion_coordinates()

        flex_signal = np.nanmean(np.vstack([flex_r, flex_l]), axis=0)
        n_frames = len(flex_signal)

        kernel_size = 5 if n_frames >= 5 else 3
        if kernel_size > n_frames:
            flex_smooth = flex_signal.copy()
        else:
            flex_smooth = signal.medfilt(flex_signal, kernel_size=kernel_size)

        min_peak_distance_frames = max(
            1, int(np.ceil(min_peak_distance_s / dt)) if dt > 0 else 1
        )
        peak_idxs, peak_props = signal.find_peaks(
            flex_smooth,
            prominence=min_peak_prominence_deg,
            distance=min_peak_distance_frames,
            height=min_peak_flexion_deg,
        )
        prominences = peak_props.get("prominences", np.array([]))

        if len(peak_idxs) > 0:
            # Keep only the largest prominent peaks, then take the first in time.
            prom_cutoff = max(min_peak_prominence_deg, 0.5 * float(np.max(prominences)))
            keep = prominences >= prom_cutoff
            candidate_peaks = peak_idxs[keep] if np.any(keep) else peak_idxs
            first_peak_idx = int(np.min(candidate_peaks))
        else:
            first_peak_idx = int(np.argmax(flex_smooth))

        rest_flex, rest_i0, rest_i1 = self._rest_flex_flat_pre_first_peak(
            flex_smooth,
            first_peak_idx,
            dt,
            min_rest_duration_s=min_rest_duration_s,
            flat_window_s=flat_window_s,
        )
        threshold = rest_flex + rest_threshold_deg
        min_rest_frames = int(np.ceil(min_rest_duration_s / dt)) if dt > 0 else 1
        min_rest_frames = max(1, min_rest_frames)

        start_idx = int(
            self._find_rest_crossing_with_duration(
                flex_smooth,
                first_peak_idx,
                threshold,
                min_rest_frames,
                direction="left",
            )
        )
        end_idx = int(
            self._find_rest_crossing_with_duration(
                flex_smooth,
                first_peak_idx,
                threshold,
                min_rest_frames,
                direction="right",
            )
        )

        curls_events = {
            "startIdx": start_idx,
            "firstCurlPeakIdx": int(first_peak_idx),
            "endIdx": end_idx,
            "startTime": float(time_vec[start_idx]),
            "firstCurlPeakTime": float(time_vec[first_peak_idx]),
            "endTime": float(time_vec[end_idx]),
            "allPeakIdx": [int(i) for i in peak_idxs],
            "allPeakProminences": [float(p) for p in prominences],
            "leftRestIdx": int(start_idx),
            "rightRestIdx": int(end_idx),
            "restFlexionDeg": float(rest_flex),
            "restThresholdDeg": float(threshold),
            "restFlexionFlatRegionIdx": (
                (int(rest_i0), int(rest_i1))
                if rest_i0 is not None and rest_i1 is not None
                else None
            ),
            "flexionCoordinateSource": coord_source,
        }

        if visualize:
            self._plot_curls_segmentation(
                time_vec=time_vec,
                flex_r=flex_r,
                flex_l=flex_l,
                flex_signal=flex_smooth,
                curls_events=curls_events,
            )

        return curls_events

    def _plot_curls_segmentation(
        self,
        time_vec,
        flex_r,
        flex_l,
        flex_signal,
        curls_events,
    ):
        source = curls_events.get("flexionCoordinateSource", "arm_flex")
        right_label = "arm_flex_r" if source == "arm_flex" else "elbow_flex_r"
        left_label = "arm_flex_l" if source == "arm_flex" else "elbow_flex_l"

        plt.figure()
        plt.plot(time_vec, flex_r, "--", alpha=0.5, label=right_label)
        plt.plot(time_vec, flex_l, "--", alpha=0.5, label=left_label)
        plt.plot(time_vec, flex_signal, "k-", label="Mean flexion (smoothed)")

        start_idx = curls_events["startIdx"]
        peak_idx = curls_events["firstCurlPeakIdx"]
        end_idx = curls_events["endIdx"]

        plt.axvspan(
            time_vec[start_idx],
            time_vec[end_idx],
            alpha=0.4,
            color="lightgray",
            label="First curl segment",
        )
        plt.plot(
            time_vec[peak_idx], flex_signal[peak_idx], "ro", markersize=6, label="First peak"
        )
        plt.axhline(
            curls_events["restFlexionDeg"], color="gray", linestyle=":", label="Rest flexion"
        )
        plt.axhline(
            curls_events["restThresholdDeg"],
            color="gray",
            linestyle="--",
            label="Rest threshold",
        )
        plt.xlabel("Time [s]")
        plt.ylabel("Arm flexion [deg]")
        plt.title("Curls: first prominent peak segmentation")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_curls_flexion(self):
        flex_r, flex_l, _ = self._get_flexion_coordinates()
        flex_signal = 0.5 * (flex_r + flex_l)
        self._plot_curls_segmentation(
            time_vec=self.markerDict["time"],
            flex_r=flex_r,
            flex_l=flex_l,
            flex_signal=flex_signal,
            curls_events=self.curlsEvents,
        )

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
                + " does not exist in curls_analysis class."
            )

        scalarDict = {}
        for scalarName in scalarNames:
            fn = getattr(self, "compute_" + scalarName)
            scalarDict[scalarName] = {}
            (
                scalarDict[scalarName]["value"],
                scalarDict[scalarName]["units"],
            ) = fn(return_all=return_all)
        return scalarDict

    def compute_curl_peak_flexion(self, return_all=False):
        peak_idx = self.curlsEvents["firstCurlPeakIdx"]
        flex_r, flex_l, _ = self._get_flexion_coordinates()
        value = float(0.5 * (flex_r[peak_idx] + flex_l[peak_idx]))
        units = "deg"
        if return_all:
            return [value], units
        return value, units

    def compute_curl_excursion(self, return_all=False):
        peak_flexion = self.compute_curl_peak_flexion()[0]
        flex_r, flex_l, _ = self._get_flexion_coordinates()
        rest_mean = 0.5 * (
            flex_r[self.curlsEvents["startIdx"]]
            + flex_l[self.curlsEvents["startIdx"]]
        )
        value = float(peak_flexion - rest_mean)
        units = "deg"
        if return_all:
            return [value], units
        return value, units

    def compute_curl_duration(self, return_all=False):
        value = float(self.curlsEvents["endTime"] - self.curlsEvents["startTime"])
        units = "s"
        if return_all:
            return [value], units
        return value, units
