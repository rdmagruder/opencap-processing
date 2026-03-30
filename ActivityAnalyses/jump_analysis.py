"""
---------------------------------------------------------------------------
OpenCap processing: jump_analysis.py
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
from utilsProcessing import lowPassFilter


class jump_analysis(kinematics):
    """Kinematics-only analysis for a countermovement jump (CMJ).

    Segmentation (single trial):
    - Use COM (vertical y) to find:
      - initialization of the downward movement (segment start)
      - when the participant returns to baseline normalcy (segment end)
    - Key events within the segment:
      - bottomIdx (lowest COM)
      - flightStartIdx/flightEndIdx (baseline-crossings, if substantial flight is present)

    Scalars (single cycle):
    - rise_time: time from COM bottom to rise initialization
    - flight_time: time between flightStart and flightEnd; 0 for missing-flight trials
    - jump_height: baseline-adjusted max COM during the flight window (or during the segment for missing-flight)
    - max_com_vel_rise: signed max COM vertical velocity during the rise window
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

        # Marker data load and filter.
        self.markerDict = self.get_marker_dict(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )

        # Coordinate values.
        self.coordinateValues = self.get_coordinate_values()

        # Optional trimming (matches tug_analysis and brooke_analysis patterns).
        if self.trimming_start > 0:
            self.idx_trim_start = np.where(
                np.round(self.markerDict["time"] - self.trimming_start, 6) <= 0
            )[0][-1]
            self.markerDict["time"] = self.markerDict["time"][self.idx_trim_start:]
            for marker in self.markerDict["markers"]:
                self.markerDict["markers"][marker] = self.markerDict["markers"][marker][
                    self.idx_trim_start:, :
                ]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = (
                np.where(
                    np.round(self.markerDict["time"], 6)
                    <= np.round(
                        self.markerDict["time"][-1] - self.trimming_end, 6
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

        # Center-of-mass values (world frame COM y).
        self._comValues = self.get_center_of_mass_values(
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values
        )
        if self.trimming_start > 0:
            self._comValues = self._comValues.iloc[self.idx_trim_start:]
        if self.trimming_end > 0:
            self._comValues = self._comValues.iloc[: self.idx_trim_end]

        # Segment a single jump cycle.
        self.jumpEvents = self.segment_jump()
        self.n_jump_cycles = 1

    def segment_jump(
        self,
        visualize=False,
        baseline_window_s=0.2,
        com_smoothing_cutoff_frequency_hz=4,
        min_down_threshold_m=0.01,
        down_threshold_fraction_of_drop=0.1,
        min_settle_band_m=0.01,
        settle_band_fraction_of_drop=0.05,
        settle_window_s=0.15,
        settle_vel_threshold_m_s=0.05,
        min_substantial_rise_m=0.02,
        flight_up_cross_tol_m=0.002,
        flight_down_cross_tol_m=0.002,
        rise_up_cross_tol_m=0.001,
        missing_flight_vel_sanity_threshold_m_s=0.1,
    ):
        """Segment a single countermovement jump using COM y.

        The logic follows the requested rules:
        1) Detect whether a successful jump happened by checking if the COM peak is
           substantially above the baseline.
        2) Use that peak as the "flight peak" and backtrack to find the lowest COM
           point, with bounce handling: if multiple troughs exist, choose the first
           trough that is within 5cm of the lowest pre-peak trough.
        3) Define flight start/end using baseline crossings.
        4) For missing-flight trials, define rise initialization as the first sustained
           upward COM motion after the chosen bottom.
        """
        time_vec = self._comValues["time"].to_numpy()
        com_y_raw = self._comValues["y"].to_numpy()

        if len(time_vec) < 5:
            raise ValueError("Jump segmentation requires at least 5 frames.")

        dt = float(np.mean(np.diff(time_vec)))
        if dt <= 0:
            raise ValueError("Invalid time step dt.")

        # Smooth COM y for robust thresholding.
        com_y = lowPassFilter(
            time_vec,
            com_y_raw,
            lowpass_cutoff_frequency=com_smoothing_cutoff_frequency_hz,
        )

        com_vel_raw = np.gradient(com_y, time_vec)
        com_vel_smooth = lowPassFilter(
            time_vec,
            com_vel_raw,
            lowpass_cutoff_frequency=min(2.0, com_smoothing_cutoff_frequency_hz),
        )

        # Baseline (pre-down) from an initial window assumed near standing normalcy.
        baseline_frames = max(5, int(round(baseline_window_s / dt)))
        baseline_frames = min(baseline_frames, max(5, len(time_vec) // 10))
        baseline_y = float(np.median(com_y[:baseline_frames]))

        # Peak-above-baseline test: check if any substantial jump occurred.
        peak_rel = int(np.argmax(com_y[baseline_frames:]))
        peak_idx = baseline_frames + peak_rel
        peak_y = float(com_y[peak_idx])

        # Downward amplitude for adaptive thresholds.
        trough_rel = int(np.argmin(com_y[baseline_frames:]))
        trough_idx = baseline_frames + trough_rel
        trough_y = float(com_y[trough_idx])

        drop_amplitude = baseline_y - trough_y
        if drop_amplitude <= 0:
            raise ValueError(
                "Could not detect a meaningful downward COM phase (baseline <= trough)."
            )

        substantial_peak_threshold_m = max(min_substantial_rise_m, 0.1 * drop_amplitude)
        jump_exists = (peak_y - baseline_y) >= substantial_peak_threshold_m

        # Backtrack: use the peak as the "flight peak" and find the lowest trough prior to it.
        # Bounce handling: choose the first trough within 5cm of the lowest trough.
        peak_end_idx = max(peak_idx, baseline_frames + 1)
        trough_search = slice(baseline_frames, peak_end_idx + 1)
        lowest_prepeak_rel = int(np.argmin(com_y[trough_search]))
        lowest_prepeak_idx = baseline_frames + lowest_prepeak_rel
        lowest_prepeak_y = float(com_y[lowest_prepeak_idx])

        # Candidate troughs are local minima; select earliest trough within +5cm of the lowest one.
        tol_from_lowest_m = 0.05
        first_trough_idx = None
        for i in range(baseline_frames + 1, peak_end_idx):
            if com_y[i] <= com_y[i - 1] and com_y[i] <= com_y[i + 1]:
                if com_y[i] <= lowest_prepeak_y + tol_from_lowest_m:
                    first_trough_idx = i
                    break
        if first_trough_idx is None:
            first_trough_idx = lowest_prepeak_idx

        bottom_idx = int(first_trough_idx)
        bottom_time = float(time_vec[bottom_idx])
        bottom_y = float(com_y[bottom_idx])

        # Segment start/end:
        # - Start: walk backward from the bottom until COM is within 5cm of baseline
        #   for a sustained window (position-only).
        # - End: walk forward until COM returns to baseline and stays there for a
        #   sustained window (position-only).
        level_tol_m = 0.05
        settle_window_n = max(3, int(round(settle_window_s / dt)))

        segment_start_idx = int(baseline_frames)
        for i in range(max(0, bottom_idx - settle_window_n), -1, -1):
            j_end = i + settle_window_n
            if j_end > bottom_idx + 1:
                continue
            y_win = com_y[i:j_end]
            near_baseline = np.all(np.abs(y_win - baseline_y) <= level_tol_m)
            if near_baseline:
                segment_start_idx = int(i)
                break

        # Segment end scan should start from the lowest trough after the jump peak.
        if jump_exists:
            landing_trough_idx = int(np.argmin(com_y[peak_idx:]) + peak_idx)
        else:
            landing_trough_idx = int(bottom_idx)

        segment_end_idx = len(time_vec) - 1
        start_scan = int(landing_trough_idx)
        for i in range(start_scan, len(time_vec) - settle_window_n):
            y_win = com_y[i : i + settle_window_n]
            near_baseline = np.all(np.abs(y_win - baseline_y) <= level_tol_m)
            if near_baseline:
                segment_end_idx = int(i + settle_window_n - 1)
                break

        if segment_end_idx <= segment_start_idx + 2:
            # Fallback: coarse window around the bottom.
            segment_start_idx = max(0, bottom_idx - int(0.5 / dt))
            segment_end_idx = min(len(time_vec) - 1, bottom_idx + int(0.8 / dt))

        # Flight start/end based on baseline crossings after the chosen bottom.
        flightStartIdx = None
        flightEndIdx = None
        flightMissing = True

        flight_time = 0.0
        jump_height = 0.0
        flightStartTime = None
        flightEndTime = None

        if jump_exists:
            # Flight start: first time COM crosses above baseline.
            up_level = baseline_y + flight_up_cross_tol_m
            for i in range(bottom_idx + 1, segment_end_idx + 1):
                if com_y[i] >= up_level and com_y[i - 1] < up_level:
                    flightStartIdx = int(i)
                    break

            # Flight end: first time COM crosses below baseline after being above.
            if flightStartIdx is not None:
                down_level = baseline_y - flight_down_cross_tol_m
                for i in range(flightStartIdx + 1, segment_end_idx + 1):
                    if com_y[i] <= down_level and com_y[i - 1] > down_level:
                        flightEndIdx = int(i)
                        break

            if flightStartIdx is not None and flightEndIdx is not None and flightEndIdx > flightStartIdx:
                flightMissing = False

        if not flightMissing:
            flightStartTime = float(time_vec[flightStartIdx])
            flightEndTime = float(time_vec[flightEndIdx])

            flight_time = float(flightEndTime - flightStartTime)
            flight_window = slice(flightStartIdx, flightEndIdx + 1)
            max_com_during_flight = float(np.max(com_y[flight_window]))
            jump_height = max(0.0, max_com_during_flight - baseline_y)
        else:
            # Missing flight:
            # COM bottom is the start of rising.
            flightStartIdx = int(bottom_idx)
            flightStartTime = float(bottom_time)
            flightEndIdx = None
            flightEndTime = None
            flight_time = 0.0
            seg_window = slice(bottom_idx, segment_end_idx + 1)
            jump_height = max(0.0, float(np.max(com_y[seg_window]) - baseline_y))

        segment_start_time = float(time_vec[segment_start_idx])
        segment_end_time = float(time_vec[segment_end_idx])

        jumpEvents = {
            "segmentStartIdx": int(segment_start_idx),
            "segmentStartTime": segment_start_time,
            "segmentEndIdx": int(segment_end_idx),
            "segmentEndTime": segment_end_time,
            "downInitIdx": int(segment_start_idx),
            "downInitTime": segment_start_time,
            "bottomIdx": int(bottom_idx),
            "bottomTime": bottom_time,
            "baselineY": baseline_y,
            "flightMissing": bool(flightMissing),
            "flightStartIdx": -1 if flightStartIdx is None else int(flightStartIdx),
            "flightStartTime": -1.0 if flightStartIdx is None else float(flightStartTime),
            "flightEndIdx": -1 if flightEndIdx is None else int(flightEndIdx),
            "flightEndTime": -1.0 if flightEndIdx is None else float(flightEndTime),
            "flightTime": float(flight_time),
            "jumpHeight": float(jump_height),
        }

        if visualize:
            self._plot_jump_segmentation(
                time_vec=time_vec,
                com_y=com_y,
                com_vel=com_vel_smooth,
                events=jumpEvents,
            )

        return jumpEvents

    def _plot_jump_segmentation(self, time_vec, com_y, com_vel, events):
        """Plot COM trace and annotated events for debugging/visual inspection."""
        plt.figure()
        plt.plot(time_vec, com_y, "k-", label="COM y (smoothed)")
        plt.axhline(events["baselineY"], color="gray", linestyle="--", linewidth=1, label="Baseline COM y")

        # Segment span.
        plt.axvspan(
            events["segmentStartTime"],
            events["segmentEndTime"],
            alpha=0.15,
            color="gray",
            label="Segment",
        )


        if not events["flightMissing"]:
            # Bottom.
            plt.plot(
                events["bottomTime"],
                com_y[events["bottomIdx"]],
                "bo",
                markersize=7,
                label="COM bottom",
            )

            fs_idx = events["flightStartIdx"]
            fe_idx = events["flightEndIdx"]
            plt.plot(
                events["flightStartTime"],
                com_y[fs_idx],
                "r^",
                markersize=7,
                label="Flight start",
            )
            plt.plot(
                events["flightEndTime"],
                com_y[fe_idx],
                "rv",
                markersize=7,
                label="Flight end",
            )
        else:
            # Flight start / rise start.
            plt.plot(
                events["flightStartTime"],
                com_y[events["flightStartIdx"]],
                "go",
                markersize=7,
                label="Rise start",
            )

        plt.xlabel("Time [s]")
        plt.ylabel("COM height [m]")
        plt.title("Countermovement jump: COM segmentation")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    def compute_rise_time(self, return_all=False):
        """Rise time: time from COM bottom to flight start.

        For successful jumps: bottom -> flight start (COM crosses above baseline).
        For missing-flight trials: flightStart is set to the COM bottom (so rise_time = 0).
        """
        value = float(self.jumpEvents["flightStartTime"] - self.jumpEvents["bottomTime"])
        units = "s"
        if return_all:
            return [value], units
        return value, units

    def compute_flight_time(self, return_all=False):
        """Flight time: duration between baseline-crossings while airborne.

        Returns 0.0 for missing-flight trials.
        """
        value = float(self.jumpEvents["flightTime"])
        units = "s"
        if return_all:
            return [value], units
        return value, units

    def compute_jump_height(self, return_all=False):
        """Jump height: baseline-adjusted max COM during flight (or segment for missing-flight)."""
        value = float(self.jumpEvents["jumpHeight"])
        units = "m"
        if return_all:
            return [value], units
        return value, units

    def compute_max_com_vel_rise(self, return_all=False):
        """Max COM vertical velocity during the rise portion of the jump.

        Window rules:
        - Successful jumps: from COM bottom (bottomIdx) to flight start (flightStartIdx, inclusive).
        - No-airtime trials: from rise start (flightStartIdx; set to bottomIdx by segmentation) to
          the end of the segmented trial (segmentEndIdx, inclusive).
        """
        time_vec = self._comValues["time"].to_numpy()
        com_y_raw = self._comValues["y"].to_numpy()

        # Match the segmentation approach: smooth COM y, then compute and smooth velocity.
        com_smoothing_cutoff_frequency_hz = 4
        com_y = lowPassFilter(
            time_vec,
            com_y_raw,
            lowpass_cutoff_frequency=com_smoothing_cutoff_frequency_hz,
        )

        com_vel_raw = np.gradient(com_y, time_vec)
        com_vel_smooth = lowPassFilter(
            time_vec,
            com_vel_raw,
            lowpass_cutoff_frequency=min(2.0, com_smoothing_cutoff_frequency_hz),
        )

        flight_missing = bool(self.jumpEvents["flightMissing"])
        if not flight_missing:
            start_idx = int(self.jumpEvents["bottomIdx"])
            end_idx = int(self.jumpEvents["flightStartIdx"])
        else:
            start_idx = int(self.jumpEvents["flightStartIdx"])
            end_idx = int(self.jumpEvents["segmentEndIdx"])

        if start_idx < 0 or end_idx < 0:
            raise ValueError(
                f"Invalid rise window indices: start_idx={start_idx}, end_idx={end_idx}."
            )
        if end_idx < start_idx:
            raise ValueError(
                f"Invalid rise window: end_idx ({end_idx}) < start_idx ({start_idx})."
            )
        if end_idx >= len(com_vel_smooth) or start_idx >= len(com_vel_smooth):
            raise ValueError(
                "Rise window indices are out of bounds for COM velocity array."
            )

        vel_slice = com_vel_smooth[start_idx : end_idx + 1]
        if vel_slice.size == 0:
            raise ValueError("Rise velocity slice is empty.")

        value = float(np.max(vel_slice))
        units = "m/s"
        if return_all:
            return [value], units
        return value, units

    def compute_scalars(self, scalarNames, return_all=False):
        """Compute scalar metrics for countermovement jumps."""
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
                + " does not exist in jump_analysis class."
            )

        scalarDict = {}
        for scalarName in scalarNames:
            fn = getattr(self, "compute_" + scalarName)
            scalarDict[scalarName] = {}
            (scalarDict[scalarName]["value"], scalarDict[scalarName]["units"]) = fn(
                return_all=return_all
            )

        return scalarDict

