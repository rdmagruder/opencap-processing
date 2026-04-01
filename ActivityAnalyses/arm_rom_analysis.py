"""
    ---------------------------------------------------------------------------
    OpenCap processing: arm_rom_analysis.py
    ---------------------------------------------------------------------------

    Arm range-of-motion analysis: reachable workspace over a segmented trial
    with multiple arm raises. Participants are expected to lower their arms
    after each 'arm raise cycle'.

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

from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from scipy import signal
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation as R

from utilsKinematics import kinematics


ARM_ROM_MARKER_KEYS = {
    "r_shoulder": "r_shoulder_study",
    "l_shoulder": "L_shoulder_study",
    "r_wrist": "r_mwrist_study",
    "l_wrist": "L_mwrist_study",
}

ARM_LENGTH_MARKERS = {
    "r": {
        "shoulder": "r_shoulder_study",
        "elbow_l": "r_lelbow_study",
        "elbow_m": "r_melbow_study",
        "wrist_l": "r_lwrist_study",
        "wrist_m": "r_mwrist_study",
    },
    "l": {
        "shoulder": "L_shoulder_study",
        "elbow_l": "L_lelbow_study",
        "elbow_m": "L_melbow_study",
        "wrist_l": "L_lwrist_study",
        "wrist_m": "L_mwrist_study",
    },
}


class arm_rom_analysis(kinematics):
    """Kinematics-only analysis for arm ROM: reachable workspace over one trial.

    Segmentation uses prominence peaks on mean wrist height, rest intervals from
    mean wrist height vs a rest band, and strict rest bracketing (drops start-high
    / end-high pseudo-cycles). Rest height is the mean of the flattest sliding
    window at trial start before the first prominent peak (not the global minimum).
    If multiple peaks fall in one raise interval, the rest band upper is raised
    globally by +5 cm then +10 cm (non-recursive); remaining multi-peak intervals
    use strongest-peak fallback. Trial bounds are first retained raise start
    through last retained raise end.
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

        self._arm_length_cache = None
        self.armRomEvents = self.segment_arm_rom()

    def _arm_length(self):
        """Arm length per side (upper arm + forearm from markers)."""
        if self._arm_length_cache is not None:
            return self._arm_length_cache

        markers = self.markerDict["markers"]
        lengths = {}
        for side, keys in ARM_LENGTH_MARKERS.items():
            for k in keys.values():
                if k not in markers:
                    raise ValueError(
                        f"Marker '{k}' not found for arm length. "
                        f"Available markers: {list(markers.keys())}"
                    )
            sh = markers[keys["shoulder"]]
            elb = (markers[keys["elbow_l"]] + markers[keys["elbow_m"]]) / 2.0
            wr = (markers[keys["wrist_l"]] + markers[keys["wrist_m"]]) / 2.0
            upper = np.mean(np.linalg.norm(elb - sh, axis=1))
            forearm = np.mean(np.linalg.norm(wr - elb, axis=1))
            lengths[side] = float(upper + forearm)

        self._arm_length_cache = lengths
        return lengths

    @staticmethod
    def _rest_height_flat_pre_first_peak(
        mean_y,
        peak_idxs,
        dt,
        min_rest_duration_s,
        flat_window_s=0.4,
    ):
        """Rest height = mean of the lowest-variance sliding window in the
        segment before the first prominent peak (start-of-trial rest plateau).
        Returns (rest_height, flat_region_start_idx, flat_region_end_idx inclusive).
        """
        min_rest_frames = int(np.ceil(min_rest_duration_s / dt)) if dt > 0 else 1
        win = max(
            min_rest_frames,
            int(np.ceil(flat_window_s / dt)) if dt > 0 else 5,
        )
        win = max(3, win)

        if len(peak_idxs) > 0:
            first_peak_idx = int(peak_idxs[0])
            pre = mean_y[:first_peak_idx]
        else:
            # No prominent peaks: estimate rest from an early trial window only.
            early_n = min(
                len(mean_y),
                max(win * 3, int(5.0 / dt) if dt > 0 else win * 3),
            )
            pre = mean_y[:early_n]

        if len(pre) == 0:
            rh = float(np.min(mean_y))
            return rh, None, None
        if len(pre) < win:
            rh = float(np.mean(pre))
            return rh, 0, len(pre) - 1

        best_i = 0
        best_var = np.inf
        for i in range(len(pre) - win + 1):
            seg = pre[i : i + win]
            v = float(np.var(seg))
            if v < best_var:
                best_var = v
                best_i = i
        rh = float(np.mean(pre[best_i : best_i + win]))
        return rh, best_i, best_i + win - 1

    def segment_arm_rom(
        self,
        visualize=False,
        rest_height_threshold_m=0.02,
        min_rest_duration_s=0.3,
        min_peak_prominence_m=0.06,
        min_peak_distance_s=0.5,
        flat_window_s=0.4,
    ):
        """Segment trial and raises: prominence peaks, rest bracketing, global
        rest-band widening (+5 cm, +10 cm) if needed, strongest-peak fallback.
        Rest height is the mean of the flattest region at trial start before the
        first prominent peak (not the global minimum).
        """
        markers = self.markerDict["markers"]
        for key in ARM_ROM_MARKER_KEYS.values():
            if key not in markers:
                raise ValueError(
                    f"Marker '{key}' not found in markerDict. "
                    f"Available markers: {list(markers.keys())}"
                )

        time_vec = self.markerDict["time"]
        dt = float(np.mean(np.diff(time_vec))) if len(time_vec) > 1 else 0.0

        r_wrist_y = markers[ARM_ROM_MARKER_KEYS["r_wrist"]][:, 1]
        l_wrist_y = markers[ARM_ROM_MARKER_KEYS["l_wrist"]][:, 1]
        mean_wrist_y = 0.5 * (r_wrist_y + l_wrist_y)

        min_rest_frames = int(np.ceil(min_rest_duration_s / dt)) if dt > 0 else 1

        min_peak_distance_frames = max(
            1,
            int(np.ceil(min_peak_distance_s / dt)) if dt > 0 else 1,
        )
        peak_idxs, peak_props = signal.find_peaks(
            mean_wrist_y,
            prominence=min_peak_prominence_m,
            distance=min_peak_distance_frames,
        )
        prominences = peak_props.get("prominences", np.array([]))

        rest_height, flat_r0, flat_r1 = self._rest_height_flat_pre_first_peak(
            mean_wrist_y,
            peak_idxs,
            dt,
            min_rest_duration_s,
            flat_window_s=flat_window_s,
        )
        rest_band_high_base = rest_height + rest_height_threshold_m

        def _get_rest_intervals(rest_band_high):
            mean_rest = mean_wrist_y <= rest_band_high
            intervals = []
            if len(mean_rest) > 0:
                in_rest = False
                start_local = 0
                for i, val in enumerate(mean_rest):
                    if val and not in_rest:
                        in_rest = True
                        start_local = i
                    elif not val and in_rest:
                        in_rest = False
                        intervals.append((start_local, i - 1))
                if in_rest:
                    intervals.append((start_local, len(mean_rest) - 1))
            return [
                (s, e) for (s, e) in intervals if (e - s + 1) >= min_rest_frames
            ]

        peak_prom_map = {
            int(idx): float(prominences[i]) if len(prominences) > i else np.nan
            for i, idx in enumerate(peak_idxs)
        }

        def _build_raises(rest_intervals):
            raises = []
            for peak_idx in peak_idxs:
                prev_rest_idx = None
                next_rest_idx = None
                for ir, (s, e) in enumerate(rest_intervals):
                    if e < peak_idx:
                        prev_rest_idx = ir
                    if s > peak_idx:
                        next_rest_idx = ir
                        break
                if prev_rest_idx is None or next_rest_idx is None:
                    continue
                left_rest = rest_intervals[prev_rest_idx]
                right_rest = rest_intervals[next_rest_idx]
                raise_start = int(left_rest[1])
                raise_end = int(right_rest[0])
                if raise_end <= raise_start:
                    continue
                raises.append(
                    {
                        "startIdx": raise_start,
                        "endIdx": raise_end,
                        "peakIdx": int(peak_idx),
                        "peakTime": float(time_vec[peak_idx]),
                        "prominence": peak_prom_map[int(peak_idx)],
                        "leftRestInterval": (int(left_rest[0]), int(left_rest[1])),
                        "rightRestInterval": (int(right_rest[0]), int(right_rest[1])),
                    }
                )
            return raises

        def _count_peaks_in_interval(start_idx, end_idx):
            return [int(p) for p in peak_idxs if start_idx <= int(p) <= end_idx]

        def _any_multi_peak(raises_list):
            by_interval = {}
            for r in raises_list:
                key = (r["startIdx"], r["endIdx"])
                by_interval.setdefault(key, []).append(r["peakIdx"])
            for peaks in by_interval.values():
                if len(set(peaks)) > 1:
                    return True
            return False

        attempt_summaries = []
        selected_rest_intervals = []
        selected_raises = []
        rest_band_high_used = rest_band_high_base

        for offset in (0.0, 0.05, 0.10, 0.15):
            rb = rest_band_high_base + offset
            rest_intervals = _get_rest_intervals(rb)
            raises = _build_raises(rest_intervals)
            n_multi = 0
            by_interval = {}
            for r in raises:
                key = (r["startIdx"], r["endIdx"])
                by_interval.setdefault(key, []).append(r["peakIdx"])
            for peaks in by_interval.values():
                if len(set(peaks)) > 1:
                    n_multi += 1
            attempt_summaries.append(
                {
                    "restBandHigh": float(rb),
                    "nRestIntervals": int(len(rest_intervals)),
                    "nRaises": int(len(raises)),
                    "nIntervalsWithMultiPeak": int(n_multi),
                }
            )
            selected_rest_intervals = rest_intervals
            selected_raises = raises
            rest_band_high_used = float(rb)
            if not _any_multi_peak(raises):
                break

        interval_groups = defaultdict(list)
        for r in selected_raises:
            interval_groups[(r["startIdx"], r["endIdx"])].append(r)

        fallback_applied = False
        final_raise_records = []
        for (local_start, local_end), group in sorted(
            interval_groups.items(), key=lambda item: item[0][0]
        ):
            peaks_here = _count_peaks_in_interval(local_start, local_end)
            r0 = group[0]
            if len(peaks_here) <= 1:
                final_raise_records.append(
                    {**r0, "fallbackStrongestPeakApplied": False}
                )
                continue

            fallback_applied = True
            best_peak = max(
                peaks_here,
                key=lambda p: (
                    peak_prom_map.get(int(p), -np.inf),
                    mean_wrist_y[int(p)],
                ),
            )
            best_peak = int(best_peak)
            left_slice = mean_wrist_y[local_start : best_peak + 1]
            right_slice = mean_wrist_y[best_peak : local_end + 1]
            left_valley = local_start + int(np.argmin(left_slice))
            right_valley = best_peak + int(np.argmin(right_slice))
            if right_valley <= left_valley:
                left_valley = local_start
                right_valley = local_end

            final_raise_records.append(
                {
                    **r0,
                    "startIdx": int(left_valley),
                    "endIdx": int(right_valley),
                    "peakIdx": best_peak,
                    "peakTime": float(time_vec[best_peak]),
                    "prominence": peak_prom_map.get(best_peak, np.nan),
                    "fallbackStrongestPeakApplied": True,
                    "intervalPeakCandidates": peaks_here,
                }
            )

        dedup = {}
        for r in final_raise_records:
            k = (r["startIdx"], r["endIdx"], r["peakIdx"])
            dedup[k] = r
        final_raise_records = sorted(dedup.values(), key=lambda x: x["startIdx"])

        raise_intervals = [(r["startIdx"], r["endIdx"]) for r in final_raise_records]

        if raise_intervals:
            start_idx = int(raise_intervals[0][0])
            end_idx = int(raise_intervals[-1][1])
        elif selected_rest_intervals:
            start_idx = int(selected_rest_intervals[0][0])
            end_idx = int(selected_rest_intervals[-1][1])
        else:
            start_idx = 0
            end_idx = len(time_vec) - 1

        start_time = float(time_vec[start_idx])
        end_time = float(time_vec[end_idx])

        arm_rom_events = {
            "startIdx": int(start_idx),
            "endIdx": int(end_idx),
            "startTime": start_time,
            "endTime": end_time,
            "restIntervals": selected_rest_intervals,
            "raiseIntervals": raise_intervals,
            "raisePeaks": final_raise_records,
            "allPeakIdx": [int(i) for i in peak_idxs],
            "allPeakProminences": [float(p) for p in prominences],
            "restHeight": float(rest_height),
            "restBandHighBase": float(rest_band_high_base),
            "restBandHighUsed": float(rest_band_high_used),
            "restBandAttemptSummaries": attempt_summaries,
            "restBandWideningAppliedCm": int(
                round((rest_band_high_used - rest_band_high_base) * 100)
            ),
            "strongestPeakFallbackApplied": bool(fallback_applied),
            "restHeightFlatRegionIdx": (
                (int(flat_r0), int(flat_r1))
                if flat_r0 is not None and flat_r1 is not None
                else None
            ),
            "restHeightFlatWindowS": float(flat_window_s),
        }

        if visualize:
            self._plot_arm_rom_segmentation(
                time_vec=time_vec,
                mean_wrist_y=mean_wrist_y,
                r_wrist_y=r_wrist_y,
                l_wrist_y=l_wrist_y,
                armRomEvents=arm_rom_events,
                rest_height=rest_height,
                rest_band_high=rest_band_high_used,
            )

        return arm_rom_events

    def _plot_arm_rom_segmentation(
        self,
        time_vec,
        mean_wrist_y,
        r_wrist_y,
        l_wrist_y,
        armRomEvents,
        rest_height,
        rest_band_high,
    ):
        plt.figure()
        plt.plot(time_vec, mean_wrist_y, "k-", label="Mean wrist height")
        plt.plot(time_vec, r_wrist_y, "--", alpha=0.6, label="Right wrist Y")
        plt.plot(time_vec, l_wrist_y, "--", alpha=0.6, label="Left wrist Y")
        plt.axhline(rest_height, color="gray", linestyle=":", label="Rest height")
        rh_flat = armRomEvents.get("restHeightFlatRegionIdx")
        if rh_flat is not None:
            fs, fe = rh_flat
            plt.axvspan(
                time_vec[fs],
                time_vec[fe],
                alpha=0.12,
                color="green",
                label="Rest height (flat region)",
            )
        plt.axhline(
            rest_band_high,
            color="gray",
            linestyle="--",
            label="Rest band upper (effective)",
        )
        plt.axvspan(
            armRomEvents["startTime"],
            armRomEvents["endTime"],
            alpha=0.4,
            color="lightgray",
            label="Segmented trial window",
        )
        for i, (s_idx, e_idx) in enumerate(armRomEvents["raiseIntervals"]):
            plt.axvspan(
                time_vec[s_idx],
                time_vec[e_idx],
                alpha=0.2,
                color="C1",
                label="Arm raise" if i == 0 else None,
            )
        if armRomEvents.get("allPeakIdx"):
            idxs = armRomEvents["allPeakIdx"]
            plt.plot(
                time_vec[idxs],
                mean_wrist_y[idxs],
                "ko",
                markersize=4,
                alpha=0.6,
                label="Prominent peaks",
            )
        if armRomEvents.get("raisePeaks"):
            kept_idxs = [d["peakIdx"] for d in armRomEvents["raisePeaks"]]
            plt.plot(
                time_vec[kept_idxs],
                mean_wrist_y[kept_idxs],
                "ro",
                markersize=5,
                alpha=0.9,
                label="Kept peaks",
            )
        plt.xlabel("Time [s]")
        plt.ylabel("Height [m]")
        plt.title("Arm ROM: wrist height and segmentation")
        plt.legend(loc="best", fontsize=8)
        plt.tight_layout()
        plt.show()

    def plot_arm_rom_wrist_height(self):
        markers = self.markerDict["markers"]
        r_wrist_y = markers[ARM_ROM_MARKER_KEYS["r_wrist"]][:, 1]
        l_wrist_y = markers[ARM_ROM_MARKER_KEYS["l_wrist"]][:, 1]
        mean_wrist_y = 0.5 * (r_wrist_y + l_wrist_y)
        rest_height = float(self.armRomEvents.get("restHeight", np.min(mean_wrist_y)))
        rest_band_high = float(
            self.armRomEvents.get("restBandHighUsed", rest_height + 0.02)
        )
        self._plot_arm_rom_segmentation(
            time_vec=self.markerDict["time"],
            mean_wrist_y=mean_wrist_y,
            r_wrist_y=r_wrist_y,
            l_wrist_y=l_wrist_y,
            armRomEvents=self.armRomEvents,
            rest_height=rest_height,
            rest_band_high=rest_band_high,
        )

    @staticmethod
    def _reachable_area_right(rs, ls, rw):
        rw_c = rw - rs
        ls_c = ls - rs
        roty = np.arctan2(ls_c[:, 2], ls_c[:, 0])
        norm_xz = np.linalg.norm(ls_c[:, [0, 2]], axis=1)
        norm_xz = np.where(norm_xz == 0, 1.0, norm_xz)
        rotz = np.arctan2(ls_c[:, 1], norm_xz)
        rw_rot = rw_c.copy()
        for i in range(rw_rot.shape[0]):
            rot = R.from_euler("yz", [roty[i], rotz[i]])
            rw_rot[i, :] = rot.apply(rw_rot[i, :])
        rw_rot[:, 0] = np.clip(rw_rot[:, 0], 0.0, None)
        unique_pts = np.unique(rw_rot, axis=0)
        if unique_pts.shape[0] < 4:
            return 0.0
        try:
            ch = ConvexHull(unique_pts)
            return float(ch.area)
        except Exception:
            return 0.0

    def _reachable_workspace_norms(self):
        """Per-side normalized reachable area and sum (right, left, total)."""
        markers = self.markerDict["markers"]
        for key in ARM_ROM_MARKER_KEYS.values():
            if key not in markers:
                raise ValueError(
                    f"Marker '{key}' not found in markerDict. "
                    f"Available markers: {list(markers.keys())}"
                )

        start = self.armRomEvents["startIdx"]
        end = self.armRomEvents["endIdx"] + 1

        rs = markers[ARM_ROM_MARKER_KEYS["r_shoulder"]][start:end, :]
        ls = markers[ARM_ROM_MARKER_KEYS["l_shoulder"]][start:end, :]
        rw = markers[ARM_ROM_MARKER_KEYS["r_wrist"]][start:end, :]
        lw = markers[ARM_ROM_MARKER_KEYS["l_wrist"]][start:end, :]

        rw_area_r = self._reachable_area_right(rs, ls, rw)

        ls_flip = ls.copy()
        rs_flip = rs.copy()
        lw_flip = lw.copy()
        ls_flip[:, 0] *= -1.0
        rs_flip[:, 0] *= -1.0
        lw_flip[:, 0] *= -1.0
        rw_area_l = self._reachable_area_right(ls_flip, rs_flip, lw_flip)

        lengths = self._arm_length()
        arm_len_r = lengths["r"]
        arm_len_l = lengths["l"]
        if arm_len_r <= 0.0 or arm_len_l <= 0.0:
            raise ValueError("Arm lengths must be positive to normalize workspace.")

        norm_r = rw_area_r / (arm_len_r ** 2)
        norm_l = rw_area_l / (arm_len_l ** 2)
        total = float(norm_r + norm_l)
        return float(norm_r), float(norm_l), total

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
                + " does not exist in arm_rom_analysis class."
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

    def compute_arm_rom_reachable_workspace(self, return_all=False):
        """Sum of left and right reachable workspace areas, each normalized by
        that side's arm_length squared.
        """
        _, _, value = self._reachable_workspace_norms()
        units = "normalized area (per arm_length^2)"
        if return_all:
            return [value], units
        return value, units

    def compute_arm_rom_reachable_workspace_right(self, return_all=False):
        """Right arm reachable workspace area normalized by arm_length squared."""
        value, _, _ = self._reachable_workspace_norms()
        units = "normalized area (per arm_length^2)"
        if return_all:
            return [value], units
        return value, units

    def compute_arm_rom_reachable_workspace_left(self, return_all=False):
        """Left arm reachable workspace area normalized by arm_length squared."""
        _, value, _ = self._reachable_workspace_norms()
        units = "normalized area (per arm_length^2)"
        if return_all:
            return [value], units
        return value, units

    def compute_arm_rom_trial_time(self, return_all=False):
        value = float(self.armRomEvents["endTime"] - self.armRomEvents["startTime"])
        units = "s"
        if return_all:
            return [value], units
        return value, units
