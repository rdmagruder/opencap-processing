"""
    ---------------------------------------------------------------------------
    OpenCap processing: tug_analysis.py
    ---------------------------------------------------------------------------

    Timed Up and Go (TUG) analysis based on center-of-mass (COM) and torso
    kinematics.

    Segmentation:
    - Primary segmentation is based on COM vertical motion (stand and sit).
    - Start of the trial is refined using torso forward-bend initiation.
    - End of the trial is the end of the COM descent back to sitting.

    Scalars:
    - Torso orientation at lift-off.
    - Maximum torso angular velocity prior to lift-off.
    - TUG time (torso initiation to final sitting).
"""

import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

from utilsKinematics import kinematics
from utilsProcessing import lowPassFilter


class tug_analysis(kinematics):
    """Kinematics-only analysis for the Timed Up and Go (TUG) activity."""

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

        # Trim marker data and coordinate values.
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

        # Heading-based rotation: x aligned with forward progression.
        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        # Body orientation and angular velocity in ground frame.
        self.body_angles = self.get_body_orientation(
            expressed_in="ground",
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )
        self.body_angular_velocity = self.get_body_angular_velocity(
            expressed_in="ground",
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values,
        )

        if self.trimming_start > 0:
            self.body_angles = self.body_angles.iloc[self.idx_trim_start :]
            self.body_angular_velocity = self.body_angular_velocity.iloc[
                self.idx_trim_start :
            ]
        if self.trimming_end > 0:
            self.body_angles = self.body_angles.iloc[: self.idx_trim_end]
            self.body_angular_velocity = self.body_angular_velocity.iloc[
                : self.idx_trim_end
            ]

        # Center-of-mass values (world frame) and rotated so x is forward.
        self._comValues = self.get_center_of_mass_values(
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values
        )
        self._comValuesRotated = self.rotate_com(
            self._comValues,
            {"y": self.rotation_about_y},
        )
        if self.trimming_start > 0:
            self._comValues = self._comValues.iloc[self.idx_trim_start :]
            self._comValuesRotated = self._comValuesRotated.iloc[self.idx_trim_start :]
        if self.trimming_end > 0:
            self._comValues = self._comValues.iloc[: self.idx_trim_end]
            self._comValuesRotated = self._comValuesRotated.iloc[: self.idx_trim_end]

        # Segment TUG (single main cycle).
        self.tugEvents = self.segment_tug()
        self.n_tug_cycles = 1

    def rotate_x_forward(self):
        """Rotate markers so that x is aligned with forward progression.

        Uses pelvis heading vector from PSIS to ASIS midpoints, averaged over
        the middle 50% of the trial, similar to gait and STS analyses.
        """
        psis_midpoint = (
            self.markerDict["markers"]["r.PSIS_study"]
            + self.markerDict["markers"]["L.PSIS_study"]
        ) / 2.0
        asis_midpoint = (
            self.markerDict["markers"]["r.ASIS_study"]
            + self.markerDict["markers"]["L.ASIS_study"]
        ) / 2.0

        heading_vector = asis_midpoint - psis_midpoint
        angle = np.unwrap(np.arctan2(heading_vector[:, 2], heading_vector[:, 0]))

        n_frames = len(self.markerDict["time"])
        start_index = int(n_frames * 0.25)
        end_index = int(n_frames * 0.75)
        angle = np.degrees(np.mean(angle[start_index:end_index], axis=0))

        marker_dict_rotated = self.rotate_marker_dict(self.markerDict, {"y": angle})

        return angle, marker_dict_rotated

    def segment_tug(
        self,
        visualize=False,
        com_height_threshold_m=0.03,
        torso_vel_threshold_rad_s=0.2,
        torso_search_window_s=2.0,
        settle_vel_threshold_m_s=0.05,
        settle_window_s=0.30,
    ):
        """Segment a single TUG trial using COM and torso orientation signals.

        Events:
        - torsoInitIdx / torsoInitTime: initiation of forward torso bending.
        - startRiseIdx / startRiseTime: onset of COM rise (chair rise).
        - sitIdx / sitTime: end of COM descent back to sitting.
        - peakForwardIdx / peakForwardTime: maximal forward COM position.
        """
        time_vec = self._comValuesRotated["time"].to_numpy()
        com_y = self._comValuesRotated["y"].to_numpy()
        com_x = self._comValuesRotated["x"].to_numpy()

        if len(time_vec) < 3:
            raise ValueError("TUG segmentation requires at least 3 frames.")

        dt = np.mean(np.diff(time_vec))

        # Smooth COM vertical trajectory for robust segmentation.
        com_y_smooth = lowPassFilter(
            time_vec,
            com_y,
            lowpass_cutoff_frequency=4,
        )
        com_y_vel = np.gradient(com_y_smooth, time_vec)
        com_y_vel_smooth = lowPassFilter(
            time_vec,
            com_y_vel,
            lowpass_cutoff_frequency=4,
        )

        # Baseline COM height from early portion of trial (assumed sitting).
        n_baseline = max(5, int(0.1 * len(com_y_smooth)))
        baseline_y = float(np.median(com_y_smooth[:n_baseline]))

        n_end = max(5, int(0.1 * len(com_y_smooth)))
        end_y = float(np.median(com_y_smooth[-n_end:]))

        # Identify main elevated phase of COM (standing and walking).
        amp_total = float(np.max(com_y_smooth) - baseline_y)
        if amp_total < com_height_threshold_m:
            raise ValueError(
                "COM vertical excursion is too small for TUG segmentation."
            )

        height_threshold = baseline_y + max(com_height_threshold_m, 0.2 * amp_total)
        above = com_y_smooth > height_threshold
        above_idx = np.where(above)[0]
        if above_idx.size == 0:
            raise ValueError("Could not detect a standing phase from COM trajectory.")

        # Main elevated phase (subject away from chair and standing).
        start_high = int(above_idx[0])
        end_high = int(above_idx[-1])

        # Onset of COM rise: walk backward from the start of elevated phase to
        # where COM is near baseline.
        start_rise_idx = start_high
        for i in range(start_high - 1, -1, -1):
            if com_y_smooth[i] <= baseline_y + 0.5 * (height_threshold - baseline_y):
                start_rise_idx = i
                break

        # Peak forward progression for reference.
        peak_forward_idx = int(np.argmax(com_x))

        search_start = peak_forward_idx
        search_end = len(com_y_smooth)

        # Find candidate descent phase after forward peak.
        downward = com_y_vel_smooth[search_start:search_end] < -settle_vel_threshold_m_s
        if np.any(downward):
            descent_start = search_start + int(np.where(downward)[0][0])
        else:
            descent_start = search_start

        # Look for the first sustained interval where:
        # 1) vertical velocity is near zero (settled), and
        # 2) COM is near the final seated level.
        settle_window_n = max(3, int(settle_window_s / dt))
        end_band = max(0.02, 0.15 * amp_total)  # tolerance around final seated level

        sit_idx = None
        for i in range(descent_start, len(com_y_smooth) - settle_window_n):
            y_window = com_y_smooth[i:i + settle_window_n]
            v_window = com_y_vel_smooth[i:i + settle_window_n]

            near_end_height = np.all(np.abs(y_window - end_y) < end_band)
            settled = np.all(np.abs(v_window) < settle_vel_threshold_m_s)

            if near_end_height and settled:
                # Choose the local minimum in this settled window.
                sit_idx = i + int(np.argmin(y_window))
                break

        if sit_idx is None:
            # Fallback 1: choose the minimum after descent begins.
            if descent_start < len(com_y_smooth) - 1:
                sit_idx = descent_start + int(np.argmin(com_y_smooth[descent_start:]))
            else:
                sit_idx = len(com_y_smooth) - 1

        # Torso forward-bend initiation: search backward from COM rise onset.
        torso_z_vel = self.body_angular_velocity["torso_z"].to_numpy() * -1.0
        max_window_frames = int(torso_search_window_s / dt)
        search_start = max(0, start_rise_idx - max_window_frames)
        search_indices = np.arange(search_start, start_rise_idx)
        if search_indices.size == 0:
            torso_init_idx = start_rise_idx
        else:
            seg_vel = torso_z_vel[search_indices]
            above_thresh = seg_vel > torso_vel_threshold_rad_s
            if np.any(above_thresh):
                torso_init_local = int(np.where(above_thresh)[0][0])
                torso_init_idx = int(search_indices[torso_init_local])
            else:
                # Default to COM rise onset if no torso initiation is detected.
                torso_init_idx = start_rise_idx

        tugEvents = {
            "torsoInitIdx": torso_init_idx,
            "torsoInitTime": float(time_vec[torso_init_idx]),
            "startRiseIdx": int(start_rise_idx),
            "startRiseTime": float(time_vec[start_rise_idx]),
            "sitIdx": int(sit_idx),
            "sitTime": float(time_vec[sit_idx]),
            "peakForwardIdx": int(peak_forward_idx),
            "peakForwardTime": float(time_vec[peak_forward_idx]),
        }

        if visualize:
            self._plot_tug_segmentation(time_vec, com_y_smooth, tugEvents)

        return tugEvents

    def _plot_tug_segmentation(self, time_vec, com_y, tugEvents):
        """Plot COM vertical trajectory with key TUG events."""
        plt.figure()
        plt.plot(time_vec, com_y, "k-", label="COM height (y)")

        t_start = tugEvents["torsoInitTime"]
        t_rise = tugEvents["startRiseTime"]
        t_sit = tugEvents["sitTime"]

        plt.axvline(t_start, color="b", linestyle="--", label="Trial start")
        plt.axvline(
            t_rise, color="g", linestyle="--", label="Chair rise onset"
        )
        plt.axvline(t_sit, color="r", linestyle="--", label="Seated")

        plt.xlabel("Time [s]")
        plt.ylabel("COM height [m]")
        plt.title("TUG: COM vertical trajectory and events")
        plt.legend(loc="best")
        plt.tight_layout()
        plt.show()

    def plot_tug_COM(self):
        """Public helper to visualize COM y and TUG events."""
        time_vec = self._comValuesRotated["time"].to_numpy()
        com_y = self._comValuesRotated["y"].to_numpy()
        com_y_smooth = lowPassFilter(time_vec, com_y, lowpass_cutoff_frequency=4)
        self._plot_tug_segmentation(time_vec, com_y_smooth, self.tugEvents)

    def compute_scalars(self, scalarNames, return_all=False):
        """Compute scalar metrics for the TUG activity.

        Parameters
        ----------
        scalarNames : list of str
            Names without the 'compute_' prefix, e.g.
            ['torso_orientation_liftoff', 'torso_angular_velocity', 'tug_time'].
        return_all : bool, optional
            If True, return per-cycle values (here a list with one entry),
            otherwise return the average value.
        """
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
                + " does not exist in tug_analysis class."
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

    def compute_torso_orientation_liftoff(self, return_all=False):
        """Torso orientation at lift-off (start of COM rise), in degrees."""
        torso_z = self.body_angles["torso_z"].to_numpy() * -1.0
        idx = self.tugEvents["startRiseIdx"]
        vals = [np.degrees(torso_z[idx])]
        units = "degrees"
        if return_all:
            return vals, units
        return float(np.mean(vals)), units

    def compute_torso_angular_velocity(self, return_all=False):
        """Maximum torso angular velocity prior to lift-off, in degrees/s."""
        torso_z_vel = self.body_angular_velocity["torso_z"].to_numpy() * -1.0
        idx_start = self.tugEvents["torsoInitIdx"]
        idx_end = self.tugEvents["startRiseIdx"]
        if idx_end <= idx_start:
            vals = [0.0]
        else:
            vals = [np.max(torso_z_vel[idx_start:idx_end])]
        vals_deg = [np.degrees(v) for v in vals]
        units = "degrees/s"
        if return_all:
            return vals_deg, units
        return float(np.mean(vals_deg)), units

    def compute_tug_time(self, return_all=False):
        """TUG time: torso initiation to final sitting, in seconds."""
        time_vec = self._comValuesRotated["time"].to_numpy()
        idx_start = self.tugEvents["torsoInitIdx"]
        idx_end = self.tugEvents["sitIdx"]
        vals = [float(time_vec[idx_end] - time_vec[idx_start])]
        units = "s"
        if return_all:
            return vals, units
        return float(np.mean(vals)), units

