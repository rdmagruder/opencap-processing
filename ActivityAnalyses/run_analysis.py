"""
    ---------------------------------------------------------------------------
    OpenCap processing: run_analysis.py
    ---------------------------------------------------------------------------

    Copyright 2023 Stanford University and the Authors

    Licensed under the Apache License, Version 2.0 (the "License"); you may not
    use this file except in compliance with the License. You may obtain a copy
    of the License at http://www.apache.org/licenses/LICENSE-2.0

    Unless required by applicable law or agreed to in writing, software
    distributed under the License is distributed on an "AS IS" BASIS,
    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
    See the License for the specific language governing permissions and
    limitations under the License.
"""

import sys
sys.path.append('../')

import numpy as np
from scipy.signal import find_peaks

from gait_analysis import gait_analysis


class run_analysis(gait_analysis):
    def __init__(self, session_dir, trial_name, leg='auto',
                 lowpass_cutoff_frequency_for_coordinate_values=-1,
                 n_run_cycles=-1, run_style='auto', trimming_start=0,
                 trimming_end=0):

        super(gait_analysis, self).__init__(
            session_dir,
            trial_name,
            lowpass_cutoff_frequency_for_coordinate_values=lowpass_cutoff_frequency_for_coordinate_values,
            modelName='LaiUhlrich2022_scaled',)

        self.trimming_start = trimming_start
        self.trimming_end = trimming_end

        self.markerDict = self.get_marker_dict(session_dir, trial_name,
            lowpass_cutoff_frequency=lowpass_cutoff_frequency_for_coordinate_values)

        self.coordinateValues = self.get_coordinate_values()

        if self.trimming_start > 0:
            self.idx_trim_start = np.where(np.round(self.markerDict['time'] - self.trimming_start, 6) <= 0)[0][-1]
            self.markerDict['time'] = self.markerDict['time'][self.idx_trim_start:,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][self.idx_trim_start:, :]
            self.coordinateValues = self.coordinateValues.iloc[self.idx_trim_start:]

        if self.trimming_end > 0:
            self.idx_trim_end = np.where(np.round(self.markerDict['time'], 6) <= np.round(self.markerDict['time'][-1] - self.trimming_end, 6))[0][-1] + 1
            self.markerDict['time'] = self.markerDict['time'][:self.idx_trim_end,]
            for marker in self.markerDict['markers']:
                self.markerDict['markers'][marker] = self.markerDict['markers'][marker][:self.idx_trim_end, :]
            self.coordinateValues = self.coordinateValues.iloc[:self.idx_trim_end]

        self.rotation_about_y, self.markerDictRotated = self.rotate_x_forward()

        self.gaitEvents = self.segment_running(n_run_cycles=n_run_cycles, leg=leg)
        self.nGaitCycles = np.shape(self.gaitEvents['ipsilateralIdx'])[0]

        self.treadmillSpeed, _ = self.compute_treadmill_speed(gait_style=run_style)

        self._comValues = None
        self._R_world_to_gait = None
        self._leg_length = None

        self.markerDictRotatedPerGaitCycle = self.rotate_vector_into_gait_frame()

        self.vae = None

    def compute_DMU(self, return_all=False, version="v01"):
        raise NotImplementedError(
            'DMU is defined for walking trials only; use gait_analysis for DMU.')

    def segment_running(self, n_run_cycles=-1, leg='auto', visualize=False):

        # n_run_cycles = -1 finds all accessible strides. Otherwise, it
        # finds that many strides, working backwards from end of trial.

        def detect_toe_peaks(r_toe_rel_x, l_toe_rel_x, prominence=0.3):
            rTO, _ = find_peaks(-r_toe_rel_x, prominence=prominence)
            lTO, _ = find_peaks(-l_toe_rel_x, prominence=prominence)
            return rTO, lTO

        def detect_calc_hs_peaks(r_calc_rel_x, l_calc_rel_x, prominence=0.3):
            rHS, _ = find_peaks(r_calc_rel_x, prominence=prominence)
            lHS, _ = find_peaks(l_calc_rel_x, prominence=prominence)
            return rHS, lHS

        def detect_alternating_run_hs(rHS, lHS):
            if len(rHS) == 0 or len(lHS) == 0:
                return False
            events = []
            for idx in rHS:
                events.append(('r', int(idx)))
            for idx in lHS:
                events.append(('l', int(idx)))
            events.sort(key=lambda x: x[1])
            for i in range(len(events) - 1):
                if events[i][0] == events[i + 1][0]:
                    return False
            return True

        r_calc_rel = (
            self.markerDict['markers']['r_calc_study'] -
            self.markerDict['markers']['r.PSIS_study'])

        r_toe_rel = (
            self.markerDict['markers']['r_toe_study'] -
            self.markerDict['markers']['r.PSIS_study'])
        r_toe_rel_x = r_toe_rel[:, 0]

        l_calc_rel = (
            self.markerDict['markers']['L_calc_study'] -
            self.markerDict['markers']['L.PSIS_study'])
        l_toe_rel = (
            self.markerDict['markers']['L_toe_study'] -
            self.markerDict['markers']['L.PSIS_study'])

        mid_psis = (self.markerDict['markers']['r.PSIS_study'] + self.markerDict['markers']['L.PSIS_study']) / 2
        mid_asis = (self.markerDict['markers']['r.ASIS_study'] + self.markerDict['markers']['L.ASIS_study']) / 2
        mid_dir = mid_asis - mid_psis
        mid_dir_floor = np.copy(mid_dir)
        mid_dir_floor[:, 1] = 0
        mid_dir_floor = mid_dir_floor / np.linalg.norm(mid_dir_floor, axis=1, keepdims=True)

        r_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor, r_calc_rel)
        l_calc_rel_x = np.einsum('ij,ij->i', mid_dir_floor, l_calc_rel)
        r_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor, r_toe_rel)
        l_toe_rel_x = np.einsum('ij,ij->i', mid_dir_floor, l_toe_rel)

        prominences = [0.3, 0.25, 0.2]

        for i, prom in enumerate(prominences):
            rHS, lHS = detect_calc_hs_peaks(
                r_calc_rel_x, l_calc_rel_x, prominence=prom)
            rTO, lTO = detect_toe_peaks(
                r_toe_rel_x, l_toe_rel_x, prominence=prom)
            if not detect_alternating_run_hs(rHS, lHS):
                if prom == prominences[-1]:
                    raise ValueError(
                        'The ordering of run events (calcaneus heel strikes) is not alternating R/L. '
                        'Consider trimming your trial using the trimming_start and trimming_end options.')
                else:
                    print('The run heel-strike peaks were not in alternating R/L order. Trying peak detection again '
                          'with prominence = ' + str(prominences[i + 1]) + '.')
            else:
                break

        if visualize:
            import matplotlib.pyplot as plt
            plt.close('all')
            plt.figure(1)
            plt.plot(self.markerDict['time'], r_toe_rel_x, label='toe')
            plt.plot(self.markerDict['time'], r_calc_rel_x, label='calc')
            plt.scatter(self.markerDict['time'][rHS], r_calc_rel_x[rHS], color='red', label='rHS')
            plt.scatter(self.markerDict['time'][rTO], r_toe_rel_x[rTO], color='blue', label='rTO')
            plt.legend()

            plt.figure(2)
            plt.plot(self.markerDict['time'], l_toe_rel_x, label='toe')
            plt.plot(self.markerDict['time'], l_calc_rel_x, label='calc')
            plt.scatter(self.markerDict['time'][lHS], l_calc_rel_x[lHS], color='red', label='lHS')
            plt.scatter(self.markerDict['time'][lTO], l_toe_rel_x[lTO], color='blue', label='lTO')
            plt.legend()

        if leg == 'auto':
            if rHS[-1] > lHS[-1]:
                leg = 'r'
            else:
                leg = 'l'

        if leg == 'r':
            hsIps = rHS
            toIps = rTO
            hsCont = lHS
            toCont = lTO
        elif leg == 'l':
            hsIps = lHS
            toIps = lTO
            hsCont = rHS
            toCont = rTO

        if len(hsIps) - 1 < n_run_cycles:
            print('You requested {} run cycles, but only {} were found. '
                  'Proceeding with this number.'.format(n_run_cycles, len(hsIps) - 1))
            n_run_cycles = len(hsIps) - 1
        if n_run_cycles == -1:
            n_run_cycles = len(hsIps) - 1
            print('Processing {} run cycles, leg: '.format(n_run_cycles) + leg + '.')

        gaitEvents_ips = np.zeros((n_run_cycles, 3), dtype=int)
        gaitEvents_cont = np.zeros((n_run_cycles, 2), dtype=int)
        if n_run_cycles < 1:
            raise Exception('Not enough run cycles found.')

        for i in range(n_run_cycles):
            gaitEvents_ips[i, 0] = hsIps[-i - 2]
            gaitEvents_ips[i, 2] = hsIps[-i - 1]

            toIpsFound = False
            for j in range(len(toIps)):
                if toIps[-j - 1] > gaitEvents_ips[i, 0] and toIps[-j - 1] < gaitEvents_ips[i, 2] and not toIpsFound:
                    gaitEvents_ips[i, 1] = toIps[-j - 1]
                    toIpsFound = True

            hsContFound = False
            toContFound = False
            for j in range(len(toCont)):
                if toCont[-j - 1] > gaitEvents_ips[i, 0] and toCont[-j - 1] < gaitEvents_ips[i, 2] and not toContFound:
                    gaitEvents_cont[i, 0] = toCont[-j - 1]
                    toContFound = True

            for j in range(len(hsCont)):
                if hsCont[-j - 1] > gaitEvents_ips[i, 0] and hsCont[-j - 1] < gaitEvents_ips[i, 2] and not hsContFound:
                    gaitEvents_cont[i, 1] = hsCont[-j - 1]
                    hsContFound = True

            if not toContFound or not hsContFound:
                print('Could not find contralateral gait event within '
                      'ipsilateral gait event range ' + str(i + 1) +
                      ' steps until the end. Skipping this step.')
                gaitEvents_cont[i, :] = -1
                gaitEvents_ips[i, :] = -1

        mask_ips = (gaitEvents_ips == -1).any(axis=1)
        if all(mask_ips):
            raise Exception('No good steps for ' + leg + ' leg.')
        gaitEvents_ips = gaitEvents_ips[~mask_ips]
        gaitEvents_cont = gaitEvents_cont[~mask_ips]

        gaitEventTimes_ips = self.markerDict['time'][gaitEvents_ips]
        gaitEventTimes_cont = self.markerDict['time'][gaitEvents_cont]

        gaitEvents = {'ipsilateralIdx': gaitEvents_ips,
                      'contralateralIdx': gaitEvents_cont,
                      'ipsilateralTime': gaitEventTimes_ips,
                      'contralateralTime': gaitEventTimes_cont,
                      'eventNamesIpsilateral': ['HS', 'TO', 'HS'],
                      'eventNamesContralateral': ['TO', 'HS'],
                      'ipsilateralLeg': leg}

        return gaitEvents
