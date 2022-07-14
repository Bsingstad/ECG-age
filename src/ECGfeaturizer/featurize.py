import numpy as np
import wfdb
import neurokit2 as nk
#import plotly.graph_objects as go
import pandas as pd
import os
from scipy.io import loadmat
from wfdb import processing
KEYS = ['ECG_P_Onsets','ECG_P_Peaks', 'ECG_P_Offsets', 'ECG_R_Onsets' 'ECG_Q_Peaks', 
        'ECG_R_Peaks', 'ECG_R_Offsets','ECG_S_Peaks', 'ECG_T_Onsets','ECG_T_Peaks','ECG_T_Offsets']


class get_features:

    def __init__(self, ecg_waveform, sample_frequency):
        self.ecg_waveform = ecg_waveform
        self.sample_frequency = sample_frequency
    
        #self.nk_data = nk.ecg_process(self.ecg_waveform,self.sample_frequency, method = "wavelet")[0]

    @staticmethod
    def _calc_mean_and_std(sequence):
        return sequence.mean(), sequence.std()

    def get_ecg_data(self):
        ecg_cleaned = nk.ecg_clean(self.ecg_waveform, sampling_rate=self.sample_frequency, method="neurokit")
        # R-peaks
        instant_peaks, rpeaks, = nk.ecg_peaks(
            ecg_cleaned=ecg_cleaned, sampling_rate=self.sample_frequency, method="neurokit", correct_artifacts=True
        )



        signals = pd.DataFrame(
            {"ECG_Raw": self.ecg_waveform, "ECG_Clean": ecg_cleaned}
        )

        # Additional info of the ecg signal
        delineate_signal, delineate_info = nk.ecg_delineate(
            ecg_cleaned=ecg_cleaned, rpeaks=rpeaks, sampling_rate=self.sample_frequency, method="dwt"
        )


        signals = pd.concat([signals, instant_peaks, delineate_signal], axis=1)
        self.nk_data = signals
   

    
    def return_peaks(self, key, return_sequence=False):
        """
        Return R, P, Q, S, T, P and T peaks
        :Parameters:
            key_idx:
                A integer from 0 and 10
            return_sequence:
                If True a sequence is returned, if False mean and standar deviation are returned
        """
        peaks = np.where(self.nk_data[key]==1)[0]
        
        if return_sequence == True:
            return peaks
        else:
            return self._calc_mean_and_std(peaks)

    def calculate_b2b_intervals(self,peaks):
        return np.diff(peaks)/self.sample_frequency


    def calculate_intervals(self,peaks_1, peaks_2):
        if not len(peaks_1) == len(peaks_2):
            if len(peaks_2) > len(peaks_1):
                if peaks_2[0] < peaks_1[0]:
                    peaks_2 = peaks_2[1:]
            elif len(peaks_2) < len(peaks_1):
                if peaks_2[-1] < peaks_1[-1]:
                    peaks_2 = peaks_2[1:]
        if not len(peaks_1) == len(peaks_2):
            for i in range(min([len(peaks_1),len(peaks_2)])-1):
                while not(peaks_1[i]<peaks_2[i]<peaks_1[i+1]):
                    if (peaks_1[i] > peaks_2[i]):
                        peaks_2 = np.delete(peaks_2,i)
                    if (peaks_2[i] > peaks_1[i+1]): 
                        peaks_1 = np.delete(peaks_1,i+1)
                if i == min([len(peaks_1),len(peaks_2)])-2:
                    break
        if len(peaks_1) > len(peaks_2):
            peaks_1 = peaks_1[:len(peaks_2)]
        elif len(peaks_1) < len(peaks_2):
            peaks_2 = peaks_2[:len(peaks_1)]
        return (peaks_2 - peaks_1)/self.sample_frequency
    
    def _reject_outliers(data, m = 4.):
        d = np.abs(data - np.median(data))
        mdev = np.median(d)
        s = d/mdev if mdev else 0.
        return data[s<m]
            

    def calc_intervals(self):
        self.get_ecg_data()
        p_onsets = self.return_peaks('ECG_P_Onsets', return_sequence=True)
        r_onsets = self.return_peaks('ECG_R_Onsets', return_sequence=True)
        pr_interval = self.calculate_intervals(p_onsets,r_onsets)
        #pr_interval = self._reject_outliers(pr_interval)
        t_offset = self.return_peaks('ECG_T_Offsets', return_sequence=True)
        qt_interval = self.calculate_intervals(r_onsets,t_offset)
        #qt_interval = self._reject_outliers(qt_interval)
        r_peaks = self.return_peaks('ECG_R_Peaks', return_sequence=True)
        rr_interval = self.calculate_b2b_intervals(r_peaks)
        #rr_interval = self._reject_outliers(rr_interval)
        return rr_interval, pr_interval, qt_interval



    