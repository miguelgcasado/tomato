#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2014 - 2018 Music Technology Group - Universitat Pompeu Fabra
#
# This file is part of tomato: https://github.com/sertansenturk/tomato/
#
# tomato is free software: you can redistribute it and/or modify it under
# the terms of the GNU Affero General Public License as published by the Free
# Software Foundation (FSF), either version 3 of the License, or (at your
# option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE. See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU Affero General Public License v3.0
# along with this program. If not, see http://www.gnu.org/licenses/
#
# If you are using this extractor please cite the following paper:
#
# Atlı, H. S., Uyar, B., Şentürk, S., Bozkurt, B., and Serra, X. (2014).
# Audio feature extraction for exploring Turkish makam music. In Proceedings
# of 3rd International Conference on Audio Technologies for Music and Media
# (ATMM 2014), pages 142–153, Ankara, Turkey.

import numpy as np
import warnings
import logging
import sys
import essentia.streaming as estr
import essentia.standard as estd
import essentia
from math import ceil
formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                              datefmt='%Y-%m-%d %H:%M:%S')
from memory_profiler import profile
def setup_custom_logger(name):
    formatter = logging.Formatter(fmt='%(asctime)s %(levelname)-8s %(message)s',
                                  datefmt='%Y-%m-%d %H:%M:%S')
    handler = logging.FileHandler('log.txt', mode='w')
    handler.setFormatter(formatter)
    screen_handler = logging.StreamHandler(stream=sys.stdout)
    screen_handler.setFormatter(formatter)
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.addHandler(screen_handler)
    return logger

logger = setup_custom_logger('myapp')

class PredominantMelody(object):
    def __init__(self, hop_size=128, frame_size=2048, bin_resolution=1.0,
                 min_frequency=55, max_frequency=1760, magnitude_threshold=0,
                 peak_distribution_threshold=1.4, filter_pitch=True,
                 confidence_threshold=36, min_chunk_size=50):

        self.hop_size = hop_size  # default hopSize of PredominantMelody
        self.frame_size = frame_size  # default frameSize of PredominantMelody
        self.bin_resolution = bin_resolution  # ~1/3 Hc; recommended for makams
        self.min_frequency = min_frequency  # default: minimum of
        # PitchSalienceFunction
        self.max_frequency = max_frequency  # default: maximum of
        # PitchSalienceFunction
        self.magnitude_threshold = magnitude_threshold  # default of
        # SpectralPeaks; 0 dB?
        self.peak_distribution_threshold = peak_distribution_threshold
        # default in PitchContours is 0.9; we need higher in makams 1.4
        self.filter_pitch = filter_pitch  # call PitchFilter
        self.confidence_threshold = confidence_threshold  # default
        # confidenceThreshold for pitchFilter
        self.min_chunk_size = min_chunk_size  # number of minimum allowed
        # samples of a chunk in PitchFilter; ~145 ms with
        # 128 sample hopSize & 44100 Fs

        self.sample_rate = 44100

    def get_settings(self):
        from essentia import __version__ as essentia_version
        citation = u"Atlı, H. S., Uyar, B., Şentürk, S., Bozkurt, B., " \
                   u"and Serra, X. (2014). Audio feature extraction for " \
                   u"exploring Turkish makam music. In Proceedings of 3rd " \
                   u"International Conference on Audio Technologies for " \
                   u"Music and Media, Ankara, Turkey."

        return {'hopSize': self.hop_size, 'frameSize': self.frame_size,
                'pitchUnit': 'Hz', 'binResolution': self.bin_resolution,
                'minFrequency': self.min_frequency,
                'maxFrequency': self.max_frequency,
                'magnitudeThreshold': self.magnitude_threshold,
                'peakDistributionThreshold': self.peak_distribution_threshold,
                'filterPitch': self.filter_pitch,
                'confidenceThreshold': self.confidence_threshold,
                'sampleRate': self.sample_rate,
                'minChunkSize': self.min_chunk_size,
                'essentiaVersion': essentia_version,
                'citation': citation}
                
    def run(self, fname):
        # load audio and eqLoudness
        # Note: MonoLoader resamples the audio signal to 44100 Hz by default
        logger.info('Load Audio and Extract pitch contour')
        contours_bins, contours_start_times, contour_saliences, duration = \
            self._extract_pitch_contours(fname)

        # run the simplified contour selection
        logger.info('Select contour')
        [pitch, pitch_salience] = self.select_contours(
            contours_bins, contour_saliences, contours_start_times, duration)

        # cent to Hz conversion
        pitch = [0. if p == 0
                 else 55. * 2. ** (self.bin_resolution * p / 1200.)
                 for p in pitch]
        pitch = essentia.array(pitch)
        pitch_salience = essentia.array(pitch_salience)

        # pitch filter
        logger.info('Pitch filter')
        if self.filter_pitch:
            pitch, pitch_salience = self._post_filter_pitch(
                pitch, pitch_salience)

        # generate time stamps
        logger.info('Generate time stamps')
        time_stamps = self._gen_time_stamps(0, len(pitch))

        # [time pitch salience] matrix
        out = np.transpose(
            np.vstack((time_stamps, pitch.tolist(), pitch_salience.tolist())))
        out = out.tolist()

        # settings
        settings = self.get_settings()
        settings.update({'source': fname})
        logger.info('End')
        return {'pitch': out, 'settings': settings}

    def extract(self, fname):
        """
        Alias of self.run
        :param fname: filename
        :return: dictionary with 'pitch' and 'settings' keys
        """
        return self.run(fname)

    def _extract_pitch_contours(self, fname):
        # Hann window with x4 zero padding
        loader = estr.MonoLoader(filename=fname)
        framecutter = estr.FrameCutter(hopSize=self.hop_size, frameSize=self.frame_size)
        loud = estr.EqualLoudness()
        windowing = estr.Windowing(zeroPadding=3 * self.frame_size)
        spectrum = estr.Spectrum(size=self.frame_size * 4)
        spectral_peaks = estr.SpectralPeaks(
            minFrequency=self.min_frequency, maxFrequency=self.max_frequency,
            magnitudeThreshold=self.magnitude_threshold,
            sampleRate=self.sample_rate, orderBy='magnitude')

        # convert unit to cents, PitchSalienceFunction takes 55 Hz as the
        # default reference
        pitch_salience_function = estr.PitchSalienceFunction(
            binResolution=self.bin_resolution)
        pitch_salience_function_peaks = estr.PitchSalienceFunctionPeaks(
            binResolution=self.bin_resolution, minFrequency=self.min_frequency,
            maxFrequency=self.max_frequency)
        pitch_contours = estd.PitchContours(
            hopSize=self.hop_size, binResolution=self.bin_resolution,
            peakDistributionThreshold=self.peak_distribution_threshold)

        # compute frame by frame
        pool = essentia.Pool()
        loader.audio >> loud.signal
        loud.signal >> framecutter.signal
        framecutter.frame >> windowing.frame >> spectrum.frame
        spectrum.spectrum >> spectral_peaks.spectrum
        spectral_peaks.frequencies >> pitch_salience_function.frequencies
        spectral_peaks.magnitudes >> pitch_salience_function.magnitudes
        pitch_salience_function.salienceFunction >> pitch_salience_function_peaks.salienceFunction

        #run_pitch_salience_function.salienceBins >> (pool, 'allframes_salience_peaks_bins')
        #run_pitch_salience_function.salienceBins >> (pool, 'allframes_salience_peaks_contourSaliences')
        #if not np.size(salience_peaks_bins):
        #    salience_peaks_bins = np.array([0])
        #if not np.size(salience_peaks_contour_saliences):
        #    salience_peaks_contour_saliences = np.array([0])

        #pool.add('allframes_salience_peaks_bins', salience_peaks_bins)
        #pool.add('allframes_salience_peaks_contourSaliences',
        #         salience_peaks_contour_saliences)

        # post-processing: contour tracking
        pitch_salience_function_peaks.salienceBins >> (pool,'saliencebins')
        pitch_salience_function_peaks.salienceValues >> (pool, 'saliencevalues')
        essentia.run(loader)

        #pool.add('saliencebins', pitch_salience_function_peaks.salienceBins)
        #pool.add('saliencevalues', pitch_salience_function_peaks.salienceValues)

        #pool['saliencebins'] >> pitch_contours.peakBins
        #pool['saliencevalues'] >> pitch_contours.peakSaliences
        """
        pitch_contours.contoursBins >> (pool, 'contoursbins')
        pitch_contours.contoursSaliences >> (pool, 'contourssaliences')
        pitch_contours.contoursStartTimes >> (pool, 'contoursstarttimes')
        pitch_contours.duration >> (pool, 'duration')

        essentia.run(loader)
        contours_bins = pool['contoursbins']
        contour_saliences = pool['contourssaliences']
        contours_start_times = pool['contoursstarttimes']
        duration = pool['duration']
        """
        contours_bins, contour_saliences, contours_start_times, duration = \
            pitch_contours(
                [f.tolist() for f in pool['saliencebins']],
                [f.tolist() for f in pool['saliencevalues']])
        return contours_bins, contours_start_times, contour_saliences, duration

    def _post_filter_pitch(self, pitch, pitch_salience):
        try:
            run_pitch_filter = estd.PitchFilter(
                confidenceThreshold=self.confidence_threshold,
                minChunkSize=self.min_chunk_size)
            pitch = run_pitch_filter(pitch, pitch_salience)

        except AttributeError:  # fall back to python implementation
            from pitchfilter.pitchfilter import PitchFilter
            run_pitch_filter = PitchFilter()

            # generate time stamps
            time_stamps = self._gen_time_stamps(0, len(pitch))

            temp_pitch = np.vstack((
                time_stamps, pitch, pitch_salience)).transpose()

            temp_pitch = run_pitch_filter.run(temp_pitch)

            pitch = temp_pitch[:, 1]
            pitch_salience = temp_pitch[:, 2]

        return pitch, pitch_salience

    def _gen_time_stamps(self, start_samp, end_samp):
        time_stamps = [s * self.hop_size / float(
            self.sample_rate) for s in range(start_samp, end_samp)]
        return time_stamps

    def select_contours(self, pitch_contours, contour_saliences, start_times,
                        duration):
        sample_rate = self.sample_rate

        hop_size = self.hop_size

        # number in samples in the audio
        num_samples = int(ceil((duration * sample_rate) / hop_size))

        # Start points of the contours in samples
        start_samples = [
            int(round(start_times[i] * sample_rate / float(hop_size)))
            for i in range(0, len(start_times))]

        pitch_contours_no_overlap = []
        start_samples_no_overlap = []
        contour_saliences_no_overlap = []
        lens_no_overlap = []
        try:
            # the pitch contours is a list of numpy arrays, parse them starting
            # with the longest contour
            while pitch_contours:  # terminate when all the contours are
                # checked
                # print len(pitchContours)

                # get the lengths of the pitchContours
                lens = [len(k) for k in pitch_contours]

                # find the longest pitch contour
                long_idx = lens.index(max(lens))

                # pop the lists related to the longest pitchContour and append
                # it to the new list
                pitch_contours_no_overlap.append(pitch_contours.pop(long_idx))
                contour_saliences_no_overlap.append(
                    contour_saliences.pop(long_idx))
                start_samples_no_overlap.append(start_samples.pop(long_idx))
                lens_no_overlap.append(lens.pop(long_idx))

                # accumulate the filled samples
                acc_idx = range(start_samples_no_overlap[-1],
                                start_samples_no_overlap[-1] +
                                lens_no_overlap[-1])

                # remove overlaps
                [start_samples, pitch_contours, contour_saliences] = \
                    self._remove_overlaps(start_samples, pitch_contours,
                                          contour_saliences, lens, acc_idx)
        except ValueError:
            # if the audio input is very short such that Essentia returns a
            # single contour as a numpy array (of length 1) of numpy array
            # (of length 1). In this case the while loop fails directly
            # as it tries to check all the truth value of an all pitch values,
            # instead of checking whether the list is empty or not.
            # Here we handle the error in a Pythonic way by simply breaking the
            # loop and assigning the inputs to outputs since a single contour
            # means nothing to filter
            pitch_contours_no_overlap = pitch_contours
            contour_saliences_no_overlap = contour_saliences
            start_samples_no_overlap = start_samples

        pitch, salience = self._join_contours(pitch_contours_no_overlap,
                                              contour_saliences_no_overlap,
                                              start_samples_no_overlap,
                                              num_samples)

        return pitch, salience

    @staticmethod
    def _join_contours(pitch_contours_no_overlap, contour_saliences_no_overlap,
                       start_samples_no_overlap, num_samples):
        # accumulate pitch and salience
        pitch = np.array([0.] * num_samples)
        salience = np.array([0.] * num_samples)
        for i in range(0, len(pitch_contours_no_overlap)):
            start_samp = start_samples_no_overlap[i]
            end_samp = start_samples_no_overlap[i] + len(
                pitch_contours_no_overlap[i])

            try:
                pitch[start_samp:end_samp] = pitch_contours_no_overlap[i]
                salience[start_samp:end_samp] = contour_saliences_no_overlap[i]
            except ValueError:
                warnings.warn("The last pitch contour exceeds the audio "
                              "length. Trimming...")

                pitch[start_samp:] = pitch_contours_no_overlap[i][:len(
                    pitch) - start_samp]
                salience[start_samp:] = contour_saliences_no_overlap[i][:len(
                    salience) - start_samp]
        return pitch, salience

    @staticmethod
    def _remove_overlaps(start_samples, pitch_contours, contour_saliences,
                         lens, acc_idx):
        # remove overlaps
        rmv_idx = []
        for i in range(0, len(start_samples)):
            # print '_' + str(i)
            # create the sample index vector for the checked pitch contour
            curr_samp_idx = range(start_samples[i], start_samples[i] + lens[i])

            # get the non-overlapping samples
            curr_samp_idx_no_overlap = list(set(curr_samp_idx) -
                                            set(acc_idx))

            if curr_samp_idx_no_overlap:
                temp = min(curr_samp_idx_no_overlap)
                keep_idx = range(temp - start_samples[i],
                                 (max(curr_samp_idx_no_overlap) -
                                  start_samples[i]) + 1)

                # remove all overlapping values
                pitch_contours[i] = np.array(pitch_contours[i])[keep_idx]
                contour_saliences[i] = np.array(contour_saliences[i])[keep_idx]
                # update the startSample
                start_samples[i] = temp
            else:  # totally overlapping
                rmv_idx.append(i)

        # remove totally overlapping pitch contours
        rmv_idx = sorted(rmv_idx, reverse=True)
        for r in rmv_idx:
            pitch_contours.pop(r)
            contour_saliences.pop(r)
            start_samples.pop(r)

        return start_samples, pitch_contours, contour_saliences
