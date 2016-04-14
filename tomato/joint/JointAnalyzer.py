import json
from scipy.io import savemat
from six.moves import cStringIO
import tempfile
import numpy as np
from copy import deepcopy
import timeit

from alignedpitchfilter.AlignedPitchFilter import AlignedPitchFilter
from alignednotemodel.AlignedNoteModel import AlignedNoteModel

from ..MCRCaller import MCRCaller
from ..IO import IO
from ..Analyzer import Analyzer
from ..Plotter import Plotter

import warnings
import logging
logging.basicConfig(level=logging.INFO)

# instantiate a mcr_caller
_mcr_caller = MCRCaller()


class JointAnalyzer(Analyzer):
    _inputs = ['notes', 'note_models', 'makam', 'pitch_filtered', 'sections',
               'tonic', 'tempo']

    def __init__(self, verbose=False):
        super(JointAnalyzer, self).__init__(verbose=verbose)

        # extractors
        self._tonicTempoExtractor = _mcr_caller.get_binary_path(
            'extractTonicTempoTuning')
        self._audioScoreAligner = _mcr_caller.get_binary_path(
            'alignAudioScore')
        self._alignedPitchFilter = AlignedPitchFilter()
        self._alignedNoteModel = AlignedNoteModel()

    def analyze(self, symbtr_txt_filename='', score_data=None,
                audio_filename='', audio_pitch=None, **kwargs):
        input_f = self._parse_inputs(**kwargs)

        # joint score-informed tonic identification and tempo estimation
        try:  # if both are given in advance don't recompute
            input_f['tonic'], input_f['tempo'] = self.extract_tonic_tempo(
                symbtr_txt_filename, score_data, audio_filename, audio_pitch)
        except RuntimeError as e:
            warnings.warn(e.message, RuntimeWarning)
            joint_features = None
            audio_features = None
            # Everything else will fail; return None
            return joint_features, audio_features

        # section linking and note-level alignment
        try:
            input_f['aligned_sections'], input_f['notes'], input_f[
                'section_links'], input_f['section_candidates'] = \
                self.align_audio_score(symbtr_txt_filename, score_data,
                                       audio_filename, audio_pitch,
                                       input_f['tonic'], input_f['tempo'])
        except RuntimeError as e:
            warnings.warn(e.message, RuntimeWarning)
            joint_features = None
            audio_features = {'tonic': input_f['tonic'],
                              'tempo': input_f['tempo']}
            return joint_features, audio_features

        # aligned pitch filter
        temp_out = self._call_analysis_step(
            'filter_pitch', input_f['pitch_filtered'], audio_pitch,
            input_f['notes'])
        if temp_out is not None:
            input_f['pitch_filtered'], input_f['notes'] = temp_out
        else:
            input_f['pitch_filtered'], input_f['notes'] = [None, None]

        # aligned note models
        temp_out = self._call_analysis_step(
            'compute_note_models', input_f['note_models'],
            input_f['pitch_filtered'], input_f['notes'],
            input_f['tonic']['symbol'])
        if temp_out is not None:
            input_f['note_models'], dummy_pd, input_f['tonic'] = temp_out
        else:
            input_f['note_models'], input_f['tonic'] = [None, None]

        joint_features = {'sections': input_f['aligned_sections'],
                          'notes': input_f['notes']}
        audio_features = {'makam': score_data['makam']['symbtr_slug'],
                          'pitch_filtered': input_f['pitch_filtered'],
                          'tonic': input_f['tonic'], 'tempo': input_f['tempo'],
                          'note_models': input_f['note_models']}

        return joint_features, audio_features

    @classmethod
    def summarize(cls, audio_features=None, score_features=None,
                  joint_features=None, score_informed_audio_features=None):
        # initialize the summary dict
        sdict = {'score': score_features, 'audio': {}, 'joint': {}}

        sdict['audio'] = cls._summarize_common_audio_features(
            audio_features, score_informed_audio_features)

        # pitch_filtered is reduntant name and it might not be computed
        sdict['audio']['pitch'] = sdict['audio'].pop("pitch_filtered", None)
        if sdict['audio']['pitch'] is None:
            sdict['audio']['pitch'] = audio_features['pitch']

        # tempo if computed
        try:
            sdict['audio']['tempo'] = score_informed_audio_features['tempo']
        except KeyError:
            logging.debug("Tempo feature is not available.")

        # accumulate joint dict
        try:
            sdict['joint']['sections'] = joint_features['sections']
            sdict['joint']['notes'] = joint_features['notes']
        except KeyError:
            sdict['joint'] = {}
            logging.debug("Section links and aligned notes are not available.")

        return sdict

    @staticmethod
    def _summarize_common_audio_features(audio_features,
                                         score_informed_audio_features):
        common_feature_names = ['makam', 'melodic_progression', 'note_models',
                                'pitch_class_distribution',
                                'pitch_distribution', 'pitch_filtered',
                                'tonic', 'transposition']

        common_audio_features = dict()
        for cf in common_feature_names:
            score_informed = (score_informed_audio_features is not None and
                              score_informed_audio_features[cf] is not None)
            if score_informed:
                common_audio_features[cf] = score_informed_audio_features[cf]
            else:
                common_audio_features[cf] = audio_features[cf]

        return common_audio_features

    def extract_tonic_tempo(self, score_filename='', score_data=None,
                            audio_filename='', audio_pitch=None):
        tic = timeit.default_timer()
        self.vprint(u"- Extracting score-informed tonic and tempo of {0:s}"
                    .format(audio_filename))

        # create the temporary input and output files wanted by the binary
        temp_score_data_file = IO.create_temp_file(
            '.json', json.dumps(score_data))

        # matlab
        matout = cStringIO()
        savemat(matout, audio_pitch)

        temp_pitch_file = IO.create_temp_file(
            '.mat', matout.getvalue())

        temp_out_folder = tempfile.mkdtemp()

        # call the binary
        callstr = ["{0:s} {1:s} {2:s} {3:s} {4:s} {5:s}".format(
            self._tonicTempoExtractor, score_filename, temp_score_data_file,
            audio_filename, temp_pitch_file, temp_out_folder)]

        out, err = _mcr_caller.call(callstr)

        # check the MATLAB output
        if "Tonic-Tempo-Tuning Extraction took" not in out:
            IO.remove_temp_files(
                temp_score_data_file, temp_pitch_file, temp_out_folder)
            raise RuntimeError("Score-informed tonic, tonic and tuning "
                               "extraction is not successful. Please "
                               "check the error in the terminal.")

        out_dict = IO.load_json_from_temp_folder(
            temp_out_folder, ['tempo', 'tonic', 'tuning'])

        # unlink the temporary files
        IO.remove_temp_files(
            temp_score_data_file, temp_pitch_file, temp_out_folder)

        # tidy outouts
        # We omit the tuning output in the binary because
        # get_aligned_note_models is more informative
        procedure = 'Score informed joint tonic and tempo extraction'

        tonic = out_dict['tonic']['scoreInformed']
        tonic = IO.dict_keys_to_snake_case(tonic)
        tonic['procedure'] = procedure
        tonic['source'] = audio_filename

        tempo = out_dict['tempo']['scoreInformed']
        tempo['average'] = IO.dict_keys_to_snake_case(tempo['average'])
        tempo['average']['procedure'] = procedure
        tempo['average']['source'] = audio_filename
        tempo['average'].pop("method", None)

        tempo['relative'] = IO.dict_keys_to_snake_case(
            tempo['relative'])
        tempo['relative']['procedure'] = procedure
        tempo['relative']['source'] = audio_filename
        tempo['relative'].pop("method", None)

        # print elapsed time, if verbose
        self.vprint_time(tic, timeit.default_timer())

        return tonic, tempo

    def align_audio_score(self, score_filename='', score_data=None,
                          audio_filename='', audio_pitch=None,
                          audio_tonic=None, audio_tempo=None):
        tic = timeit.default_timer()
        self.vprint(u"- Aligning audio recording {0:s} and music score {1:s}."
                    .format(audio_filename, score_filename))

        # create the temporary input and output files wanted by the binary
        temp_score_data_file = IO.create_temp_file(
            '.json', json.dumps(score_data))

        # tonic has to be enclosed in the key 'score_informed' and all the
        # keys have to start with a capital letter
        audio_tonic_ = IO.dict_keys_to_camel_case(audio_tonic)
        temp_tonic_file = IO.create_temp_file(
            '.json', json.dumps({'scoreInformed': audio_tonic_}))

        # tempo has to be enclosed in the key 'score_informed' and all the
        # keys have to start with a capital letter
        audio_tempo_ = deepcopy(audio_tempo)
        audio_tempo_['relative'] = IO.dict_keys_to_camel_case(
            audio_tempo['relative'])
        audio_tempo_['average'] = IO.dict_keys_to_camel_case(
            audio_tempo['average'])
        temp_tempo_file = IO.create_temp_file(
            '.json', json.dumps({'scoreInformed': audio_tempo_}))

        # matlab
        matout = cStringIO()
        savemat(matout, audio_pitch)

        temp_pitch_file = IO.create_temp_file(
            '.mat', matout.getvalue())

        temp_out_folder = tempfile.mkdtemp()

        # call the binary
        callstr = ["{0:s} {1:s} {2:s} '' {3:s} {4:s} {5:s} {6:s} '' "
                   "{7:s}".format(self._audioScoreAligner, score_filename,
                                  temp_score_data_file, audio_filename,
                                  temp_pitch_file, temp_tonic_file,
                                  temp_tempo_file, temp_out_folder)]

        out, err = _mcr_caller.call(callstr)
        # check the MATLAB output
        if "Audio-score alignment took" not in out:
            IO.remove_temp_files(
                temp_score_data_file, temp_tonic_file, temp_tempo_file,
                temp_pitch_file, temp_out_folder)
            raise RuntimeError("Audio score alignment is not successful. "
                               "Please check the error in the terminal.")

        out_dict = IO.load_json_from_temp_folder(
            temp_out_folder, ['sectionLinks', 'alignedNotes'])

        # unlink the temporary files
        IO.remove_temp_files(temp_score_data_file, temp_tonic_file,
                             temp_tempo_file, temp_pitch_file, temp_out_folder)

        notes = [IO.dict_keys_to_snake_case(n)
                 for n in out_dict['alignedNotes']['notes']]

        # print elapsed time, if verbose
        self.vprint_time(tic, timeit.default_timer())

        return (out_dict['sectionLinks']['alignedLinks'], notes,
                out_dict['sectionLinks']['sectionLinks'],
                out_dict['sectionLinks']['candidateLinks'])

    def filter_pitch(self, pitch, aligned_notes):
        tic = timeit.default_timer()
        self.vprint(u"- Filtering predominant melody of {0:s} after "
                    u"audio-score alignment.".format(pitch['source']))
        aligned_notes_ = [IO.dict_keys_to_camel_case(n)
                          for n in deepcopy(aligned_notes)]

        pitch_temp, notes_filtered, synth_pitch = \
            self._alignedPitchFilter.filter(pitch['pitch'], aligned_notes_)

        notes_filtered = [IO.dict_keys_to_snake_case(n)
                          for n in notes_filtered]

        pitch_filtered = deepcopy(pitch)
        pitch_filtered['pitch'] = pitch_temp
        pitch_filtered['citation'] = 'SenturkThesis'
        pitch_filtered['procedure'] = 'Pitch filtering according to ' \
                                      'audio-score alignment'

        # print elapsed time, if verbose
        self.vprint_time(tic, timeit.default_timer())

        return pitch_filtered, notes_filtered

    def compute_note_models(self, pitch, aligned_notes, tonic_symbol):
        tic = timeit.default_timer()
        self.vprint(u"- Computing the note models for {0:s}".
                    format(pitch['source']))

        aligned_notes_ = [IO.dict_keys_to_camel_case(n)
                          for n in deepcopy(aligned_notes)]

        note_models, pitch_distibution, tonic = self._alignedNoteModel.\
            get_models(pitch['pitch'], aligned_notes_, tonic_symbol)

        for note in note_models.keys():
            note_models[note] = IO.dict_keys_to_snake_case(
                note_models[note])

        tonic = IO.dict_keys_to_snake_case(tonic['alignment'])
        tonic['source'] = pitch['source']

        # print elapsed time, if verbose
        self.vprint_time(tic, timeit.default_timer())

        return note_models, pitch_distibution, tonic

    def set_tonic_tempo_extractor_params(self, **kwargs):
        raise NotImplementedError

    def set_audio_score_aligner_params(self, **kwargs):
        raise NotImplementedError

    def set_pitch_filter_params(self, **kwargs):
        self._set_params('_alignedPitchFilter', **kwargs)

    def set_note_model_params(self, **kwargs):
        self._set_params('_alignedNoteModel', **kwargs)

    @staticmethod
    def plot(summarized_features):
        pitch = np.array(deepcopy(
            summarized_features['audio']['pitch']['pitch']))
        pitch[pitch[:, 1] < 20.0, 1] = np.nan  # remove inaudible for plots

        pitch_distribution = deepcopy(
            summarized_features['audio']['pitch_distribution'])

        sections = deepcopy(summarized_features['joint']['sections'])

        try:  # convert the bins to hz, if they are given in cents
            pitch_distribution.cent_to_hz()
        except ValueError:
            logging.debug('The pitch distribution should already be in hz')

        try:
            note_models = deepcopy(summarized_features['joint']['note_models'])
        except KeyError:  # aligned note_models is not computed
            note_models = deepcopy(summarized_features['audio']['note_models'])

        melodic_progression = deepcopy(
            summarized_features['audio']['melodic_progression'])

        try:
            aligned_notes = deepcopy(summarized_features['joint']['notes'])
        except KeyError:  # audio score alignment failed
            aligned_notes = None

        return Plotter.plot_audio_features(
            pitch=pitch, pitch_distribution=pitch_distribution,
            sections=sections, notes=aligned_notes, note_models=note_models,
            melodic_progression=melodic_progression)
