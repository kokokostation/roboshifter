import os
from shutil import copyfile

from data_manager import DataManager
from features import FeatureExtractor
from meta_manager import MetaManager
from classifier import Roboshifter
from root2py import Root2Py
from train_data import TrainDataMaker
from utils import Maybe

COLLECTOR_PATHS = {
    'MONET_HISTOS': 'monet_histos.pickle',
    'TREND_LINEAR_DATA': 'trend_linear_data.pickle',
    'DATA_REF': 'data_ref.pickle',
    'TRAIN_DATA': 'train.pickle',
    'HISTO_TYPES': 'histo_types.pickle',
    'ASSESED_INTERACTIONS': 'assesed_interactions.pickle',
    'FEATURE_DIR': 'features',
    'RUN_DIR': 'run',
    'REF_DIR': 'ref'
}


class Collector(DataManager):
    def __init__(self, njobs=4, root_dir='.', paths=COLLECTOR_PATHS):
        paths = {key: os.path.join(root_dir, value) for key, value in paths.items()}

        DataManager.__init__(self, paths)

        self.njobs = njobs

        self.init()

    def init(self):
        dirs = [value for key, value in self.paths.items() if key[-3:] == 'DIR']

        if not os.path.exists(dirs[0]):
            for dir in dirs:
                os.mkdir(dir)

            assesed_interactions_file = os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                                     'data', 'assesed_interactions.pickle')
            copyfile(assesed_interactions_file, self.paths['ASSESED_INTERACTIONS'])

    def get_histograms_list(self):
        histograms = self.read_pickle('MONET_HISTOS')

        histograms_list = []
        for path, histos in histograms.items():
            for histokey, title in histos.items():
                histograms_list.append({
                    "path": path,
                    "key": histokey,
                    "title": title
                })

        return histograms_list

    def get_histo_keys(self):
        return list(set([histo['key'] for histo in self.get_histograms_list()]))

    def get_train_data(self):
        return self.read_pickle('TRAIN_DATA')

    def write_train_data(self, data):
        self.write_pickle(data, 'TRAIN_DATA')

    def get_histo_types(self):
        return self.read_pickle('HISTO_TYPES')

    def write_histo_types(self, data):
        self.write_pickle(data, 'HISTO_TYPES')

    def get_ht_interactions(self):
        return Roboshifter.make_ht_interactions()

    def get_assesed_interactions(self):
        return self.read_pickle('ASSESED_INTERACTIONS')

    def get_interactions(self):
        ht_interactions = self.get_ht_interactions()

        result = {}

        for hk, ht in self.get_histo_types().items():
            for key, value in ht_interactions[ht].items():
                result[(hk, key)] = value

        for key, value in self.get_assesed_interactions().items():
            if 'nothing' in value:
                del result[key]
            else:
                result[key] = value

        return result

    def get_run(self, run):
        return self.make_result(self.get_root2py(False, run), 'No run data available')

    def get_reference_by_hash(self, hash):
        return self.get_root2py(True, hash)

    def get_data_ref(self):
        return self.read_pickle('DATA_REF')

    def make_result(self, result, error_on_fail):
        return Maybe(error_message=[error_on_fail]) if result is None else result

    def get_reference(self, run):
        result = None
        data_ref = self.get_data_ref()

        if run in data_ref:
            result = self.get_reference_by_hash(data_ref[run])

        return self.make_result(result, 'No reference data available')

    def get_linear_data(self):
        return self.read_pickle('TREND_LINEAR_DATA')

    def write_run_features(self, features, run_number):
        self.write_pickle(features, 'FEATURE_DIR', run_number, 'pickle')

    def get_run_features(self, run_number):
        return self.read_pickle('FEATURE_DIR', run_number, 'pickle')

    def write_root2py(self, reference, obj, run_handle):
        self.write_pickle(obj, 'REF_DIR' if reference else 'RUN_DIR', run_handle, 'pickle')

    def get_root2py(self, reference, run_handle):
        return self.read_pickle('REF_DIR' if reference else 'RUN_DIR', run_handle, 'pickle')

    def make_meta(self, external_collector):
        meta_manager = MetaManager(self, external_collector)
        meta_manager.make_meta()

    def make_data(self, external_collector, run_numbers):
        self.make_meta(external_collector)

        root2py = Root2Py(self, external_collector, self.njobs)
        root2py.process_runs(run_numbers)

        feature_extractor = FeatureExtractor(self, self.njobs)
        feature_extractor.make_features(run_numbers)

        train_data_maker = TrainDataMaker(self)
        train_data_maker.make_train_data(run_numbers, feature_extractor)

    def get_roboshifter_data(self, run_numbers):
        return TrainDataMaker(self).get_roboshifter_data(run_numbers)








