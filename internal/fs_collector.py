import os

import ROOT

from data_manager import DataManager
from roboshifter.external.external_collector import ExternalCollector

PATHS = {
    'HISTO_MAP': 'data/meta/monet_histos.json',
    'RUN_FILES': 'data/meta/run_files.json',
    'TREND_LINEAR_DATA': 'data/meta/trend_linear_data.json',

    'DATA_REF': 'wd/data_ref.pickle',

    'MONET_DATA_PATH': 'data/monet-roboshifter/'
}


class FsCollector(ExternalCollector, DataManager):
    def __init__(self, root):
        paths = {key: os.path.join(root, value) for key, value in PATHS.items()}

        DataManager.__init__(self, paths)

    def get_monet_histos(self):
        return self.read_file('HISTO_MAP')

    def get_trend_linear_data(self):
        return self.read_file('TREND_LINEAR_DATA')

    def get_run_files(self):
        return self.read_file('RUN_FILES')

    def read_root(self, fname):
        return ROOT.TFile(os.path.join(self.paths['MONET_DATA_PATH'],
                                       '{}.root'.format(fname)))

    def get_run_tfile(self, run_number):
        return self.read_root(run_number)

    def get_reference_tfile(self, run_number):
        data_ref = self.read_pickle('DATA_REF')
        ref = data_ref.get(run_number)

        return self.read_root(os.path.join('ref', ref)) if ref is not None else None
