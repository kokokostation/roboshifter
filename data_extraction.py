import pickle
import os
import json
import numpy as np
import pandas as pd
import datetime as dt
from repoze.lru import lru_cache
from collections import defaultdict

from data_manager import DataManager, PATHS


class DataExtractor(DataManager):
    def __init__(self, paths=PATHS):
        DataManager.__init__(self, paths)

    @property
    @lru_cache(maxsize=228)
    def bootstrap_fit(self):
        return pd.read_pickle(self.paths['BOOTSTRAP_FIT'])

    @property
    @lru_cache(maxsize=228)
    def data_ref(self):
        return self.get_data_ref()

    @staticmethod
    def np_histo(histo):
        if isinstance(histo, dict):
            for key, value in histo.items():
                histo[key] = DataExtractor.np_histo(value)
            return histo
        elif isinstance(histo, list):
            return np.array(histo)
        else:
            return histo

    @staticmethod
    def inverse_dict(d):
        result = defaultdict(list)

        for key, value in d.items():
            result[value].append(key)

        return result

    @staticmethod
    def inverse_to_dict(d):
        result = {}

        for key, value in d.items():
            for v in value:
                result[v] = key

        return result

    def get_run_helper(self, prefix, run):
        run_file = os.path.join(prefix, '{}.pickle'.format(run))
        return DataExtractor.np_histo(pd.read_pickle(run_file))
    
    def get_run(self, run):
        return self.get_run_helper(self.paths['RUN_DIR'], run)

    def get_reference_by_hash(self, hash):
        return self.get_run_helper(self.paths['REF_DIR'], hash)
    
    def get_reference(self, run):
        data_ref = self.get_data_ref()

        if run in data_ref:
            return self.get_reference_by_hash(data_ref[run])
        else:
            return None

    def get_data_ref(self):
        return pd.read_pickle(self.paths['DATA_REF'])

    def get_linear_data(self):
        return pd.read_pickle(self.paths['LINEAR_DATA_PREPARED'])

    WEIRD_HT = {
        'WeirdTProfile': [
            'RICH/RiLongTrkEff/All/effVChi2PDOF',
            'Velo/VeloTrackMonitor/Pseudoefficiency_per_sensor_vs_sensorID',
        ],
        'WeirdMuTProfile': [
            'MuIDLambdaPlot/pion/Prof_eff',
            'MuIDLambdaPlot/proton/Prof_eff'
        ]
    }

    def get_histo_types(self):
        result = pd.read_pickle(self.paths['HISTO_TYPES'])

        for ht, histos in DataExtractor.WEIRD_HT.items():
            for hk in histos:
                result[hk] = ht

        return result

    def get_histograms_list(self):
        with open(self.paths['HISTO_MAP']) as histo_file:
            histograms = json.load(histo_file)

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
    
    def get_valid_runs(self):
        return pd.read_pickle(self.paths['VALID_RUNS'])

    def get_train_data(self):
        return pd.read_pickle(self.paths['TRAIN_DATA'])

    def get_rejected_runs(self):
        return pd.read_pickle(self.paths['REJECTED_RUNS'])

    def get_bootstrap_result(self):
        return pd.read_pickle(self.paths['BOOTSTRAP_RESULT'])

    def get_ht_interactions(self):
        return pd.read_pickle(self.paths['INTERACTIONS'])

    def get_interactions(self):
        ht_int = self.get_ht_interactions()
        histo_types = self.get_histo_types()

        result = {}

        for hk, ht in histo_types.items():
            for key, value in ht_int[ht].items():
                result[(hk, key)] = value

        for key, value in self.get_assesed_interactions().items():
            if 'nothing' in value:
                del result[key]
            else:
                result[key] = value

        return result

    def get_train_ref_hashes(self):
        df = self.get_train_data()

        return list(set(self.data_ref[i] for i in df.index))

    def get_th1d_types(self):
        result = defaultdict(list)

        with open(self.paths['DISTRIBUTION_CLASSES'], 'r') as infile:
            for line in infile.read().splitlines():
                if ':' in line:
                    cls = line[:-1]
                else:
                    result[cls].append(line)

        return result

    def get_run_numbers(self, ref_hashes):
        return [run_number for run_number, ref_hash in self.data_ref.items()
                if ref_hash in ref_hashes]

    def get_assesed_interactions(self):
        path = self.paths['ASSESED_INTERACTIONS']

        if os.path.isfile(path):
            return pd.read_pickle(path)
        else:
            return []


if __name__ == '__main__':
    de = DataExtractor()

    de.get_interactions()