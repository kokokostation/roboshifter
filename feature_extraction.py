import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
import os
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
import json
from sklearn.preprocessing import LabelEncoder
import datetime as dt
import scipy.stats as sps
import pickle
from collections import defaultdict
from time import mktime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error as mse
from tqdm import tqdm
from utils import Maybe

from data_manager import DataManager, PATHS
from data_extraction import DataExtractor


class FeatureExtractor(DataManager):
    def __init__(self, paths=PATHS):
        DataManager.__init__(self, paths)
        self.data_extractor = DataExtractor()

    @staticmethod
    def to_timestamp(date):
        return int(mktime(date.timetuple()))

    @staticmethod
    def make_distribution(histo):
        vals = histo['vals'].astype(np.float64)

        vals = np.maximum(0, vals)

        distr = vals / vals.sum()

        return rv_discrete(values=(np.arange(distr.shape[0]), distr))

    @staticmethod
    def assemble_name(prefix, name):
        pn = [prefix, name]

        for i, item in enumerate(pn):
            if not isinstance(item, tuple):
                pn[i] = (item,)

        return pn[0] + pn[1]

    @staticmethod
    def rename_features(features, prefix):
        features = features.copy()

        features.index = features.index.map(partial(FeatureExtractor.assemble_name, prefix))

        return features

    def statistical_distance(self, strong, weak):
        features = {
            'variational_distance': np.abs(strong.pk - weak.pk).sum(),
            'kolmogorov_smirnov_statistic': np.abs(strong.pk.cumsum() - weak.pk.cumsum()).max(),
        }

        return pd.Series(features)

    def distribution_features(self, distr):
        mean, var, skew, kurt = distr.stats(moments='mvsk')

        features = {
            'mean': mean,
            'var': var,
            'skew': skew,
            'kurt': kurt,
        }

        return pd.Series(features)

    def regression_distance(self, runv, refv):
        abs_diff = np.abs(refv - runv)

        return Maybe(value=pd.Series({
            'variational': np.sum(abs_diff),
            'kolmogorov': np.max(abs_diff),
            'mse': mse(runv, refv)
        }))

    def value_features(self, runv, refv, prefix='stats'):
        runs, refs = runv.sum(), refv.sum()

        features = pd.Series({
            'runs': runs,
            'refs': refs,
            'refs - runs': refs - runs,
        })

        result = pd.concat([features, self.regression_distance(runv, refv)])

        return Maybe(value=FeatureExtractor.rename_features(result, prefix))

    def histo_features(self, runh, refh, const_std_features=True):
        result = pd.Series()

        try:
            rund, refd = map(FeatureExtractor.make_distribution, [runh, refh])
        except ValueError:
            return Maybe(error_message='Integral for this histogram equals to zero')

        runf = self.distribution_features(rund)
        reff = self.distribution_features(refd)
        distancef = self.statistical_distance(refd, rund)
        funcf = reff - runf

        flavors = [(runf, 'run'), (distancef, 'distance'), (funcf, 'func')]
        if const_std_features:
            flavors.append((reff, 'ref'))

        for features, flavor in flavors:
            result = result.append(FeatureExtractor.rename_features(features, flavor))

        return Maybe(value=result)

    def th1d_features(self, runh, refh):
        return Maybe.concat([self.histo_features(runh, refh),
                             self.value_features(runh['vals'], refh['vals'])])

    def get_efficiency(self, teff):
        result = deepcopy(teff['passed'])
        total = teff['total']['vals']
        total[total == 0] = 1
        result['vals'] /= total

        return result, teff['total']['vals'] + teff['passed']['vals']

    def tefficiency_features(self, runh, refh):
        (run_eff, run_stats), (ref_eff, ref_stats) = map(self.get_efficiency, [runh, refh])

        return Maybe.concat([self.histo_features(run_eff, ref_eff),
                             self.value_features(run_stats, ref_stats),
                             self.value_features(run_eff['vals'], ref_eff['vals'], 'eff')])

    def tprofile_features(self, runh, refh):
        distance = Maybe(value=FeatureExtractor.rename_features(
            self.regression_distance(runh['vals'], refh['vals']),
            'distance'))

        result = Maybe.concat([distance,
                               self.value_features(runh['entries'], refh['entries']),
                               self.histo_features({'vals': runh['errs']}, {'vals': refh['errs']})])

        return result

    def weird_tprofile_features(self, runh, refh):
        mask = (runh['vals'] != 0) & (refh['vals'] != 0)

        for d in [runh, refh]:
            d['vals'] = d['vals'][mask]

        s = mask.sum()

        if s == 0:
            return Maybe(error_message='Too many zeros in graph to process features')
        else:
            return Maybe.concat([Maybe(value=pd.Series({('zeros', 'vals'): s})),
                                 self.tprofile_features(runh, refh)])

    @staticmethod
    def get_histo(histo_key, ht, runh):
        result = runh[histo_key]

        if ht == 'TH1D':
            result['vals'] = np.maximum(0, result['vals'])

        return result

    def get_features(self, run_number):
        runh = self.data_extractor.get_run(run_number)
        refh = self.data_extractor.get_reference(run_number)

        histo_type = self.data_extractor.get_histo_types()

        HANDLERS = {
            'TH1D': self.th1d_features,
            'TEfficiency': self.tefficiency_features,
            'TProfile': self.tprofile_features,
            'WeirdTProfile': self.weird_tprofile_features,
            'WeirdMuTProfile': self.weird_tprofile_features
        }

        result = pd.Series()

        for histo_key in self.data_extractor.get_histo_keys():
            if histo_key not in runh or histo_key in FeatureExtractor.EXCLUDED_TRAIN_HISTOS:
                continue

            ht = histo_type[histo_key]
            handler = HANDLERS.get(ht, None)
            if handler is not None:
                rund, refd = map(partial(FeatureExtractor.get_histo, histo_key, ht), [runh, refh])

                features = handler(rund, refd)

                result = result.append(FeatureExtractor.rename_features(
                    features, ('my', ht, histo_key)))

        return result.astype(np.float64)
    
    def make_features(self):
        valid_runs = self.data_extractor.get_linear_data().index.tolist()

        processed = [int(f[:-7]) for f in os.listdir(self.paths['FEATURE_DIR'])]

        not_processed = set(valid_runs) - set(processed)

        args = zip([self] * len(not_processed), not_processed)

        # for arg in args:
        #     process_run(arg)

        pool = Pool(4)
        pool.map(process_run, args)

    def get_linear_data(self):
        with open(self.paths['LINEAR_DATA'], 'r') as ld:
            linear_data = json.load(ld)

        df = pd.DataFrame.from_dict(linear_data, orient='index').drop('rundb_data', axis=1)
        df.index = df.index.astype(np.int)

        rundb = {key: json.loads(value['rundb_data']) for key, value in linear_data.items()}
        rundf = pd.DataFrame.from_dict(rundb, orient='index')
        rundf.index = rundf.index.astype(np.int)
        rundf = rundf[FeatureExtractor.NUMS + FeatureExtractor.CATEGORICAL + FeatureExtractor.TIME]

        for col in FeatureExtractor.CATEGORICAL:
            rundf[col] = LabelEncoder().fit_transform(rundf[col])

        for col in FeatureExtractor.NUMS:
            rundf[col] = rundf[col].astype(np.float64)

        for col in FeatureExtractor.TIME:
            rundf[col] = rundf[col].map(FeatureExtractor.get_time)

        rundf['run_length'] = rundf['endtime'] - rundf['starttime']

        df = rundf.merge(df, left_index=True, right_index=True)

        df['reference'] = pd.Series(df.index, index=df.index).map(
            self.data_extractor.get_data_ref())

        rejected_runs = self.data_extractor.get_rejected_runs().keys()
        with open(self.paths['NO_STATS'], 'r') as infile:
            lines = infile.read().splitlines()
            no_stats = map(int, lines)

        df = df[~df.index.isin(no_stats + rejected_runs)]
        # print df[df.isnull().any(axis=1)].index.tolist()

        df = df.dropna()

        return df

    NUMS = ['avHltPhysRate', 'avL0PhysRate', 'avLumi', 'avMu', 'avPhysDeadTime', 'beamenergy',
            'beamgasTrigger', 'betaStar', 'endlumi', 'lumiTrigger', 'magnetCurrent',
            'nobiasTrigger', 'partitionid', 'run_state', 'tck', 'veloOpening']
    CATEGORICAL = ['LHCState', 'activity', 'magnetState', 'partitionname', 'program',
                   'programVersion', 'runtype', 'state', 'triggerConfiguration', 'veloPosition',
                   'destination']
    TIME = ['starttime', 'endtime']

    WEIRD_CATEGORICAL = ['LHCState', 'activity', 'runtype', 'triggerConfiguration', 'program',
                         'programVersion', 'destination', 'partitionname', 'veloPosition', 'state',
                         'magnetState']
    WEIRD_NUMS = ['beamenergy', 'beamgasTrigger', 'betaStar', 'lumiTrigger', 'magnetCurrent',
                  'nobiasTrigger', 'partitionid', 'run_state', 'tck', 'veloOpening']

    @staticmethod
    def get_time(x):
        return FeatureExtractor.to_timestamp(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))

    EXCLUDE_REFS = [
        '72d07236b6e45150631ddd8fe9a1c89d',  # reference low stats
        '24c6e0d6d6a17eeb2964150b4b2eaf72',  # reference low stats
    ]

    def make_linear_data(self):
        df = self.get_linear_data()

        begin, end = 174410, 187000

        df = df[df.flag.isin([0, 1])]
        df = df[(begin < df.index) & (df.index < end)]
        df = df[~df['reference'].isin(FeatureExtractor.EXCLUDE_REFS)]
        df['switch'] = (df['reference'] != df['reference'].shift(1)).astype(np.int).cumsum()
        short_switches = [switch for switch in df.switch.unique()
                          if (df.switch == switch).sum() < 20]
        df = df[~df.switch.isin(short_switches)]

        # df = df.loc[:, ~df.columns.isin(FeatureExtractor.WEIRD_CATEGORICAL +
        #                                 FeatureExtractor.WEIRD_NUMS)]

        df = df.rename(columns=lambda col: ('linear', col) if col != 'flag' else 'flag')

        df.to_pickle(self.paths['LINEAR_DATA_PREPARED'])

    EXCLUDED_TRAIN_HISTOS = ["RICH/RichDecodingErrors/decodingErrors"]

    def tune_histo_features(self, features):
        features = features[[col for col in features.columns if
                             not (col[0] == 'my' and col[2] in FeatureExtractor.EXCLUDED_TRAIN_HISTOS)]]

        for col in features:
            if col[0] == 'my' and col[-2:] == ('stats', 'runs'):
                features[col[:3] + ('stats', 'runs_ratio')] = \
                    features[col] / features[('linear', 'run_length')]

        return features

    def make_train_data(self):
        linear_data = self.data_extractor.get_linear_data()

        series_list = []
        run_numbers_list = []

        for run_number in linear_data.index:
            fpath = os.path.join(self.paths['FEATURE_DIR'], '{}.pickle'.format(run_number))

            if os.path.isfile(fpath):
                series = pd.read_pickle(fpath)
                series_list.append(series.loc[~series.index.duplicated(keep='first')])
                run_numbers_list.append(run_number)
            else:
                print run_number, linear_data.loc[run_number, 'flag']

        features = pd.DataFrame(series_list, index=run_numbers_list)

        # change for something normal
        is_reference = True
        for col in features:
            if col[-2:] == ('stats', 'kolmogorov'):
                is_reference &= features[col] == 0

        features = features[~is_reference]

        features = features.merge(linear_data, how='inner', left_index=True, right_index=True)

        features = self.tune_histo_features(features)

        features.to_pickle(self.paths['TRAIN_DATA'])

    def make_interactions(self):
        interactions = defaultdict(dict)
        rule = {
            ('func', 'func'): ['gauss'],
            ('func', 'distance'): ['check'],
            ('distance', 'distance'): ['line', 'line_gauss']
        }
        from roboshifter import Roboshifter

        for ht, features in Roboshifter.HISTO_FEATURES.items():
            for col1, col2 in Roboshifter.get_pairs(features):
                t = col1[0], col2[0]

                val = None
                if t in rule:
                    val = rule[t]
                elif ht == 'TEfficiency':
                    val = ['check' if col2[1] == 'variational' else 'gauss']

                if val is not None:
                    interactions[ht][col1, col2] = val

            for feature in features:
                if feature[0] == 'distance' or feature[1] == 'variational':
                    val = ['1d_distance']
                else:
                    val = ['1d_gauss']

                interactions[ht][(feature,)] = val

        with open(self.paths['INTERACTIONS'], 'wb') as outfile:
            pickle.dump(interactions, outfile)


def process_run(arg):
    self, run_number = arg

    path = os.path.join(self.paths['FEATURE_DIR'], '{}.pickle'.format(run_number))

    try:
        self.get_features(run_number).to_pickle(path)
    except Exception as e:
        print run_number
        print e


def process_bootstrap(arg):
    self, ref_hash, reference, hk, runs, index = arg

    try:
        rng = np.geomspace(runs.min(), runs.max(), 30, dtype=np.int64)
        feature_df = pd.DataFrame([self.bootstrap(reference, runn) for runn in rng], rng)

        with open(os.path.join(self.paths['BOOTSTRAP_RESULT_FILES'], index), 'wb') as outfile:
            pickle.dump((ref_hash, hk, feature_df), outfile)
    except Exception as e:
        print ref_hash, hk
        print e


if __name__ == '__main__':
    fe = FeatureExtractor()
    fe.make_train_data()