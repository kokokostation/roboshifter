import numpy as np
import pandas as pd
from scipy.stats import rv_discrete
from copy import deepcopy
from multiprocessing import Pool
from functools import partial
from sklearn.preprocessing import LabelEncoder
import datetime as dt
from collections import defaultdict
from sklearn.metrics import mean_squared_error as mse
from enum import IntEnum

from utils import Maybe, to_timestamp, inverse_dict


class Flag(IntEnum):
    TRAIN = 0
    TAIL = 1
    BAD = 2


class FeatureContainer:
    def __init__(self, flag=Flag.TRAIN, errors=None):
        if errors is None:
            errors = []

        self.data = pd.Series(dtype=np.float64)
        self.flag = flag
        self.histo_errors = defaultdict(list)
        self.errors = errors

    def add_features(self, series):
        self.data = self.data.append(series)

    def has_features(self):
        return not self.data.empty

    def set_flag(self, flag):
        self.flag = flag

    def add_histo_errors(self, histo_key, errors):
        self.histo_errors[histo_key].extend(errors)

    def add_errors(self, errors):
        self.errors.extend(errors)

    @staticmethod
    def concat(feature_containers):
        return pd.DataFrame(
            [[c.flag, c.histo_errors, c.errors] for c in feature_containers],
            index=feature_containers.index,
            columns=['flag', 'histo_errors', 'errors'],
        ).astype({
            'flag': np.int,
            'histo_errors': np.object,
            'errors': np.object
        })


def make_distribution(histo):
    vals = histo['vals'].astype(np.float64)
    s = vals.sum()

    if s == 0:
        raise ValueError()

    distr = vals / s

    if 'errs' in histo:
        errs = histo['errs'].astype(np.float64)
        normed_errs = errs / s
        bars = {'vals': distr, 'errs': normed_errs}
    else:
        bars = None

    return rv_discrete(values=(np.arange(distr.shape[0]), distr)), bars



def assemble_name(prefix, name):
    pn = [prefix, name]

    for i, item in enumerate(pn):
        if not isinstance(item, tuple):
            pn[i] = (item,)

    return pn[0] + pn[1]


def rename_features(features, prefix):
    features = features.copy()

    features.index = features.index.map(partial(assemble_name, prefix))

    return features


def statistical_distance(strong, weak):
    features = {
        'variational_distance': np.abs(strong.pk - weak.pk).sum(),
        'kolmogorov_smirnov_statistic': np.abs(strong.pk.cumsum() - weak.pk.cumsum()).max(),
    }

    return pd.Series(features)


def distribution_features(distr):
    mean, var, skew, kurt = distr.stats(moments='mvsk')

    features = {
        'mean': mean,
        'var': var,
        'skew': skew,
        'kurt': kurt,
    }

    return pd.Series(features, dtype=np.float64)


def regression_distance(runv, refv):
    abs_diff = np.abs(refv - runv)

    return pd.Series({
        'variational': np.sum(abs_diff),
        'kolmogorov': np.max(abs_diff),
        'mse': mse(runv, refv)
    })


def max_error_ratio(runh, refh):
    diffs = runh['vals'] - refh['vals']
    errors = np.sqrt(runh['errs'] ** 2 + refh['errs'] ** 2)
    index = errors != 0

    return np.max(np.abs(diffs[index] / errors[index]))


def value_features(runv, refv, prefix='stats'):
    runs, refs = runv.sum(), refv.sum()

    features = pd.Series({
        'runs': runs,
        'refs': refs,
        'refs - runs': refs - runs,
    })

    result = pd.concat([features, regression_distance(runv, refv)])

    return Maybe(value=rename_features(result, prefix))


def histo_features(runh, refh, const_std_features=True):
    result = pd.Series()

    try:
        (rund, run_bars), (refd, ref_bars) = map(make_distribution, [runh, refh])
    except ValueError:
        return Maybe(error_message=["Integral for histogram equals zero"])

    runf = distribution_features(rund)
    reff = distribution_features(refd)
    distancef = statistical_distance(refd, rund)
    funcf = reff - runf

    flavors = [(runf, 'run'), (distancef, 'distance'), (funcf, 'func')]
    if const_std_features:
        flavors.append((reff, 'ref'))

    for features, flavor in flavors:
        result = result.append(rename_features(features, flavor))

    if run_bars is not None:
        result = result.append(pd.Series({('alarm', 'max_error_ratio'):
                                          max_error_ratio(run_bars, ref_bars)}))

    return Maybe(value=result)


def th1d_features(runh, refh):
    return Maybe.concat([histo_features(runh, refh),
                         value_features(runh['vals'], refh['vals']),
                         Maybe(value=pd.Series({('alarm', 'mean'): runh['mean']}))])


def get_efficiency(teff):
    result = deepcopy(teff['passed'])
    total = teff['total']['vals'].copy()
    total[total == 0] = 1
    result['vals'] /= total

    return result, teff['total']['vals'] + teff['passed']['vals']


def tefficiency_features(runh, refh):
    (run_eff, run_stats), (ref_eff, ref_stats) = map(get_efficiency, [runh, refh])

    return Maybe.concat([histo_features(run_eff, ref_eff), value_features(run_stats, ref_stats),
                         value_features(run_eff['vals'], ref_eff['vals'], 'eff')])


def tprofile_features(runh, refh):
    distance = Maybe(value=rename_features(regression_distance(runh['vals'], refh['vals']),
                                           'distance'))

    alarm = Maybe(value=rename_features(pd.Series({
        'max_abs': np.max(np.abs(runh['vals'])),
        'max_error_ratio': max_error_ratio(runh, refh),
    }), 'alarm'))

    result = Maybe.concat([distance, alarm, value_features(runh['entries'], refh['entries']),
                           histo_features({'vals': runh['errs']}, {'vals': refh['errs']})])

    return result


def weird_tprofile_features(runh, refh):
    mask = (runh['vals'] != 0) & (refh['vals'] != 0)

    s = mask.sum()

    if s == 0:
        return Maybe(error_message=['Too many zeros in graph to process features'])
    else:
        for d in [runh, refh]:
            for key, value in d.items():
                if isinstance(value, np.ndarray):
                    d[key] = value[mask]

        return Maybe.concat([Maybe(value=pd.Series({('zeros', 'vals'): s})),
                             tprofile_features(runh, refh)])


def decoding_errors_features(runh, refh):
    return Maybe(value=pd.Series({('decoding', 'errors'): runh['vals'].sum()}))


def get_time(x):
    return to_timestamp(dt.datetime.strptime(x, '%Y-%m-%dT%H:%M:%S'))


class FeatureExtractor:
    HANDLERS = {
        'TH1D': th1d_features,
        'TH2D': lambda runh, refh: Maybe(error_message=['No handler for TH2D available']),  # it's sad
        'TEfficiency': tefficiency_features,
        'TProfile': tprofile_features,
        'WeirdTProfile': weird_tprofile_features,
        'WeirdMuTProfile': weird_tprofile_features,
        'DecodingErrors': decoding_errors_features
    }

    WEIRD_HT = {
        'WeirdTProfile': [
            'RICH/RiLongTrkEff/All/effVChi2PDOF',
            'Velo/VeloTrackMonitor/Pseudoefficiency_per_sensor_vs_sensorID',
        ],
        'WeirdMuTProfile': [
            'MuIDLambdaPlot/pion/Prof_eff',
            'MuIDLambdaPlot/proton/Prof_eff'
        ],
        'DecodingErrors': [
            'RICH/RichDecodingErrors/decodingErrors'
        ]
    }

    NUMS = ['avHltPhysRate', 'avL0PhysRate', 'avLumi', 'avMu', 'avPhysDeadTime', 'beamenergy',
            'beamgasTrigger', 'betaStar', 'endlumi', 'lumiTrigger', 'magnetCurrent',
            'nobiasTrigger', 'partitionid', 'run_state', 'tck', 'veloOpening']
    CATEGORICAL = ['LHCState', 'activity', 'magnetState', 'partitionname', 'program',
                   'programVersion', 'runtype', 'state', 'triggerConfiguration', 'veloPosition',
                   'destination']
    TIME = ['starttime', 'endtime']

    def __init__(self, collector, njobs):
        self.collector = collector
        self.njobs = njobs

    def tune_histo(self, ht, runh):
        runh = deepcopy(runh)

        if ht == 'TH1D':
            runh['vals'] = np.maximum(0, runh['vals'])

        return runh

    def get_histo_type(self, histo_key, data):
        histo_types = inverse_dict(FeatureExtractor.WEIRD_HT)

        return histo_types.get(histo_key, data['type'])

    def get_features(self, run_number):
        refh = self.collector.get_reference(run_number)
        runh = self.collector.get_run(run_number)

        present = Maybe.concat_id([runh, refh])
        if present.is_error():
            return FeatureContainer(Flag.BAD, present.error_message)

        result = FeatureContainer()

        for histo_key in self.collector.get_histo_keys():
            histo_present = Maybe.concat_id([a[histo_key] for a in present.value])

            if histo_present.is_error():
                result.add_histo_errors(histo_key, histo_present.error_message)
                result.set_flag(Flag.TAIL)
            else:
                runh, refh = histo_present.value
                ht = self.get_histo_type(histo_key, runh)

                # !!!!!!!!!!!!!!!!!!!!!!!!!!!
                if ht == 'TH2D':
                    continue

                handler = FeatureExtractor.HANDLERS[ht]

                runh, refh = map(partial(self.tune_histo, ht), [runh, refh])

                features = handler(runh, refh)

                if features.is_error():
                    result.add_histo_errors(histo_key, features.error_message)
                    result.set_flag(Flag.TAIL)
                else:
                    renamed = rename_features(features.value, ('my', ht, histo_key))

                    result.add_features(renamed)

        if not result.has_features():
            result.add_errors(['No histo features at all.'])
            result.set_flag(Flag.BAD)

        return result

    def make_features(self, run_numbers):
        args = zip([self] * len(run_numbers), run_numbers)

        # for arg in args:
        #     process_run(arg)

        pool = Pool(self.njobs)
        pool.map(process_run, args)
        pool.close()
        pool.join()

    def get_linear_data(self):
        linear_data = self.collector.get_linear_data()

        df = pd.DataFrame.from_dict(linear_data, orient='index').drop('rundb_data', axis=1)
        df.index = df.index.astype(np.int)

        rundb = {key: value['rundb_data'] for key, value in linear_data.items()}
        rundf = pd.DataFrame.from_dict(rundb, orient='index')
        rundf.index = rundf.index.astype(np.int)
        rundf = rundf[FeatureExtractor.NUMS + FeatureExtractor.CATEGORICAL + FeatureExtractor.TIME]

        for col in FeatureExtractor.CATEGORICAL:
            rundf[col] = LabelEncoder().fit_transform(rundf[col])

        for col in FeatureExtractor.NUMS:
            rundf[col] = rundf[col].astype(np.float64)

        for col in FeatureExtractor.TIME:
            rundf[col] = rundf[col].map(get_time)

        rundf['run_length'] = rundf['endtime'] - rundf['starttime']
        rundf.loc[rundf['run_length'] < 0, 'run_length'] = np.nan

        df = rundf.merge(df, left_index=True, right_index=True)

        df['reference'] = pd.Series(df.index, index=df.index).map(self.collector.get_data_ref())

        df['switch'] = (df['reference'] != df['reference'].shift(1)).astype(np.int).cumsum()

        df = df.rename(columns=lambda col: ('linear', col) if col != 'flag' else 'flag')

        return df


def process_run(arg):
    self, run_number = arg

    self.collector.write_run_features(self.get_features(run_number), run_number)


def get_train_features(X):
    fit_index = X[('info', 'flag')] == Flag.TRAIN
    train_X = X.loc[fit_index, 'features']

    return fit_index, train_X