import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, HuberRegressor
from collections import defaultdict
from scipy.stats import pearsonr
from functools import partial
from multiprocessing import Manager, Pool
from functools import wraps
from standard_scaler import RobustScaler, StandardScaler

from roboshifter_utils import mahalanobis, reindex, normalize_df, mcd, huber, \
    get_linear_features, get_histo_features, RoboshifterError
from report_maker import ReportMaker
from utils import NullStripper, TailHandler
from threshold_classifier import ThresholdClassifier
from features import get_train_features
from utils import plain_batch_generator


def prepare_gauss(cov, df, key, cd, data, fit):
    return Roboshifter.prepare_normalize(df, data['stat_flag'], data['switch'],
                                         Roboshifter.normalize_ellipse, cov)


def prepare_check(df, key, cd, data, fit):
    if fit:
        X, y = Roboshifter.prepare_huber(df, data['stat_flag'])
        X = np.abs(X)

        cd['checks'][key] = Roboshifter.fit_huber(X, y)

    check = cd['checks'][key]

    return Roboshifter.prepare_normalize(df, data['stat_flag'], data['switch'],
                                         partial(Roboshifter.check_normalizer, check), False)


def prepare_line_helper(df, key, cd, data, fit):
    if fit and key not in cd['lines']:
        X, y = Roboshifter.prepare_huber(df, data['stat_flag'])

        cd['lines'][key[:2]] = Roboshifter.fit_huber(X, y)


def prepare_line_gauss(df, key, cd, data, fit):
    prepare_line_helper(df, key, cd, data, fit)

    x, y = df.columns
    params = cd['lines'][key[:2]]

    df[y] -= params['a'] * df[x] + params['b']

    return prepare_gauss(True, df, key, cd, data, fit)


def prepare_line(df, key, cd, data, fit):
    prepare_line_helper(df, key, cd, data, fit)

    line = cd['lines'][key[:2]]

    projection = np.dot(df - np.array([0, line['b']]), np.array([1, line['a']]).T) / \
                 np.sqrt(1 + line['a'] ** 2)

    return prepare_1d_distance(pd.DataFrame(projection, index=df.index),
                               key, cd, data, fit)


def prepare_1d_gauss(df, key, cd, data, fit):
    return Roboshifter.prepare_1d(df, data['stat_flag'], data['switch'], np.abs)


def prepare_1d_distance(df, key, cd, data, fit):
    return Roboshifter.prepare_1d(df, data['stat_flag'], data['switch'], lambda x: x)


def fit_gaussian(df, key, cd, data, fit):
    cd['gaussians'][key] = mcd(df.loc[data['stat_flag'] == 0])


def predict_mahalanobis(df, key, cd, data, fit):
    return mahalanobis(df, cd['gaussians'][key])


def predict_euclidean(df, key, cd, data, fit):
    return np.sqrt((df ** 2).sum(axis=1))


def handle_interactions_helper(pack):
    features, cd, data, fit, handlers, args = pack

    for key, df, interaction in args:
        handler = handlers.get(interaction, lambda *x: x[0])

        new_key = key + (interaction,)

        ns = NullStripper(df)
        in_df, in_data = ns.get_notnull(df, data)

        result = handler(in_df, new_key, cd, in_data, fit)

        if result is not None:
            features[new_key] = ns.make_data(result)


def predict(f):
    @wraps(f)
    def wrapper(self, X, y=None):
        result = f(self, X, y, y is not None)

        return result

    return wrapper


class Roboshifter:
    HISTO_FEATURES = {
        'TH1D': [
            ('func', 'mean'),
            ('func', 'var'),
            ('func', 'skew'),
            ('func', 'kurt'),

            ('distance', 'variational_distance'),
            ('distance', 'kolmogorov_smirnov_statistic')
        ],
        'TProfile': [
            ('distance', 'variational'),
            ('distance', 'kolmogorov'),
        ],
        'WeirdTProfile': [
            ('distance', 'variational')
        ],
        'WeirdMuTProfile': [
            ('func', 'mean'),

            ('distance', 'variational'),
            ('distance', 'kolmogorov'),
        ],
        'TEfficiency': [
            ('func', 'mean'),
            ('func', 'var'),
            ('func', 'skew'),
            ('func', 'kurt'),

            ('eff', 'refs - runs'),

            ('eff', 'variational'),
        ]
    }

    CONTINUOUS_LINEAR = [
        'dq_pi0_resolution_err',
        'dq_jpsi_mass_err',
        'dq_jpsi_yield_err',
        'dq_jpsi_resolution_err',
        'dq_jpsi_mass',
        'dq_mean_number_of_OT_times_err',
        'dq_mean_number_of_PV_err',
        'dq_pi0_mass',
        'dq_jpsi_yield',
        'dq_pi0_mass_err',
        'dq_jpsi_resolution',
        'dq_mean_number_of_OT_times',
        'dq_pi0_resolution',
        'dq_mean_number_of_PV',
        'avMu',
        'endlumi',
        'avPhysDeadTime',
        'avLumi',
        'avHltPhysRate',
        'avL0PhysRate'
    ]

    SELECTED_LINEAR = [
        'dq_jpsi_yield_err',
        'dq_pi0_mass_err',
        'avHltPhysRate'
    ]

    INTERACTION_PREPARE = {
        'mixture_gauss': partial(prepare_gauss, True),
        'gauss': partial(prepare_gauss, False),
        'check': prepare_check,
        'line': prepare_line,
        'line_gauss': prepare_line_gauss,
        '1d_gauss': prepare_1d_gauss,
        '1d_distance': prepare_1d_distance
    }

    INTERACTION_FIT = {
        'gauss': fit_gaussian,
        'line_gauss': fit_gaussian,
        'check': fit_gaussian
    }

    INTERACTION_PREDICT = {
        'gauss': predict_mahalanobis,
        'check': predict_mahalanobis,
        'line_gauss': predict_mahalanobis,
        'mixture_gauss': predict_euclidean
    }

    DEFAULT_INTERACTIONS_RULE = {
        ('func', 'func'): ['gauss'],
        ('func', 'distance'): ['check'],
        ('distance', 'distance'): ['line', 'line_gauss']
    }

    PREDICT_KEYS = ReportMaker.ROBOSHIFTER_PREDICT_KEYS + [
        'stat_proba',
        'linear_proba'
    ]

    MIN_SWITCH_LENGTH = 20
    MAX_REFERENCE_STATS_RATIO = 5

    @staticmethod
    def get_pairs(l):
        return [(a1, a2) for i, a1 in enumerate(l) for a2 in l[i + 1:]]

    @staticmethod
    def make_ht_interactions():
        interactions = defaultdict(dict)

        for ht, features in Roboshifter.HISTO_FEATURES.items():
            for col1, col2 in Roboshifter.get_pairs(features):
                t = col1[0], col2[0]

                val = Roboshifter.DEFAULT_INTERACTIONS_RULE.get(t)
                if val is None and ht == 'TEfficiency':
                    val = ['check' if col2[1] == 'variational' else 'gauss']

                if val is not None:
                    interactions[ht][col1, col2] = val

            for feature in features:
                if feature[0] == 'distance' or feature[1] == 'variational':
                    val = ['1d_distance']
                else:
                    val = ['1d_gauss']

                interactions[ht][(feature,)] = val

        return interactions

    def __init__(self, interactions=None, mean_window=5, std_window=10, pollution_rate=0.1,
                 feature_filter_threshold=0.9, feature_threshold_percentile=0.95, njobs=4,
                 roc_auc_threshold=0.6, verbose=False):
        if interactions is None:
            from collector import Collector

            interactions = Collector().get_interactions()

        self.interactions = interactions
        self.mean_window = mean_window
        self.std_window = std_window
        self.pollution_rate = pollution_rate
        self.feature_filter_threshold = feature_filter_threshold
        self.feature_threshold_percentile = feature_threshold_percentile
        self.njobs = njobs
        self.roc_auc_threshold = roc_auc_threshold
        self.verbose = verbose

        self.renew()

        self.prepare_selected_linear_filter = partial(self.prepare_linear_filter,
                                                      Roboshifter.SELECTED_LINEAR)
        self.prepare_all_linear_filter = partial(self.prepare_linear_filter,
                                                 Roboshifter.CONTINUOUS_LINEAR)

    def renew(self):
        self.classifier_data = {}

        self.info = {}

        self.full_fit_X = None
        self.fit_X = None
        self.fit_y = None

    def rolling(self, key, window, switch, series, df):
        result = Roboshifter.iterate_switches(
            series, switch,
            lambda curr_switch: curr_switch.rolling(window=window).apply(
                lambda x: huber(x)[key]))

        return reindex(df.index, result)

    def prepare_features(self, data, exclude=None, local_std=True, local_mean=True):
        if self.verbose:
            print 'preparing features'

        if exclude is None:
            exclude = []

        df = data['X']
        stat_flag = data['ss']['stat_flag']

        strong_df = df[stat_flag == 0]
        switch = data['ss']['switch'][stat_flag == 0]

        features = pd.DataFrame()

        for col in df.columns:
            if col not in exclude:
                notnull = strong_df[col].notnull()

                nn_switch = switch[notnull]
                nn_strong_df_col = strong_df[col].loc[notnull]

                if local_std:
                    rolling_std = self.rolling('s', self.std_window, nn_switch,
                                               nn_strong_df_col, df)
                else:
                    rolling_std = 1

                if local_mean:
                    rolling_mean = self.rolling('mu', self.mean_window, nn_switch,
                                                nn_strong_df_col, df)
                else:
                    rolling_mean = 0

                feature = (df[col] - rolling_mean) / rolling_std

                features[col] = feature
            else:
                features[col] = df[col]

        if self.verbose:
            print 'finished preparing features'

        return features

    def get_threshold(self, proba, flag, outlier_quantile):
        df = pd.concat([proba, flag], axis=1, keys=['proba', 'flag']).sort_values('proba')
        index = (df.flag.astype(np.float64).cumsum() / df.flag.sum()) \
            .searchsorted([outlier_quantile], 'right')

        return df.iloc[index[0]].proba

    def predict_filter(self, X, preparer, prefix, y=None, fts_prefix=None):
        if self.verbose:
            print 'predicting {} filter'.format(prefix)

        fit = y is not None
        if fts_prefix is None:
            fts_prefix = prefix

        all_X = preparer(X, y)

        ns = NullStripper(all_X)
        X = ns.get_notnull(all_X)

        ndf = pd.DataFrame(index=X.index)

        ndf['proba'] = self.classifier_data['filters'][fts_prefix].predict_proba(X)[:, 1]
        ndf['predict'] = self.classifier_data['filters'][fts_prefix].predict(X)

        fts = self.classifier_data['filter_thresholds']
        if fit and fts_prefix not in fts:
            fts[fts_prefix] = self.get_threshold(ndf.proba, y, self.pollution_rate)
        ndf['flag'] = (ndf.proba > fts[fts_prefix]).astype(np.int64)

        ndf = ns.make_data(ndf)

        for col in ndf:
            name = '{}_{}'.format(prefix, col)
            self.info[fit][name] = ndf[col].rename(name)

        if self.verbose:
            print 'finished predicting {} filter'.format(prefix)

    def fit_filter(self, preparer, model, name):
        if self.verbose:
            print 'fitting {} filter'.format(name)

        X, y = preparer(self.fit_X, self.fit_y), self.fit_y

        self.classifier_data['filters'][name] = model
        self.classifier_data['filters'][name].fit(X, y)

        self.predict_filter(self.fit_X, preparer, name, self.fit_y)

        if self.verbose:
            print 'finished fitting {} filter'.format(name)

    def prepare_stat_filter(self, X, y=None):
        ndf = X[[('linear', 'run_length'), ('integral', 'runssum')]]

        return normalize_df(ndf)

    def fit_stat_filter(self):
        self.fit_filter(self.prepare_stat_filter,
                        LogisticRegression(class_weight='balanced'),
                        'stat')

        self.predict_filter(self.fit_tail_X, self.prepare_stat_filter, 'tail_stat',
                            self.fit_tail_y, 'stat')

    def predict_stat_filter(self, X, y=None):
        self.predict_filter(X, self.prepare_stat_filter, 'stat', y)

    def get_fit_stat_flag(self, tail):
        key, y = ('tail_stat_flag', self.fit_tail_y) if tail else ('stat_flag', self.fit_y)
        stat_flag = self.info[True][key].copy()
        stat_flag[y == 1] = 1

        return stat_flag

    def make_prepare_data(self, X, tail):
        return {
            'X': X,
            'ss': pd.DataFrame({'stat_flag': self.get_fit_stat_flag(tail),
                                'switch': X[('linear', 'switch')]})
        }

    def make_tail_handler(self, X):
        th = TailHandler()
        data = th.make_tail({
            'X': self.fit_tail_X,
            'ss': pd.DataFrame({'stat_flag': self.get_fit_stat_flag(True),
                                'switch': self.fit_tail_X[('linear', 'switch')]})
        }, {
            'X': X,
            'ss': pd.DataFrame({'stat_flag': self.info[False]['stat_flag'],
                                'switch': X[('linear', 'switch')]})
        })

        return data, th

    def prepare_linear_filter(self, cols, X, y=None):
        fit = y is not None

        if fit:
            data = self.make_prepare_data(X, False)
        else:
            data, th = self.make_tail_handler(X)

        data['X'] = get_linear_features(data['X'], cols)

        df = np.abs(self.prepare_features(data))

        if not fit:
            df = th.cut_tail(df)

        return df

    def fit_linear_filter(self):
        self.fit_filter(self.prepare_selected_linear_filter,
                        LogisticRegression(class_weight='balanced'),
                        'linear')

    def fit_linear_thresholds(self):
        df = self.prepare_all_linear_filter(self.fit_X, self.fit_y)

        model = ThresholdClassifier(self.feature_threshold_percentile)
        model.fit(df)

        self.classifier_data['thresholds']['linear'] = model

    def predict_linear_filter(self, X, y=None):
        self.predict_filter(X, self.prepare_selected_linear_filter, 'linear', y)

    @predict
    def predict_linear_thresholds(self, X, y, fit):
        df = self.prepare_all_linear_filter(X, y)

        model = self.classifier_data['thresholds']['linear']
        self.info[fit]['linear_prediction'] = model.predict(df)

    def fit_feature_filter(self, splitted):
        result = {}

        for hk, features in splitted.items():
            mask = {col: True for col in features}

            for i, col1 in enumerate(features):
                if np.abs(features[col1].std() / features[col1].mean()) < 0.01:
                    mask[col1] = False

                if mask[col1]:
                    for col2 in features.columns[i + 1:]:
                        if mask[col2]:
                            r = Roboshifter.pearsonr(features[col1], features[col2])

                            if r > self.feature_filter_threshold:
                                mask[col2] = False

            result[hk] = [col for col, val in mask.items() if val]

        self.classifier_data['feature_filter'] = result

    def filter_features(self, splitted):
        ff = self.classifier_data['feature_filter']
        result = {}

        for ht, df in splitted.items():
            result[ht] = df[ff[ht]]

        return result

    PACK_SIZE = 100

    def handle_interactions(self, generator, handlers, fit, data=None):
        manager = Manager()

        features = manager.dict()
        cd = {key: manager.dict(value) for key, value in self.classifier_data.items()}

        packs = plain_batch_generator(
            generator,
            lambda args: (features, cd, data, fit, handlers, args),
            Roboshifter.PACK_SIZE
        )

        pool = Pool(self.njobs)

        for batch in plain_batch_generator(packs, lambda x: x, self.njobs * 4):
            # for b in batch:
            #     handle_interactions_helper(b)

            pool.map(handle_interactions_helper, batch)

        pool.close()
        pool.join()

        self.classifier_data = {key: dict(value) for key, value in cd.items()}

        return dict(features)

    def prepare_generator(self, data):
        for key, interactions in self.interactions.items():
            hk, cols = key
            if hk in data:
                df = data[hk]

                if all(col in df for col in cols):
                    for interaction in interactions:
                        yield key, df[list(cols)], interaction

    def fit_predict_generator(self, data):
        for key, df in data.items():
            yield key[:2], df, key[2]

    @staticmethod
    def ellipse_callback(ellipse, strong_index, cov):
        params = mcd(ellipse[strong_index])

        result = ellipse - params['center']

        if cov:
            w, v = np.linalg.eig(params['cov'])

            A = np.dot(np.sqrt(np.diag(1. / w)), v.T)

            result = np.dot(A, result.T).T

        return pd.DataFrame(result, columns=ellipse.columns, index=ellipse.index)

    @staticmethod
    def gaussian_callback(gaussian, strong_index, cov):
        params = huber(gaussian[strong_index])

        result = gaussian - params['mu']

        if cov:
            result /= params['s']

        return result

    @staticmethod
    def pearsonr(x, y):
        return np.abs(pearsonr(x, y)[0])

    @staticmethod
    def normalize_location(data, stat_flag, threshold, callback, cov):
        if (stat_flag == 0).sum() < threshold:
            return data
        else:
            return callback(data, stat_flag == 0, cov)

    @staticmethod
    def normalize_ellipse(ellipse, stat_flag, cov):
        return Roboshifter.normalize_location(ellipse, stat_flag, Roboshifter.MIN_SWITCH_LENGTH,
                                              Roboshifter.ellipse_callback, cov)

    @staticmethod
    def normalize_gauss(gauss, stat_flag, cov):
        return Roboshifter.normalize_location(gauss, stat_flag, Roboshifter.MIN_SWITCH_LENGTH,
                                              Roboshifter.gaussian_callback, cov)

    @staticmethod
    def iterate_switches(df, switches, callback):
        return pd.concat([callback(df.loc[switches == switch]) for switch in switches.unique()],
                         axis=0)

    @staticmethod
    def prepare_normalize(df, stat_flag, switches, normalizer, cov):
        return Roboshifter.iterate_switches(
            df, switches,
            lambda curr_switch: normalizer(curr_switch, stat_flag.loc[curr_switch.index], cov))

    @staticmethod
    def prepare_huber(df, stat_flag):
        fit_df = df.loc[stat_flag == 0]

        X, y = fit_df.as_matrix().T
        return X.reshape((-1, 1)), y

    @staticmethod
    def check_normalizer(check, df, stat_flag, cov):
        df = df.copy()

        x, y = df.columns

        df[y] -= check['a'] * np.abs(df[x]) + check['b']
        df = Roboshifter.normalize_ellipse(df, stat_flag, cov)

        return df

    @staticmethod
    def prepare_1d(df, stat_flag, switches, callback):
        result = callback(
            Roboshifter.prepare_normalize(df, stat_flag, switches, Roboshifter.normalize_gauss,
                                          False))

        return result[result.columns[0]]

    @staticmethod
    def fit_huber(X, y):
        model = HuberRegressor()

        model.fit(X, y)

        return {
            'a': model.coef_[0],
            'b': model.intercept_
        }

    def split_features(self, features):
        histo_mapping = defaultdict(list)
        col_mapping = {}

        for col in features.columns:
            if col[0] == 'my':
                histo_mapping[col[2]].append(col)
                col_mapping[col] = col[3:]

        result = {}

        for histo in histo_mapping.keys():
            result[histo] = features[histo_mapping[histo]].rename(columns=col_mapping)

        return result

    def prepare_ee(self, data, fit):
        if self.verbose:
            print 'preparing figures'

        df = get_histo_features(data['X'], Roboshifter.HISTO_FEATURES)

        splitted = self.split_features(df)

        if fit:
            self.fit_feature_filter(splitted)

        splitted = self.filter_features(splitted)

        features = self.handle_interactions(self.prepare_generator(splitted),
                                            Roboshifter.INTERACTION_PREPARE,
                                            fit,
                                            data['ss'])

        if self.verbose:
            print 'finished preparing figures'

        return features

    def fit_ee(self):
        if self.verbose:
            print 'fitting figures'

        data = self.make_prepare_data(self.fit_X, False)

        features = self.prepare_ee(data, True)

        self.handle_interactions(self.fit_predict_generator(features),
                                 Roboshifter.INTERACTION_FIT,
                                 True,
                                 data['ss'])

        scores = self.predict_ee_scores(features, True)

        model = StandardScaler()
        model.fit(scores[self.fit_y == 0])
        self.classifier_data['scalers']['feature'] = model

        self.predict_ee(self.fit_X, self.fit_y)

        if self.verbose:
            print 'finished fitting figures'

    def predict_ee_scores(self, features, fit):
        scores = self.handle_interactions(self.fit_predict_generator(features),
                                          Roboshifter.INTERACTION_PREDICT, fit)
        scores = pd.DataFrame(scores)

        self.info[fit]['ee_scores'] = scores

        return scores

    @predict
    def predict_ee(self, X, y, fit):
        if self.verbose:
            print 'predicting figures'

        if fit:
            scores = self.info[True]['ee_scores']
        else:
            data, th = self.make_tail_handler(X)

            features = self.prepare_ee(data, False)
            features = th.cut_tail_dict(features)

            scores = self.predict_ee_scores(features, False)

        ft_preds = self.classifier_data['scalers']['feature'].transform(scores)

        result = defaultdict(lambda: pd.Series(0., X.index))
        counts = defaultdict(int)

        for key in ft_preds:
            result[key[0]] += ft_preds[key]
            counts[key[0]] += 1

        for key in result.keys():
            result[key] /= counts[key]

        self.info[fit]['histo_score'] = pd.DataFrame(result)

        if self.verbose:
            print 'finished predicting figures'

    def init_fit(self, X, y):
        self.renew()

        self.fit_tail_X = X['features'].copy()
        self.fit_tail_y = y.copy()

        fit_index, train_X = get_train_features(X)
        last_switch = train_X[('linear', 'switch')] == train_X[('linear', 'switch')].iloc[-1]
        if last_switch.sum() < Roboshifter.MIN_SWITCH_LENGTH:
            fit_index &= ~last_switch

        self.fit_X = X.loc[fit_index, 'features'].copy()
        self.fit_y = y.loc[fit_index].copy()

        if self.fit_X.empty:
            raise RoboshifterError('No valid data to fit')

        self.info[True] = {}

        for key in ['filters', 'filter_thresholds', 'feature_thresholds',
                    'checks', 'lines', 'gaussians', 'final', 'thresholds',
                    'feature_filter', 'scalers']:
            self.classifier_data[key] = {}

    def fit(self, X, y):
        self.init_fit(X, y)

        self.fit_stat_filter()
        self.fit_linear_filter()
        self.fit_linear_thresholds()
        self.fit_ee()

        return self

    def init_predict(self, X):
        self.info[False] = {}

        return X['features'].copy()

    def get_predict_keys(self, fit=False, keys=None):
        if keys is None:
            keys = Roboshifter.PREDICT_KEYS

        return {key: self.info[fit][key] for key in keys}

    def predict(self, X):
        X = self.init_predict(X)

        self.predict_stat_filter(X)
        self.predict_linear_filter(X)
        self.predict_linear_thresholds(X)
        self.predict_ee(X)

        return self.get_predict_keys()