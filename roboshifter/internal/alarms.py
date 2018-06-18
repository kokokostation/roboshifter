import pandas as pd
from functools import partial

from threshold_classifier import ThresholdClassifier
from roboshifter_utils import get_linear_features, get_histo_feature
from features import get_train_features
from utils import nan_gt


def get_histo_feature_helper(feature, name, fltr, X):
    return get_histo_feature(X, fltr.alarms[name].index, feature)


def prepare_mean(getter, name, fltr, X):
    data = getter(X, fltr.alarms[name].index)

    return (data - fltr.alarms[name]['mean'].as_matrix()).abs()


def prepare(name, fltr, X):
    handler = AlarmFilter.INTERACTION_PREPARE[name]
    data = handler(name, fltr, X)

    return data


def fit_mean(name, fltr, X):
    data = prepare(name, fltr, X)

    tc = ThresholdClassifier(fltr.quantile)
    tc.fit(data)

    fltr.cd[name] = tc


def predict_error_bars(name, fltr, X):
    data = get_histo_feature_helper(('alarm', 'max_error_ratio'), name, fltr, X)

    for col in data:
        data[col] = nan_gt(data[col], 1)

    return data


def predict_fluctuations(name, fltr, X):
    data = get_histo_feature_helper(('alarm', 'max_abs'), name, fltr, X)

    for col, c in zip(data, fltr.alarms[name]['std']):
        data[col] = nan_gt(data[col], c)

    return data


def predict_mean_std(name, fltr, X):
    data = get_histo_feature_helper(('alarm', 'mean'), name, fltr, X)

    alarms = fltr.alarms[name]
    for col, mean, std in zip(data, alarms['mean'], alarms['mean_std']):
        data[col] = nan_gt((data[col] - mean).abs(), std)

    return data


def predict_mean(name, fltr, X):
    tc = fltr.cd[name]
    data = prepare(name, fltr, X)

    return tc.predict(data)


def correct_cols(name, data):
    index = {
        'my': 2,
        'linear': 1
    }

    data.columns = ['{}::{}'.format(name, col[index[col[0]]]) for col in data.columns]

    return data


class AlarmFilter:
    ALARMS = {
        'error_bars': [
            {'histo': "Track/TrackVertexMonitor/track IP X vs phi"},
            {'histo': "Track/TrackVertexMonitor/track IP Y vs eta"},
            {'histo': "Track/TrackVertexMonitor/track IP X vs eta"},
            {'histo': "Track/TrackVertexMonitor/track IP Y vs phi"},
            {'histo': "Track/TrackVertexMonitor/fast track IP Y vs phi"},
            {'histo': "Track/TrackVertexMonitor/fast track IP Y vs eta"},
            {'histo': "Track/TrackVertexMonitor/fast track IP X vs eta"},
            {'histo': "Track/TrackVertexMonitor/fast track IP X vs phi"},
            {'histo': "MuIDJpsiPlot/probe/Prof_eff"}
        ],
        'error_bars_ignore_low_stats': [
            {'histo': "MuIDLambdaPlot/pion/Prof_eff"},
            {'histo': "MuIDLambdaPlot/proton/Prof_eff"}
        ],
        'mean': [
            {'histo': "MuIDLambdaPlot/proton/Chg", 'mean': 0},
            {'histo': "MuIDJpsiPlot/probe/Chg", 'mean': 0},
            {'histo': "Track/TrackMonitor/Long/VeloPhiResidual", 'mean': 0},
            {'histo': "Track/TrackMonitor/Long/VeloRResidual", 'mean': 0},
            {'histo': "Track/TrackMonitor/Long/ITResidual", 'mean': 0},
            {'histo': "Track/TrackMonitor/Long/TTResidual", 'mean': 0},
            {'histo': "Track/TrackMonitor/Long/OTResidual", 'mean': 0},
            {'histo': "MuIDLambdaPlot/IM", 'mean': 1115},
            {'histo': "MuIDJpsiPlot/IM", 'mean': 3063},
            {'histo': "MakeDstSel.DaughtersPlots/Mass(D0)", 'mean': 1864.84},
            {'histo': "JpsiDQMonitorPlots.MassPlotTool/peak/M_J_psi_1S", 'mean': 3096.916}
        ],
        'mean_std': [
            {'histo': "Track/TrackPV2HalfAlignMonitor/Left-Right PV delta x", 'mean': 0, 'mean_std': 0.02},
            {'histo': "Track/TrackPV2HalfAlignMonitor/Left-Right PV delta y", 'mean': 0, 'mean_std': 0.02},
            {'histo': "Track/TrackPV2HalfAlignMonitor/Left-Right PV delta z", 'mean': 0, 'mean_std': 0.1},
            {'histo': "Track/TrackVeloOverlapMonitor/residualCPhi", 'mean': 0, 'mean_std': 0.03},
            {'histo': "Track/TrackVeloOverlapMonitor/overlapResidualPhi", 'mean': 0, 'mean_std': 0.03},
            {'histo': "Track/TrackVeloOverlapMonitor/residualAPhi", 'mean': 0, 'mean_std': 0.03},
            {'histo': "Track/TrackVeloOverlapMonitor/residualAR", 'mean': 0, 'mean_std': 0.008},
            {'histo': "Track/TrackVeloOverlapMonitor/residualCR", 'mean': 0, 'mean_std': 0.008},
            {'histo': "Track/TrackVeloOverlapMonitor/overlapResidualR", 'mean': 0, 'mean_std': 0.008},
            {'histo': "Track/TTTrackMonitor/TTaX/Overlap residual", 'mean': 0, 'mean_std': 0.05},
            {'histo': "Track/TTTrackMonitor/TTaU/Overlap residual", 'mean': 0, 'mean_std': 0.05},
            {'histo': "Track/TTTrackMonitor/TTbX/Overlap residual", 'mean': 0, 'mean_std': 0.05},
            {'histo': "Track/TTTrackMonitor/TTbV/Overlap residual", 'mean': 0, 'mean_std': 0.05},
        ],
        'fluctuations_around_constant': [
            {'histo': "Track/TrackVertexMonitor/fast track IP X vs phi", 'std': 0.005},
            {'histo': "Track/TrackVertexMonitor/track IP X vs phi", 'std': 0.005}
        ],
        'th1d_error_bars': [
            {'histo': "Track/TrackVertexMonitor/fast track IP X"},
            {'histo': "Track/TrackVertexMonitor/fast track IP Y"},
            {'histo': "Track/TrackVertexMonitor/track IP Y"},
            {'histo': "Track/TrackVertexMonitor/track IP X"},
            {'histo': "IT/ITClusterMonitor/Number of clusters vs TELL1"},
            {'histo': "TT/TTClusterMonitor/Number of clusters vs TELL1"}
        ],
        'linear_alarms': [
            {'histo': 'dq_jpsi_mass', 'mean': 3.096916},
            {'histo': 'dq_pi0_mass', 'mean': 134.9767}
        ]
    }

    INTERACTION_PREPARE = {
        'mean': partial(prepare_mean, partial(get_histo_feature, feature=('alarm', 'mean'))),
        'linear_alarms': partial(prepare_mean, get_linear_features)
    }

    INTERACTION_FIT = {
        'mean': fit_mean,
        'linear_alarms': fit_mean
    }

    INTERACTION_PREDICT = {
        'error_bars': predict_error_bars,
        'error_bars_ignore_low_stats': predict_error_bars,
        'mean': predict_mean,
        'mean_std': predict_mean_std,
        'fluctuations_around_constant': predict_fluctuations,
        'th1d_error_bars': predict_error_bars,
        'linear_alarms': predict_mean
    }

    def __init__(self, quantile=0.95):
        self.quantile = quantile
        self.alarms = {key: pd.DataFrame(value).set_index('histo')
                       for key, value in AlarmFilter.ALARMS.items()}
        self.cd = None

    def reset(self):
        self.cd = {}

    def fit(self, X):
        self.reset()

        _, X = get_train_features(X)

        for name, handler in AlarmFilter.INTERACTION_FIT.items():
            handler(name, self, X)

        return self

    def predict(self, X):
        X = X['features']

        return pd.concat([correct_cols(name, handler(name, self, X))
                          for name, handler in AlarmFilter.INTERACTION_PREDICT.items()], axis=1)
