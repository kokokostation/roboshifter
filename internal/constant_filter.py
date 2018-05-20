from collections import Counter
import pandas as pd
import numpy as np

from roboshifter_utils import get_linear_features, get_histo_features
from features import get_train_features
from utils import nan_eq


class ConstantFilter:
    LINEAR_FEATURES = [
        'LHCState',
        'activity',
        'runtype',
        'triggerConfiguration',
        'program',
        'destination',
        'partitionname',
        'veloPosition',
        'state',
        'beamenergy',
        'beamgasTrigger',
        'betaStar',
        'lumiTrigger',
        'partitionid',
        'run_state'
    ]

    HISTO_FEATURES = {
        'DecodingErrors': [
            ('decoding', 'errors')
        ]
    }

    def prepare_features(self, X):
        return pd.concat([get_linear_features(X, ConstantFilter.LINEAR_FEATURES),
                          get_histo_features(X, ConstantFilter.HISTO_FEATURES)], axis=1)

    def fit(self, X):
        _, X = get_train_features(X)

        features = self.prepare_features(X)

        self.consts = {col: Counter(features[col]).most_common(1)[0][0]
                       for col in features}

        return self

    def predict(self, X):
        X = X['features']

        features = self.prepare_features(X)

        for col in features:
            features[col] = 1 - nan_eq(features[col], self.consts[col])

        return {
            'feature_flag': features,
            'overall_flag': (features != 0).any(axis=1).astype(np.int)
        }
