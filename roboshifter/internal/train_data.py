import pandas as pd
import numpy as np

from features import FeatureContainer, Flag
from classifier import Roboshifter


class TrainDataMaker:
    def __init__(self, collector):
        self.collector = collector

    def tune_histo_features(self, features):
        stat_cols = {prefix: [col for col in features['features'].columns
                              if col[0] == 'my' and col[-2:] == ('stats', prefix)]
                     for prefix in ['runs', 'refs']}

        means = features['features'][stat_cols['runs']].mean(axis=0).values
        means /= means.sum()

        for prefix in ['runs', 'refs']:
            df = features['features'][stat_cols[prefix]]
            s = df.sum(axis=1) / df.notnull().astype(np.int).multiply(means).sum(axis=1)
            features[('features', ('integral', '{}sum'.format(prefix)))] = s

    BEGIN = 174410

    @staticmethod
    def write_features_error(df, index, error_msg, flag):
        df.loc[index, ('info', 'flag')] = flag
        df.loc[index, ('info', 'errors')].apply(lambda l: l.append(error_msg))

    @staticmethod
    def my_features_present(features):
        return len([col for col in features['features'] if col[0] == 'my']) != 0

    def make_train_data(self, out_run_numbers, feature_extractor):
        feature_containers = pd.Series(index=out_run_numbers, dtype=np.object)

        for run_number in out_run_numbers:
            features = self.collector.get_run_features(run_number)

            if features is None:
                features = FeatureContainer(Flag.BAD, ['No features file for this run'])

            feature_containers[run_number] = features

        linear_data = feature_extractor.get_linear_data()
        features = pd.DataFrame([c.data for c in feature_containers],
                                index=feature_containers.index)
        features = features.merge(linear_data, how='left', left_index=True, right_index=True)
        features = pd.concat([features, FeatureContainer.concat(feature_containers)],
                             keys=['features', 'info'], axis=1)

        features_linear_data = features['features'][linear_data.columns]
        features[('info', 'missing_trend_linear')] = features_linear_data.isnull()\
            .apply(lambda row: row[row].index.tolist(), axis=1)
        features.loc[features_linear_data.isnull().any(axis=1), ('info', 'flag')] = Flag.TAIL
        features.loc[features_linear_data[('linear', 'run_length')].isnull(), ('info', 'flag')] = Flag.BAD

        if TrainDataMaker.my_features_present(features):
            is_reference = True
            for col in features['features']:
                if col[0] == 'my' and col[-2:] == ('stats', 'kolmogorov'):
                    is_reference &= features[('features', col)] == 0

            TrainDataMaker.write_features_error(
                features, is_reference,
                'This run is taken as reference so there is no point to analyze it',
                Flag.BAD)

        curr_features = self.collector.get_train_data()
        if curr_features is None:
            curr_features = pd.DataFrame()
        else:
            curr_features = curr_features.loc[:, features.columns]

        curr_features = curr_features.append(features)
        curr_features = curr_features.loc[~curr_features.index.duplicated(keep='last')].sort_index()

        if TrainDataMaker.my_features_present(curr_features):
            self.tune_histo_features(curr_features)

            refs = curr_features['features'].groupby([('linear', 'reference')])
            refsum = refs.first()[('integral', 'refssum')]
            mean = curr_features[('features', ('integral', 'runssum'))].mean()

            for ref, value in refsum.iteritems():
                if value * Roboshifter.MAX_REFERENCE_STATS_RATIO < mean:
                    error_msg = 'The statistics of this run reference are too low so ' \
                                'histogram predictions are unreliable'
                    TrainDataMaker.write_features_error(curr_features, refs.groups[ref],
                                                        error_msg, Flag.TAIL)

        curr_features.loc[~curr_features[('features', 'flag')].isin([0, 1]),
                          ('info', 'flag')] = Flag.TAIL

        error_msg = 'Runnumber is less than {} -- data then was too weird'\
            .format(TrainDataMaker.BEGIN)

        TrainDataMaker.write_features_error(curr_features,
                                            slice(None, TrainDataMaker.BEGIN - 1),
                                            error_msg,
                                            Flag.BAD)

        switch_col = ('features', ('linear', 'switch'))
        switches = curr_features[curr_features[('info', 'flag')] == Flag.TRAIN]\
            .groupby([switch_col])
        for switch, index in [(sw, switches.groups.get(sw))
                              for sw in curr_features[switch_col].unique()]:
            if index is None or len(index) < Roboshifter.MIN_SWITCH_LENGTH:
                error_msg = \
                    'The switch this run belongs to is too short ({} ' \
                    'samples) so histogram predictions are unreliable'.format(
                        0 if index is None else len(index))

                true_index = curr_features[switch_col] == switch
                TrainDataMaker.write_features_error(curr_features, true_index,
                                                    error_msg, Flag.TAIL)

        self.collector.write_train_data(curr_features)

        histo_types = {}
        for col in curr_features['features'].columns:
            if col[0] == 'my':
                ht, hk = col[1:3]
                histo_types[hk] = ht
        self.collector.write_histo_types(histo_types)

    def make_X_y(self, df):
        df = df[df[('info', 'flag')] <= Flag.TAIL]

        features = df['features'].drop('flag', axis=1)
        info = df['info'][['flag']]
        X = pd.concat([features, info], keys=['features', 'info'], axis=1)

        y = df[('features', 'flag')]

        return X, y

    def get_roboshifter_data(self, run_numbers):
        df = self.collector.get_train_data()

        unknown_runs = set(run_numbers) - set(df.index)
        run_numbers = sorted(set(df.index) & set(run_numbers))

        train_df = df.loc[:run_numbers[0] - 1]

        switch_col = ('features', ('linear', 'switch'))
        right_index = df[df[switch_col] == df[switch_col].loc[run_numbers[-1]]].index[-1]
        test_df = df.loc[run_numbers[0]:right_index]

        X_train, y_train = self.make_X_y(train_df)

        X_test, _ = self.make_X_y(test_df)
        test_info = test_df['info']

        return X_train, X_test, y_train, test_info, unknown_runs