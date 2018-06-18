import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import precision_score, accuracy_score, recall_score

from roboshifter.internal.collector import Collector
from roboshifter.internal.features import Flag
from roboshifter.internal.classifier import Roboshifter


def prfa(y_test, y_pred):
    return [precision_score(y_test, y_pred),
            recall_score(y_test, y_pred),
            accuracy_score(y_test, y_pred),
            y_test.sum()]


def get_pure_train_data():
    collector = Collector()

    traindf = collector.get_train_data()
    traindf = traindf[traindf[('info', 'flag')] == Flag.TRAIN]

    return traindf


def cut_tails(rs, index):
    rs.fit_tail_X = rs.fit_tail_X.iloc[:index]
    rs.fit_tail_y = rs.fit_tail_y.iloc[:index]
    rs.info[True]['tail_stat_flag'] = rs.info[True]['tail_stat_flag'].iloc[:index]


def make_X_y(df):
    flag = ('features', 'flag')
    return df.drop([flag], axis=1), df[flag]


def roboshifter_concatenator(dicts):
    return {key: pd.concat([d[key] for d in dicts]) for key in dicts[0].keys()}


def roboshifter_first_fit(model):
    return model.get_predict_keys(True, ['stat_proba', 'linear_proba', 'histo_score'])


def evaluate_roboshifter(begin, df, predict_modifier=None, njobs=4, verbose=False):
    collector = Collector()
    rs = Roboshifter(collector, njobs=njobs, verbose=verbose)
    X, y = make_X_y(df)
    switch = df[('features', ('linear', 'switch'))]

    return evaluate_for_switches(begin, X, y, switch, rs,
                                 roboshifter_concatenator,
                                 roboshifter_first_fit,
                                 predict_modifier)


def evaluate_classifier(begin, X, y, switch, model, predict_modifier=None):
    _, preds, ys = evaluate_for_switches(begin, X, y, switch, model, np.concatenate,
                                         lambda x: None, predict_modifier)
    preds = pd.Series(preds, index=ys.index)

    result = []
    for s in switch.loc[ys.index].unique():
        index = switch[switch == s].index
        result.append(prfa(ys.loc[index], preds.loc[index]))

    return np.array(result)


def evaluate_for_switches(begin, X, y, switch, model, concatenator, first_fit,
                          predict_modifier=None):
    switch_borders = [switch[switch == s].index.max() for s in switch.unique()]

    first_fit_result = None
    predictions = []
    ys = pd.Series()

    for i, v in tqdm(list(enumerate(switch_borders[:-1]))):
        train, test = slice(None, v), slice(v + 1, switch_borders[i + 1])
        X_train, X_test, y_train, y_test = X.loc[train], X.loc[test], y.loc[train], y.loc[test]

        if len(X_train) > begin:
            model.fit(X_train, y_train)
            if first_fit_result is None:
                first_fit_result = first_fit(model)

            prediction = model.predict(X_test)
            if predict_modifier is not None:
                prediction = predict_modifier(prediction, model)

            predictions.append(prediction)
            ys = ys.append(y_test)

    return first_fit_result, concatenator(predictions), ys