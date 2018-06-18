import hashlib
import numpy as np
import pandas as pd
from time import mktime
from functools import partial
from operator import gt, eq


def plain_batch_generator(gen, callback, batch_size):
    try:
        while True:
            res = []
            for _ in xrange(batch_size):
                res.append(next(gen))

            yield callback(res)
    except StopIteration:
        yield callback(res)


def md5(s):
    m = hashlib.md5()
    m.update(s)
    return m.hexdigest()


def to_timestamp(date):
    return int(mktime(date.timetuple()))


def np_histo(histo):
    if isinstance(histo, dict):
        for key, value in histo.items():
            histo[key] = np_histo(value)
        return histo
    elif isinstance(histo, list):
        return np.array(histo)
    else:
        return histo


class Maybe:
    def __init__(self, value=None, error_message=None):
        self.value = value
        self.error_message = error_message

    def is_error(self):
        return self.error_message is not None

    @staticmethod
    def concat_helper(callback, items):
        errors = [a.error_message for a in items if a.is_error()]

        if not len(errors):
            return Maybe(value=callback([a.value for a in items]))
        else:
            return Maybe(error_message=sum(errors, []))

    @staticmethod
    def concat(items):
        return Maybe.concat_helper(pd.concat, items)

    @staticmethod
    def concat_id(items):
        return Maybe.concat_helper(lambda x: x, items)


def inverse_dict(d):
    result = {}

    for key, value in d.items():
        for v in value:
            result[v] = key

    return result


class NullStripper:
    def __init__(self, data):
        self.notnull = data.notnull()
        if isinstance(data, pd.DataFrame):
            self.notnull = self.notnull.all(axis=1)

    def map_one(self, arg):
        return arg[self.notnull].copy() if arg is not None else None

    def get_notnull(self, *args):
        if len(args) > 1:
            return map(self.map_one, args)
        else:
            return self.map_one(args[0])

    def make_data(self, df):
        return df.reindex(self.notnull.index)


def nan_op(op, series, n):
    result = op(series, n).astype(np.int)
    result[series.isnull()] = np.nan

    return result


nan_gt = partial(nan_op, gt)
nan_eq = partial(nan_op, eq)


class TailHandler:
    def __init__(self):
        self.data_index = None

    def make_tail(self, tail, data):
        self.data_index = data.values()[0].index

        return {key: pd.concat([value, data[key]]) for key, value in tail.items()}

    def cut_tail_dict(self, data):
        return {key: self.cut_tail(value) for key, value in data.items()}

    def cut_tail(self, data):
        return data.loc[self.data_index]


def dict2df_keys(d, keys):
    return pd.DataFrame({key: d[key] for key in keys})