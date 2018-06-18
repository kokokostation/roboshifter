import os
import pandas as pd
import pickle


class DataManager:
    def __init__(self, paths):
        self.paths = paths

    def get_paths(self, path_name):
        result = []

        for f in os.listdir(self.paths[path_name]):
            abs_path = os.path.join(self.paths[path_name], f)
            if os.path.isfile(abs_path):
                result.append((f, abs_path))

        return result

    def read_file(self, path_name, suffix=None, extension=None):
        with open(self.make_fname(path_name, suffix, extension), 'r') as infile:
            return infile.read()

    def make_fname(self, path_name, suffix, extension):
        fname =  self.paths[path_name] if suffix is None \
            else os.path.join(self.paths[path_name], str(suffix))

        if extension is not None:
            fname = '{}.{}'.format(fname, extension)

        return fname

    def read_pickle(self, path_name, suffix=None, extension=None):
        fname = self.make_fname(path_name, suffix, extension)

        if os.path.isfile(fname):
            return pd.read_pickle(fname)
        else:
            return None

    def write_pickle(self, obj, path_name, suffix=None, extension=None):
        with open(self.make_fname(path_name, suffix, extension), 'wb') as out_file:
            pickle.dump(obj, out_file)