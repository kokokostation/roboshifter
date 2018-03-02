import os


PATHS = {
    'MONET_DATA_PATH': '/run/media/mikhail/cf8beb1e-b35f-4563-b2be-db41b8597493/home/mikhail/data/data/monet-roboshiter/',
    'HISTO_MAP': '/home/mikhail/lhcb/data/meta/monet_histos.json',
    'RUN_FILES': '/home/mikhail/lhcb/data/meta/run_files.json',
    'RUN_DIR': '/home/mikhail/lhcb/data/run',
    'REF_DIR': '/home/mikhail/lhcb/data/ref',
    'LINEAR_DATA': '/home/mikhail/lhcb/data/meta/trend_linear_data.json',
    'HISTO_TYPES': '/home/mikhail/lhcb/data/meta/histo_types.pickle',
    'ERRORS_PATH': '/home/mikhail/lhcb/data/meta/errors',
    'DATA_REF': '/home/mikhail/lhcb/data/meta/data_ref.pickle',
    'VALID_RUNS': '/home/mikhail/lhcb/data/meta/valid_runs.pickle',
    'FEATURE_DIR': '/home/mikhail/lhcb/data/features',
    'REJECTED_RUNS': '/home/mikhail/lhcb/data/meta/rejected_runs.pickle',
#    'STATS': '/home/mikhail/lhcb/data/meta/stats.pickle',
    'NO_STATS': '/home/mikhail/lhcb/omeprazol/no_stats',
    'TRAIN_DATA': '/home/mikhail/lhcb/data/train',
    'LINEAR_DATA_PREPARED': '/home/mikhail/lhcb/monet-trends-ml/training_data/run_data.pickle',
    'BOOTSTRAP_RESULT_FILES': '/home/mikhail/lhcb/data/meta/bootstrap_files',
    'BOOTSTRAP_RESULT': '/home/mikhail/lhcb/data/meta/bootstrap',
    'BOOTSTRAP_FIT': '/home/mikhail/lhcb/data/meta/bootstrap_fit',
    'PREPROCESSED_X': '/home/mikhail/lhcb/data/meta/preprocessed_x',
    'DISTRIBUTION_CLASSES': '/home/mikhail/lhcb/data/meta/distribution_classes',
    'RUN_TIMES': '/home/mikhail/lhcb/data/meta/run_times',
    'INTERACTIONS': '/home/mikhail/lhcb/data/meta/interactions',
    'ASSESED_INTERACTIONS': '/home/mikhail/lhcb/data/meta/interactions.pickle'
}


class DataManager:
    def __init__(self, paths=PATHS):
        self.paths = paths

    def get_files(self, path_name):
        result = []

        for f in os.listdir(self.paths[path_name]):
            abs_path = os.path.join(self.paths[path_name], f)
            if os.path.isfile(abs_path):
                result.append((f, abs_path))

        return result