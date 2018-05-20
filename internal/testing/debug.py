# import pandas as pd
#
# from roboshifter.internal.collector import Collector
# from roboshifter.internal.fs_collector import FsCollector
#
#
# runs = pd.read_pickle('runs')
# fsc = FsCollector('/mnt/temp/roboshifter')
#
# collector = Collector(njobs=12)
# collector.make_data(fsc, runs)


from roboshifter.external.controller import Controller
from roboshifter.internal.fs_collector import FsCollector
from roboshifter.internal.testing.evaluation_utils import get_pure_train_data

fs_collector = FsCollector('/mnt/temp/roboshifter')
controller = Controller(fs_collector, 12)

df = get_pure_train_data()
runs = df.iloc[500:550].index.tolist()
prediction = controller.predict_runs(runs)