import simplejson
from multiprocessing import cpu_count

from roboshifter.internal.collector import Collector
from roboshifter.internal.constant_filter import ConstantFilter
from roboshifter.internal.classifier import Roboshifter
from roboshifter.internal.roboshifter_utils import RoboshifterError
from roboshifter.internal.report_maker import ReportMaker
from roboshifter.internal.alarms import AlarmFilter


class Controller:
    def __init__(self, external_collector, njobs=1, root_dir='.'):
        """
        A class for Roboshifter usage.

        Parameters
        ----------
        external_collector : roboshifter.external.external_collector.ExternalCollector
            ExternalCollector instance to load data with
        njobs : int, optional (default=1)
            The number of jobs to run in parallel for both data processing and algorithm
            application. If -1, then the number of jobs is set to the number of cores.
        root_dir : str, optional (default='.')
            Working directory to store data in.
        """

        if njobs == -1:
            njobs = cpu_count()

        self.__external_collector = external_collector
        self.__collector = Collector(njobs, root_dir)
        self.__njobs = njobs

    def __predict_runs_helper(self, run_numbers):
        rm = ReportMaker()

        X_train, X_test, y_train, test_info, unknown_runs = \
            self.__collector.get_roboshifter_data(run_numbers)

        try:
            rs = Roboshifter(self.__collector, verbose=False, njobs=self.__njobs)
            rs_prediction = rs.fit(X_train, y_train).predict(X_test)
        except RoboshifterError as re:
            return rm.make_fail_report(re)

        cf = ConstantFilter()
        cf_prediction = cf.fit(X_train).predict(X_test)

        af = AlarmFilter()
        af_prediction = af.fit(X_train).predict(X_test)

        return rm.make_report(rs_prediction, cf_prediction, af_prediction,
                              test_info, unknown_runs, run_numbers)

    def update_meta(self):
        """
        controller.update_meta()
.
        Update trend_linear_data.json, monet_histos.json, run_files.json with external_collector.
        """

        self.__collector.make_meta(self.__external_collector)

    def update_runs(self, run_numbers):
        """
        controller.update_runs(run_numbers)

        For each run in run_numbers controller will load data with external_collector and
        prepare all the required features.
        trend_linear_data.json, monet_histos.json, run_files.json will be updated as well.

        Parameters
        ----------
        run_numbers : list[int]
           Run numbers to update.
        """

        self.__collector.make_data(self.__external_collector, run_numbers)

    def predict_runs(self, run_numbers):
        """
        controller.predict_runs(run_numbers)

        Fit classifier on available data and then make a report for each run in run_numbers.

        Parameters
        ----------
        run_numbers : list[int]
           Run numbers to make reports for.

        Returns
        -------
        Json with the reports.
        """

        return simplejson.dumps(self.__predict_runs_helper(run_numbers), indent=4, ignore_nan=True)
