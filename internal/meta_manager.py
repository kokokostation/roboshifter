import json
from utils import md5


class MetaManager:
    def __init__(self, collector, external_collector):
        self.collector = collector
        self.external_collector = external_collector

    def make_meta(self):
        # TREND_LINEAR_DATA------------------------------------------------------------------------

        trend_linear_data = json.loads(self.external_collector.get_trend_linear_data())

        for value in trend_linear_data.values():
            value['rundb_data'] = json.loads(value['rundb_data'])

        self.collector.write_pickle(trend_linear_data, 'TREND_LINEAR_DATA')

        # RUN_FILES--------------------------------------------------------------------------------

        run_files = json.loads(self.external_collector.get_run_files())

        data_ref = {}
        for run_number, files in run_files.items():
            if files['ref'] is not None:
                data_ref[int(run_number)] = md5(files['ref'])

        self.collector.write_pickle(data_ref, 'DATA_REF')

        # MONET_HISTOS-----------------------------------------------------------------------------

        monet_histos = json.loads(self.external_collector.get_monet_histos())

        self.collector.write_pickle(monet_histos, 'MONET_HISTOS')