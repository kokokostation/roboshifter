import ROOT
import os
import pickle
import hashlib
import json
from tqdm import tqdm
import numpy as np

from data_extraction import DataExtractor
from data_manager import DataManager, PATHS


class DataPreparer(DataManager):
    def __init__(self, paths=PATHS):
        DataManager.__init__(self, paths)

        self.data_extractor = DataExtractor(paths)
        self.histo_keys = self.data_extractor.get_histo_keys()

    def try_get_object(self, f, objname):
        objname = str(objname)
        voname = objname.split("/")

        o = f.Get(objname)

        # PATCH for RICH histrograms
        if not o:
            o = f.Get("{}.{}/{}".format(voname[0], voname[1], objname))

        # PATCH for CALO 2 and Muon 6 histrograms
        if not o:
            o = f.Get(objname.replace(voname[0] + "/", "", 1))

        # PATCH for muon 5
        if not o and len(voname) > 2:
            o = f.Get(voname[1] + voname[2] + "/" + objname)

        # PATCH for muon 6 eff histogram direct copy paste from the online libs
        if not o and voname[0] == "Efficiency":
            try:
                myname = voname[1]
                name_regions, name_station = "Region", "Station"
                nom = ROOT.TH1D()
                denom = ROOT.TH1D()
                if name_regions in myname:
                    nom = f.Get("MuEffMonitor/m_RegionsEff_num")
                    denom = f.Get("MuEffMonitor/m_RegionsEff_den")
                elif name_station in myname:
                    nom = f.Get("MuEffMonitor/m_StationsEff_num")
                    denom = f.Get("MuEffMonitor/m_StationsEff_den")
                if ROOT.TEfficiency.CheckConsistency(nom, denom):
                    efficiency_T = ROOT.TEfficiency(nom, denom)
                    o = efficiency_T
            except:
                return None
        return o

    def handle_th1d(self, t):
        rng = xrange(1, t.GetNbinsX() + 1)

        return {
            'xs': map(t.GetBinCenter, rng),
            'vals': map(t.GetBinContent, rng),
            'widths': map(t.GetXaxis().GetBinWidth, rng)
        }

    def handle_th2d(self, t):
        global_bins = [t.GetBin(binx, biny)
                       for binx in xrange(1, t.GetNbinsX() + 1)
                       for biny in xrange(1, t.GetNbinsY() + 1)]
        return {
            'vals': map(t.GetBinContent, global_bins)
        }

    def handle_tefficiency(self, t):
        return {
            'total': self.handle_th1d(t.GetTotalHistogram()),
            'passed': self.handle_th1d(t.GetPassedHistogram())
        }

    def handle_tprofile(self, t):
        rng = xrange(1, t.GetNbinsX() + 1)

        return {
            'xs': map(t.GetBinCenter, rng),
            'vals': map(t.GetBinContent, rng),
            'widths': map(t.GetXaxis().GetBinWidth, rng),
            'errs': map(t.GetBinError, rng),
            'entries': map(t.GetBinEntries, rng)
        }

    def process_run(self, fname, error_file):
        HANDLERS = {
            ROOT.TH1D: self.handle_th1d,
            ROOT.TH2D: self.handle_th2d,
            ROOT.TEfficiency: self.handle_tefficiency,
            ROOT.TProfile: self.handle_tprofile,
            ROOT.TObject: lambda t: None
        }

        tfile = ROOT.TFile(fname)

        contents = {}

        for histo_key in self.histo_keys:
            t = self.try_get_object(tfile, histo_key)

            contents[histo_key] = HANDLERS[type(t)](t)

            if type(t) == ROOT.TObject:
                error_file.write('{} {}\n'.format(fname.split('/')[-1], histo_key))

        return contents

    def map_dir(self, path, out_path):
        with open(self.paths['ERRORS_PATH'], 'w') as error_file:
            for f, abs_path in tqdm(self.get_files(path)):
                    with open(os.path.join(out_path, '{}.pickle'.format(f.split('.')[0])),
                              'wb') as out_file:
                        pickle.dump(self.process_run(abs_path, error_file), out_file)

    @staticmethod
    def md5(s):
        m = hashlib.md5()
        m.update(s)
        return m.hexdigest()

    def make_ref_mapping(self):
        with open(self.paths['RUN_FILES'], 'r') as run_file:
            run_files = json.load(run_file)

        data_ref = {}
        for run, files in run_files.items():
            if files['ref'] is None:
                continue

            ref_path = os.path.join("ref", "{}.root".format(DataPreparer.md5(files['ref'])))
            data_path = os.path.join("{}.root".format(run))

            if all(os.path.isfile(os.path.join(self.paths['MONET_DATA_PATH'], fname)) for fname in
                   [ref_path, data_path]):
                data_ref[int(data_path[:-5])] = ref_path[4:-5]

        with open(self.paths['DATA_REF'], 'wb') as out_file:
            pickle.dump(data_ref, out_file)

    def make_histo_types(self, some_run='195979.root'):
        histo_types = {}
        tfile = ROOT.TFile(os.path.join(self.paths['MONET_DATA_PATH'], some_run))

        for histo_key in self.histo_keys:
            t = self.try_get_object(tfile, histo_key)
            assert type(t) != ROOT.TObject
            histo_types[histo_key] = str(type(t)).split('.')[-1][:-2]

        with open(self.paths['HISTO_TYPES'], 'wb') as out_file:
            pickle.dump(histo_types, out_file)

    @staticmethod
    def missing_histos(histos):
        return [key for key, value in histos.items() if value is None]

    def make_valid_runs(self):
        valid_runs = []
        rejected_runs = {}

        data_ref = self.data_extractor.get_data_ref()
        missing_histos_ref = {}

        for run_file in tqdm(os.listdir(self.paths['RUN_DIR'])):
            run_number = int(run_file[:-7])

            runh = self.data_extractor.get_run(run_number)
            missing_histos_run = DataPreparer.missing_histos(runh)

            refh = self.data_extractor.get_reference(run_number)
            reference_available = refh is not None
            if reference_available and data_ref[run_number] not in missing_histos_ref:
                missing_histos_ref[data_ref[run_number]] = DataPreparer.missing_histos(refh)

            if not missing_histos_run and reference_available \
                    and not missing_histos_ref[data_ref[run_number]]:
                valid_runs.append(run_number)
            else:
                rejected_runs[run_number] = {
                    'missing_histos_run': missing_histos_run,
                    'missing_histos_ref': missing_histos_ref[data_ref[run_number]] if reference_available else [],
                    'reference_available': reference_available
                }
                
        with open(self.paths['VALID_RUNS'], 'wb') as out_file:
            pickle.dump(valid_runs, out_file)

        with open(self.paths['REJECTED_RUNS'], 'wb') as out_file:
            pickle.dump(rejected_runs, out_file)
            
    def make_data(self):
        for d, od in [(self.paths['MONET_DATA_PATH'], self.paths['RUN_DIR']),
                      ('{}/ref'.format(self.paths['MONET_DATA_PATH']), self.paths['REF_DIR'])]:
            self.map_dir(d, od)

    def make_meta(self):
        self.make_ref_mapping()
        self.make_histo_types()
        self.make_valid_runs()