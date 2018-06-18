import ROOT
from functools import partial
from multiprocessing import Pool

from utils import Maybe, np_histo

from renderer.data_load.libhistograms import try_get_object


def handle_th1d(t):
    rng = xrange(1, t.GetNbinsX() + 1)

    return {
        'xs': map(t.GetBinCenter, rng),
        'vals': map(t.GetBinContent, rng),
        'widths': map(t.GetXaxis().GetBinWidth, rng),
        'mean': t.GetMean(),
        'errs': map(t.GetBinError, rng),
        'type': 'TH1D'
    }


def handle_th2d(t):
    global_bins = [t.GetBin(binx, biny)
                   for binx in xrange(1, t.GetNbinsX() + 1)
                   for biny in xrange(1, t.GetNbinsY() + 1)]
    return {
        'vals': map(t.GetBinContent, global_bins),
        'type': 'TH2D'
    }


def handle_tefficiency(t):
    return {
        'total': handle_th1d(t.GetTotalHistogram()),
        'passed': handle_th1d(t.GetPassedHistogram()),
        'type': 'TEfficiency'
    }


def handle_tprofile(t):
    rng = xrange(1, t.GetNbinsX() + 1)

    return {
        'xs': map(t.GetBinCenter, rng),
        'vals': map(t.GetBinContent, rng),
        'widths': map(t.GetXaxis().GetBinWidth, rng),
        'errs': map(t.GetBinError, rng),
        'entries': map(t.GetBinEntries, rng),
        'type': 'TProfile'
    }


HANDLERS = {
    ROOT.TH1D: handle_th1d,
    ROOT.TH2D: handle_th2d,
    ROOT.TEfficiency: handle_tefficiency,
    ROOT.TProfile: handle_tprofile,
    ROOT.TObject: lambda _: None,
    type(None): lambda _: None
}


class Root2Py:
    def __init__(self, collector, external_collector, njobs):
        self.collector = collector
        self.external_collector = external_collector
        self.njobs = njobs

    def process_runs(self, run_numbers):
        args = []

        histo_keys = self.collector.get_histo_keys()
        data_ref = self.collector.get_data_ref()

        for run_number in run_numbers:
            args.append((self,
                         histo_keys,
                         run_number,
                         run_number,
                         False))

        ref_hashes = set()
        for run_number in run_numbers:
            ref_hash = data_ref.get(run_number)

            if ref_hash is not None and ref_hash not in ref_hashes:
                args.append((self,
                             histo_keys,
                             run_number,
                             ref_hash,
                             True))

                ref_hashes.add(ref_hash)

        # for arg in args:
        #     process_run(arg)

        pool = Pool(self.njobs)
        pool.map(process_run, args)
        pool.close()
        pool.join()


def process_tfile(tfile, histo_keys, handle):
    contents = {}

    for histo_key in histo_keys:
        t = try_get_object(tfile, histo_key)
        typ = type(t)

        if typ not in HANDLERS:
            val = Maybe(error_message=[
                'Error while processing .root. No handler available for {} of {}'
                    .format(typ, handle)])
        else:
            handled = HANDLERS[typ](t)

            if handled is None:
                val = Maybe(error_message=['Error while processing .root. No data for {}'
                            .format(handle)])
            else:
                val = Maybe(value=np_histo(handled))

        contents[histo_key] = val

    return contents


def process_run(arg):
    self, histo_keys, run_number, handle, reference = arg

    getter = self.external_collector.get_reference_tfile if reference \
        else self.external_collector.get_run_tfile
    writer = partial(self.collector.write_root2py, reference)

    tfile = getter(run_number)

    if tfile is None:
        val = Maybe(error_message=["External collector doesn't give any .root for the handle"
                                   " {} of run {}".format(handle, run_number)])
    else:
        val = Maybe(value=process_tfile(tfile, histo_keys, handle))

    writer(val, handle)