from features import Flag


def get_handles(series, index, mapper):
    return map(mapper, series[index].index.tolist())


def make_features(series, mapper=None):
    if mapper is None:
        mapper = lambda x: x

    return {
        'suspicious': get_handles(series, series.notnull() & (series != 0), mapper),
        'no_info': get_handles(series, series.isnull(), mapper)
    }


def alarm_mapper(entry):
    typ, histo = entry.split('::')

    return {
        'type': typ,
        'histo': histo
    }


def make_histos(series):
    series = series[series.notnull()]

    return series.to_dict()


class ReportMaker:
    ROBOSHIFTER_PREDICT_KEYS = [
        'linear_prediction',
        'histo_score',
        'stat_flag',
        'linear_flag'
    ]

    HANDLERS = {
        'linear_prediction': make_features,
        'histo_score': make_histos
    }

    @staticmethod
    def rename(key):
        if key[0] == 'my':
            return key[2]
        elif key[0] == 'linear':
            return key[1]

    def make_fail_report(self, re):
        return {
            'roboshifter_status': 'FAIL',
            'error_message': re.message
        }

    def make_report(self, rs_prediction, cf_prediction, af_prediction,
                    test_info, unknown_runs, run_numbers):
        for value in [rs_prediction['linear_prediction'], cf_prediction['feature_flag']]:
            value.rename(columns=ReportMaker.rename, inplace=True)

        reports = {}

        for run in run_numbers:
            if run in unknown_runs:
                report = {
                    'status': 'FAIL',
                    'error_messages': ['No data for this run in robosifter database at all. '
                                       'Consider calling roboshifter.external.controller.Controller.update_runs for this run.']
                }
            else:
                data = test_info.loc[run]

                report = {
                    'error_messages': data['errors'],
                    'histo_errors': data['histo_errors']
                }

                if data['flag'] == Flag.BAD:
                    report['status'] = 'FAIL'
                else:
                    report['status'] = 'SUCCESS'

                    for key in ReportMaker.ROBOSHIFTER_PREDICT_KEYS:
                        report[key] = rs_prediction[key].loc[run]

                        handler = ReportMaker.HANDLERS.get(key)
                        if handler is not None:
                            report[key] = handler(report[key])

                    report['constant_filter'] = make_features(cf_prediction['feature_flag'].loc[run])

                    report['alarm_filter'] = make_features(af_prediction.loc[run], alarm_mapper)

            reports[run] = report

        return {
            'roboshifter_status': 'SUCCESS',
            'reports': reports
        }