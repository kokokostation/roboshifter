from features import Flag


def no_info(series):
    return series[series.isnull()].index.tolist()


def make_features(series):
    return {
        'suspicious': series[series.notnull() & (series != 0)].index.tolist(),
        'no_info': no_info(series)
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

    def make_report(self, rs_prediction, cf_prediction, af_prediction, test_info, run_numbers):
        for value in [rs_prediction['linear_prediction'], cf_prediction['feature_flag']]:
            value.rename(columns=ReportMaker.rename, inplace=True)

        reports = {}

        for run in run_numbers:
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

                report['alarm_filter'] = make_features(af_prediction.loc[run])

            reports[run] = report

        return {
            'roboshifter_status': 'SUCCESS',
            'reports': reports
        }