import pandas as pd


def plain_batch_generator(gen, callback, batch_size):
    try:
        while True:
            res = []
            for _ in xrange(batch_size):
                res.append(next(gen))

            yield callback(res)
    except StopIteration:
        yield callback(res)


class Maybe:
    def __init__(self, value=None, error_message=None):
        self.value = value
        self.error_message = error_message

    def is_error(self):
        return self.error_message is None

    @staticmethod
    def concat(items):
        errors = [a.error_message for a in items if a.is_error()]

        if not len(errors):
            return Maybe(value=pd.concat([a.value for a in items]))
        else:
            return '\n'.join(errors)