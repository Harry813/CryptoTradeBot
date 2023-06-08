import os


def parse_timeframe (timeframe):
    amount = int(timeframe[0:-1])
    unit = timeframe[-1]
    if 'y' == unit:
        scale = 60 * 60 * 24 * 365
    elif 'M' == unit:
        scale = 60 * 60 * 24 * 30
    elif 'w' == unit:
        scale = 60 * 60 * 24 * 7
    elif 'd' == unit:
        scale = 60 * 60 * 24
    elif 'h' == unit:
        scale = 60 * 60
    elif 'm' == unit:
        scale = 60
    elif 's' == unit:
        scale = 1
    else:
        raise ValueError('timeframe unit {} is not supported'.format(unit))
    return amount * scale


def get_path(filename: str = None):
    p = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        filename if filename else ""
    )
    return p
