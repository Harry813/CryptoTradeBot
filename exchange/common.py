SUPPORTED_EXCHANGES = [
    'binance'
]


def remove_credentials (config) -> None:
    """
    Removes exchange keys from the configuration and specifies dry-run
    Used for backtesting / hyperopt / edge and utils.
    Modifies the input dict!
    """
    if config.get('dry_run', False):
        config['exchange']['key'] = ''
        config['exchange']['secret'] = ''
        config['exchange']['password'] = ''
        config['exchange']['uid'] = ''
