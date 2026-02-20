from __future__ import annotations
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__" and __package__ is None:
    import os
    import sys
    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))  # .../myproject
    except NameError:
        # temprary trick for C-u C-c C-c
        pkg_dir = os.getcwd()
    parent = os.path.dirname(pkg_dir)                     # .../
    sys.path.insert(0, parent)
    __package__ = 'openood-results'


def main():
    import sys
    import argparse
    from .utils import ConfigDict, set_loggers, df_results, df_filter_parse
    import pandas as pd

    argv = '--results_dir ./results/lab-ia filter --epoch 200 --set cifar100'

    argv = None if sys.argv[0] else argv.split()

    config = ConfigDict()

    parser = argparse.ArgumentParser()
    parser = config.create_parser()

    subparsers = parser.add_subparsers()

    parser_filter = subparsers.add_parser('filter', help='table filter help')

    args, remainig_args = parser.parse_known_args(argv)

    config.update(args)
    config.setup()
    set_loggers(**config)

    logger.info('Looking for results in {}'.format(config.get('results_directory')))

    for line in str(config).split('\n'):
        logger.debug(line)

    df = df_results(**config)

    df = df_filter_parse(df, parser=parser_filter, argv=remainig_args, **config)

    df.sort_index(inplace=True)

    if not len(df):
        logger.error('No df (all results are filtered out')
        return

    with pd.option_context("display.date_dayfirst", True, "display.date_yearfirst", False):
        df_str = df.to_string(float_format='{:.1f}'.format)

    df_width = max(len(_) for _ in df_str.split('\n'))
    print(df_str)
    print('='*df_width)
    # print(df.index.names)
    for _, v in df.drop_index.items():
        if len(v) > 1:
            df.drop_index[_] = '--'
        else:
            df.drop_index[_] = next(iter(v))

    print(ConfigDict(df.drop_index, _registering_default=False))


if __name__ == '__main__':

    main()
