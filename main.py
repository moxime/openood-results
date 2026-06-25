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
    from .utils import ConfigDict, set_loggers, df_results, plot_scores
    import pandas as pd

    argv = '--results_dir ./results/lab-ia filter --epoch 200 --set cifar100'

    argv = None if sys.argv[0] else argv.split()

    config = ConfigDict()

    parser = config.create_parser()

    args, filter_args = parser.parse_known_args(argv)

    config.update(args)
    config.setup()

    set_loggers(**config['logger'])

    logger.info('Looking for results in {}'.format(config['load'].get('result_directory')))

    for line in str(config).split('\n'):
        logger.debug(line)

    df = df_results(**config['load'])

    df.reorder_index_levels(**config['table'])

    logger.debug('Filter args: {}'.format(', '.join(filter_args)))
    unknown_args = df.filter_parse_args(parser=parser, argv=filter_args, **config['table'])

    if unknown_args:
        logger.warning('Unknown args: {}'.format(', '.join(unknown_args)))

    if not len(df):
        logger.error('No df (all results are filtered out')
    else:
        print(df.to_string(**config['table']))

    try:
        config['table']['show'].append('has_scores')
        config['table']['columns']['SCORES'] = 'SCORES'
        df.drop_index_level(**config['table'])
        plot_scores(df, **config['scores'], wait=True)
    except ValueError:
        logger.error('No plot done')


if __name__ == '__main__':

    main()
