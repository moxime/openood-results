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
    from .utils import ConfigDict, set_loggers, df_results, plot_scores, scores_stats
    import pandas as pd

    argv = '--results_dir ./results/lab-ia filter --epoch 200 --set cifar100'

    argv = None if sys.argv[0] else argv.split()

    config = ConfigDict()

    parser = config.create_parser()

    args, filter_args = parser.parse_known_args(argv)

    config.update(args)
    config.setup()

    set_loggers(**config.logger)

    logger.info('Looking for results in {}'.format(config.load.result_directory))

    for line in str(config).split('\n'):
        logger.debug(line)

    try:
        df = df_results(**config.load)
    except ValueError:
        logger.error('No results to be loaded in {}'.format(config.load.result_directory))
        return

    logger.debug('Filter args: {}'.format(', '.join(filter_args)))

    unknown_args = df.filter_parse_args(parser=parser, argv=filter_args, **config.table)
    scores_stats(df, **config.scores)

    if unknown_args:
        logger.error('Unknown args: {}'.format(', '.join(unknown_args)))

    try:
        df.print(**config.table)
        df.drop_levels(**config.table).to_latex(**config.table.tex)
    except ValueError:
        pass

    plot_scores(df.drop_levels(**config.table), **config.scores, wait=True)


if __name__ == '__main__':

    main()
