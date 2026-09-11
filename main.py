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
    import pandas as pd
    from .utils import ConfigDict, set_loggers, df_results, plot_scores, compute_scores_stats

    config = ConfigDict(config_root='default')

    filter_args = config.parse_args()

    set_loggers(**config.logger)

    logger.info('Looking for results in {}'.format(config.load.result_directory))

    for line in str(config).split('\n'):
        logger.debug(line)

    try:
        df = df_results(**config.load)
    except ValueError as e:
        logger.error('No results to be loaded in {}'.format(config.load.result_directory))
        logger.debug(e)
        return

    logger.debug('Filter args: {}'.format(', '.join(filter_args)))

    unknown_args = df.filter_parse_args(argv=filter_args, **config.table)

    compute_scores_stats(df, **config.scores)

    if config.from_file:
        df_ = {}
        subconfigs = ConfigDict(config.from_file)

        for name in subconfigs.concat:
            subconfig = config.copy()
            subconfig.table.update(**subconfigs[name])
            subdf = df.copy()
            subdf.filter(**subconfig.table.filters)
            subdf = subdf.drop_levels(**subconfig.table)
            df_[name] = subdf
            logger.info('Built table {} of len {}'.format(name, len(subdf)))

        configupdate = subconfigs.get('full') or {}
        config.table.update(configupdate)
        df = df.concat(df_.values(), **config.table)
    else:
        df = df.drop_levels(**config.table)

    try:
        df.print(**config.table)
        df.to_latex(**config.table.tex)
    except ValueError:
        pass

    if unknown_args:
        logger.error('Unknown args: {}'.format(', '.join(unknown_args)))

    plot_scores(df, **config.table.plot, wait=True)


if __name__ == '__main__':

    main()
