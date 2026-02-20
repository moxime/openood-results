import logging
import time
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from .load import fetch_results
from pathlib import Path

logger = logging.getLogger(__name__)


def concatenate(*dfs, index_fill_values={}, **kw):

    index_dict = defaultdict(list)
    for df in dfs:
        for name in df.index.names:
            index_dict[name].append(df.index.names.index(name))

    for _ in index_dict:
        index_dict[_] = np.exp(index_dict[_]).mean()
    # print(dict(index_dict))
    sorted_index = sorted(index_dict, key=index_dict.get)

    df_ = []
    for df in dfs:
        index = df.index

        if not (isinstance(index, pd.MultiIndex)):
            df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])

        index_frame = df.index.to_frame()

        for c in sorted_index:
            if c not in index_frame.columns:
                index_frame[c] = index_fill_values.get(c)

        df.index = pd.MultiIndex.from_frame(index_frame[sorted_index])

        df_.append(df)

    return pd.concat(df_)


def df_results(df_columns={'FPR@95': 'fpr', 'AUROC': 'auc'},
               epoch='last', parse_dates=['date'], flash=False, **kw):

    t0 = time.time()
    res_dir = kw.get('results_directory')
    csv_path = Path(res_dir) / 'table.csv'

    if flash:
        try:
            df = pd.read_csv(csv_path, parse_dates=parse_dates)
            df.set_index([_ for _ in df.columns if _ not in df_columns], inplace=True)
        except FileNotFoundError:
            logger.warning('Flash df is true but {} does not exist, will fetch results'.format(csv_path))
            flash = False

    if not flash:

        logger.info('Looking for results in {}'.format(res_dir))
        list_of_dfs = list(fetch_results(**kw))
        logger.info('Found {} results in {:.1f}s'.format(len(list_of_dfs), time.time() - t0))
        df = concatenate(*list_of_dfs, **kw)
        df.to_csv(csv_path)
        logger.info('Table saved in {}'.format(csv_path))

    kept_cols = [_ for _ in df.columns if df_columns.get(_)]

    df = df[kept_cols]

    df.rename(columns=df_columns, inplace=True)

    df.drop_index = None  # to suppress warning
    df.drop_index = {}

    for n in df.index.names:
        values = set(df.index.get_level_values(n))
        if len(values) == 1:
            df.drop_index[n] = values

    t0 -= time.time()

    logger.info('Loaded {} lines in {:.1f}s'.format(len(df), -t0))
    return df


def df_filter_parse(df, hidden_index=['exp'], parser=None, argv=None, drop=True, **k):

    if not parser:
        parser = argparse.ArgumentParser()

    for name in df.index.names:
        values = list(set(df.index.get_level_values(name)))
        parser.add_argument('--{}'.format(name), nargs='*',
                            dest='filter.{}'.format(name),
                            default=values, type=type(values[0]))

    hidden = set(df.index.names) & set(hidden_index)
    logger.debug('hidden index: {}'.format(', '.join(hidden)))
    parser.add_argument('--show', nargs='*', choices=hidden, default=[])

    if argv:
        args, _ = parser.parse_known_args(argv)

        drop_index = df.drop_index
        for k in df.index.names:
            df = df.iloc[df.index.isin(vars(args)['filter.{}'.format(k)], level=k)]
            values = set(df.index.get_level_values(k))
            if len(values) == 1 or k in hidden:
                drop_index[k] = values

        for _ in args.show:
            drop_index.pop(_)

        df.drop_index = None  # to suppress warning
        df.drop_index = drop_index

        if drop:
            for _ in df.drop_index:
                df.index = df.index.droplevel(_)

    return df


if __name__ == '__main__':
    from utils.configdict import ConfigDict
    from utils.logger import set_loggers
    import sys

    import argparse

    argv = '--results_dir ./results/lab-ia filter --epoch 200 --set cifar100'

    argv = None if sys.argv[0] else argv.split()

    config = ConfigDict()

    parser = argparse.ArgumentParser()
    config.create_parser(parser=parser, exclude=['config_keys'])

    subparsers = parser.add_subparsers()

    parser_filter = subparsers.add_parser('filter', help='table filter help')

    args, _ = parser.parse_known_args(argv)

    config.update(args)
    set_loggers(**config)

    df = df_results(**config)
    df = df_filter_parse(df, **config, parser=parser_filter, argv=_)

    df.sort_index(inplace=True)
    print(df.to_string())
    print('='*20)
    # print(df.index.names)
    print(ConfigDict(df.drop_index, _registering_default=False))
