import logging
import argparse
from collections import defaultdict
import numpy as np
import pandas as pd
from .load import fetch_results

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


def create_filter_parser(df, hidden_index=['exp'], parser=None, **k):

    if not parser:
        parser = argparse.ArgumentParser()

    hidden = []
    for name in df.index.names:
        values = list(set(df.index.get_level_values(name)))
        parser.add_argument('--{}'.format(name), nargs='*',
                            dest='filter.{}'.format(name),
                            default=values, type=type(values[0]))

    hidden = set(df.index.names) ^ set(hidden_index)
    logger.debug('hidden index: {}'.format(', '.join(hidden)))
    parser.add_argument('--show', nargs='*', choices=hidden)

    return parser


def df_results(df_columns={'FPR@95': 'fpr', 'AUROC': 'auc'}, epoch='last', **kw):

    logger.info('Looking for results in {}'.format(kw.get('results_directory')))
    list_of_dfs = list(fetch_results(**kw))
    logger.info('Found {} results'.format(len(list_of_dfs)))

    df = concatenate(*list_of_dfs, **kw)

    kept_cols = [_ for _ in df.columns if df_columns.get(_)]

    df = df[kept_cols]

    df.rename(columns=df_columns, inplace=True)

    dropped_index = {}

    for n in df.index.names:
        values = set(df.index.get_level_values(n))
        if len(values) == 1:
            df.index = df.index.droplevel(n)
            dropped_index[n] = next(iter(values))

    return df, dropped_index


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

    df, dropped_index = df_results(**config)

    for name in df.index.names:
        values = list(set(df.index.get_level_values(name)))
        # print('***', '--{}'.format(name), type(values[0]), values)
        parser_filter.add_argument('--{}'.format(name), nargs='*',
                                   dest='filter.{}'.format(name),
                                   default=values, type=type(values[0]))

    parser.parse_args(argv, namespace=args)

    # print('FARGS\n', args)

    for k in df.index.names:
        df = df.iloc[df.index.isin(vars(args)['filter.{}'.format(k)], level=k)]

    for n in df.index.names:
        values = set(df.index.get_level_values(n))
        if len(values) == 1:
            df.index = df.index.droplevel(n)
            dropped_index[n] = next(iter(values))

    df.sort_index(inplace=True)
    print(df.to_string())
    print('='*20)
    # print(df.index.names)
    print(ConfigDict(dropped_index, _registering_default=False))
