import logging
import argparse
import numpy as np

logger = logging.getLogger(__name__)


def ftype(o):

    def _type(s):

        if s.lower() in ('null', 'none'):
            return None

        if isinstance(o, bool):
            return s.lower() in ('true', 'yes')

        return type(o)(s)

    return _type


def df_sort_index(df, index_order=['set', '...', 'ood', 'epoch', 'date'], index_dependencies={}, **kw):

    index_names = list(df.index.names)

    try:
        dots = index_order.index('...')
        pre_sort = index_order[:dots]
        post_sort = index_order[dots+1:]
    except ValueError:
        pre_sort = index_order
        post_sort = []

    logger.debug('Index order: {} ... {}'.format(', '.join(pre_sort), ', '.join(post_sort)))

    index_order_ = [*pre_sort,
                    *[_ for _ in index_names if _ not in [*pre_sort, *post_sort]],
                    *post_sort]

    index_order = []
    for i in index_order_:
        index_order.append(i)
        if i in index_dependencies:
            for _ in index_dependencies[i]:
                if _ in index_order:
                    index_order.remove(_)
                    index_order.append(_)

    logger.debug('Index order: {}'.format(', '.join(index_order)))
    df.reset_index(inplace=True)
    df.set_index(index_order, inplace=True)
    df.sort_index(inplace=True)

    return df


def df_filter_parse_args(df, hidden_index=['exp'], parser=None, argv=None, drop=True, **kw):

    if not parser:
        parser = argparse.ArgumentParser()

    for name in df.index.names:
        values = list(set(df.index.get_level_values(name)))
        while True:
            try:
                values.remove(np.nan)
            except ValueError:
                break

        values_ = ','.join(map(str, values))
        if len(values_) > 50:
            values_ = values_[:47]+'...'

        logger.debug('Adding parser argument --{} of type {} '
                     '({} default values: {})'.format(name, type(values[0]).__name__,
                                                      len(values), values_))

        parser.add_argument('--{}'.format(name), nargs='*',
                            dest='filter.{}'.format(name),
                            default=values, type=ftype(values[0]))

    hidden = set(df.index.names) & set(hidden_index)
    logger.debug('hidden index: {}'.format(', '.join(hidden)))
    parser.add_argument('--show', nargs='*', choices=hidden, default=[])
    parser.add_argument('--last', nargs='?', default=0, const=10, type=int)

    df = df_sort_index(df, **kw)

    if argv:
        args, _ = parser.parse_known_args(argv)

        drop_index = df.drop_index
        for k in df.index.names:
            df_len = len(df)
            kept = vars(args)['filter.{}'.format(k)]
            values_before = set(df.index.get_level_values(k))
            df = df.iloc[df.index.isin(kept, level=k)]
            values = set(df.index.get_level_values(k))
            logger.debug('Filtering {} {}->{} {}'.format(k, df_len, len(df),
                                                         kept if len(values) < len(values_before) else ''))
            if len(values) == 1 or k in hidden:
                drop_index[k] = values

        if args.last:
            df = df.sort_index(level='date')
        df = df.iloc[-args.last:]

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
    df = df_filter_parse_args(df, **config, parser=parser_filter, argv=_)

    df.sort_index(inplace=True)
    print(df.to_string())
    print('='*20)
    # print(df.index.names)
    print(ConfigDict(df.drop_index, _registering_default=False))
