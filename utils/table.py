import logging
import argparse
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def ftype(o):

    def _type(s):

        if s.lower() in ('null', 'none'):
            return None

        if isinstance(o, bool):
            return s.lower() in ('true', 'yes')

        return type(o)(s)

    return _type


class ResDF(pd.DataFrame):

    drop_index = {}

    def reorder_index_levels(self, index_order=['set', '...', 'ood', 'epoch', 'date'],
                             index_dependencies={}, **kw):

        index_names = list(self.index.names)

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
        super().reset_index(inplace=True)
        super().set_index(index_order, inplace=True)
        super().sort_index(inplace=True)

    def drop_index_level(self, hidden_index=['exp'], drop_unique=True, show=[], **kw):
        hidden = set(self.index.names) & set(hidden_index)

        for k in self.index.names:
            values = set(self.index.get_level_values(k))
            if k in show:
                continue
            if (len(values) == 1 and drop_unique) or k in hidden:
                self.drop_index[k] = values

        if len(self.drop_index) == len(self.index.names):
            self.drop_index.pop('job')
        logger.debug('hidden index: {}'.format(', '.join(self.drop_index)))
        for _ in self.drop_index:
            self.index = self.index.droplevel(_)

    def filter_parse_args(self, parser=None, argv=None, **kw):

        if not parser:
            parser = argparse.ArgumentParser()

        for name in self.index.names:
            values = list(set(self.index.get_level_values(name)))
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

        parser.add_argument('--last', nargs='?', default=0, const=10, type=int)

        if argv:
            args, _ = parser.parse_known_args(argv)

            for k in self.index.names:
                df_len = len(self)
                kept = vars(args)['filter.{}'.format(k)]
                values_before = set(self.index.get_level_values(k))
                self.drop(self.index[~self.index.isin(kept, level=k)], inplace=True)
                values = set(self.index.get_level_values(k))
                logger.debug('Filtering {} {}->{} {}'.format(k, df_len, len(self),
                                                             kept if len(values) < len(values_before) else ''))

            if args.last:
                self.sort_index(level='date', inplace=True)
                self.drop(self.index[:- --args.last], inplace=True)

            return _
        return []

    def to_string(self, columns={'FPR@95': 'fpr', 'AUROC': 'auc'}, show_dropped=True,
                  float_format='{:.1f}'.format, **kw):

        self.drop_index_level(**kw)

        removed_cols = [_ for _ in self.columns if not columns.get(_)]
        self.drop(removed_cols, axis='columns', inplace=True)
        self.rename(columns=columns, inplace=True)

        with pd.option_context("display.date_dayfirst", True, "display.date_yearfirst", False):
            df_str = super().to_string(float_format=float_format)

        df_width = max(len(_) for _ in df_str.split('\n'))

        df_str += '\n'
        df_str += '=' * df_width + '\n'

        if show_dropped:
            for k, v in self.drop_index.items():
                if len(v) > 1:
                    df_str += '{}: [{}]\n'.format(k, len(v))
                else:
                    df_str += '{}: {}\n'.format(k, *v)

        return df_str


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
