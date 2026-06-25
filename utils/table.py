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

    def __init__(self, *a, **kw):

        super().__init__(*a, **kw)
        self.dropped_index = None
        self.dropped_index = {}

    def copy(self):

        df = type(self)(super().copy())
        df.dropped_index = self.dropped_index.copy()
        return df

    # def drop(self, labels=None, *, inplace=True, **kw):
    #     if
    #     df = self if inplace else self.copy()
    #     pd.DataFrame.drop(df, *a, **kw, inplace=True)
    #     if inplace:
    #         return None
    #     return df

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

        index_order_ = [_ for _ in index_order_ if _ in index_names]
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

    def drop_index_level(self, hidden_index=['exp'], drop_unique=True, show=[], inplace=True, **kw):
        if not inplace:
            df = self.copy()
            df.drop_index_level(hidden_index=hidden_index, drop_unique=drop_unique, show=show,
                                inplace=True, **kw)
            return df
        hidden = set(self.index.names) & set(hidden_index)

        for k in self.index.names:
            index_k = self.index.get_level_values(k)
            values = set(index_k)
            if k in show:
                continue
            if (len(values) == 1 and drop_unique) or k in hidden:
                self.dropped_index[k] = index_k

        if len(self.dropped_index) == len(self.index.names):
            self.dropped_index.pop('job')
        logger.debug('hidden index: {}'.format(', '.join(self.dropped_index)))
        for _ in self.dropped_index:
            if _ in self.index.names:
                self.index = self.index.droplevel(_)

    def restore_index(self, level):

        idx = self.dropped_index.pop(level)
        df[level] = idx
        df.set_index(level, inplace=True, append=True)

    def filter_parse_args(self, parser=None, argv=None, **kw):

        if not parser:
            parser = argparse.ArgumentParser()

        for name in self.index.names:
            values = list(set(self.index.get_level_values(name)))
            while False:
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
                  list_values=None,
                  float_format='{:.1f}'.format, **kw):

        df = self.drop_index_level(**kw, inplace=False)

        removed_cols = [_ for _ in df.columns if not columns.get(_)]
        df.drop(removed_cols, axis='columns', inplace=True)
        df.rename(columns=columns, inplace=True)

        with pd.option_context("display.date_dayfirst", True, "display.date_yearfirst", False):
            df_str = pd.DataFrame.to_string(df, float_format=float_format)

        df_width = max(len(_) for _ in df_str.split('\n'))

        df_str += '\n'

        if show_dropped:
            df_str += '=' * df_width + '\n'
            for k, index in df.dropped_index.items():
                v = set(index)
                if len(v) > 1:
                    df_str += '{}: [{}]\n'.format(k, len(v))
                else:
                    df_str += '{}: {}\n'.format(k, *v)

        if list_values:
            try:
                values = set(self.index.get_level_values(list_values))
                logger.info('{}: {}'.format(list_values,
                                            ' -- '.join(map(str, values))))
            except (ValueError, KeyError):
                logger.error('{} not in index'.format(list_values))

        return df_str


if __name__ == '__main__':
    from utils.configdict import ConfigDict
    from utils.logger import set_loggers
    from utils.load import df_results
    import sys

    import argparse

    argv = '--load.result_dir ./results/lab-ia/main --ood old_mix'.split()

    argv = None if sys.argv[0] else argv

    config = ConfigDict()

    parser = config.create_parser()

    args, filter_args = parser.parse_known_args(argv)

    config.update(args)

    set_loggers(**config.logger)

    df = df_results(**config.load)
    df.reorder_index_levels(**config.table)

    unknown_args = df.filter_parse_args(parser=parser, argv=filter_args, **config.table)

    print(df.index.names)
    df.sort_index(inplace=True)
    print(df.to_string(**config.table))
