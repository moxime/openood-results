import argparse
import numpy as np
import pandas as pd
import sys
import time

from .logger import logger


def ftype(t):

    def _type(s):

        if s.lower() == 'nan':
            return np.nan

        if s.lower() in ('null', 'none'):
            return None

        if t is bool:
            return s.lower() in ('true', 'yes')

        return t(s)

    return _type


def set_with_nan(iterable, return_type=False):
    """To make a set from an iterable with only one nan if any

    Rmk: set([a, a, b, nan, nan]) will return {a, b, nan, nan}

    """
    s = set(iterable)
    has_nan = False
    dtype = float

    for _ in list(s):

        if not isinstance(_, float):
            dtype = type(_)
            continue
        if np.isnan(_):
            s.remove(_)
            has_nan = True

    if has_nan:
        s.add(np.nan)

    if not return_type:
        return s
    return s, dtype


class ResDF(pd.DataFrame):

    def __init__(self, *a, **kw):

        super().__init__(*a, **kw)
        self._dropped_index = None
        self._dropped_index = {}
        self._fullindex = None

    def copy(self, **kw):

        df = type(self)(super().copy(**kw))
        df._dropped_index = self._dropped_index.copy()
        return df

    class Subsetter:
        def __init__(self, df, locator):
            self.locator = locator
            self.dropped_index = df._dropped_index

        def __getitem__(self, *vargs, **kwargs):
            df = ResDF(self.locator.__getitem__(*vargs, **kwargs))
            df._dropped_index = self.dropped_index.copy()
            return df

        def __setitem__(self, i, x):
            return self.locator.__setitem__(i, x)

    @property
    def loc(self):
        return self.Subsetter(self, super().loc)

    def sort_index(self, *a, **kw):

        logger.debug('Sort index')
        # once sorted, full index is no longer reliable
        self._fullindex = None
        return super().sort_index(*a, **kw)

    def unstack(self, *a, **kw):
        d = super().unstack(*a, **kw)
        return type(self)(d)

    @property
    def fullindex(self):
        if not self._dropped_index:
            return self.index
        return self._fullindex

    def reorder_index_levels(self, index_order=['set', '...', 'ood', 'epoch', 'date'],
                             index_dependencies={}, **kw):

        index_names = list(self.index.names)
        assert not self._dropped_index

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
        self.reset_index(inplace=True)
        self.set_index(index_order, inplace=True)
        self.sort_index(inplace=True)

    def drop_levels(self, exp_index=['job'], hide=[], drop_unique=True, show=[],
                    columns_rename={'FPR@95': 'fpr', 'AUROC': 'auc'},
                    **kw):

        df = self.rename(columns=columns_rename)
        if self._dropped_index:
            logger.warning('index already dropped returing df.copy()')
            return df
        assert not self._dropped_index

        df._fullindex = self.index.copy()

        hidden = set(df.index.names) & (set(hide) | set(exp_index))

        for k in df.index.names:
            index_k = df.index.get_level_values(k)
            values = set_with_nan(index_k)
            if k in show:
                continue
            if (len(values) == 1 and drop_unique) or k in hidden:
                df._dropped_index[k] = index_k

        if len(df._dropped_index) == len(df.index.names):
            df._dropped_index.pop('job')
        logger.debug('hidden index: {}'.format(', '.join(df._dropped_index)))
        for _ in df._dropped_index:
            df.index = df.index.droplevel(_)

        return df.agg(**kw['agg'])

    def agg(self, op='max', column=None, **kw):

        if op != 'max':
            raise NotImplementedError

        if column is None:
            return self

        index_names = list(self.index.names)[:-1]

        idx = self[column].groupby(index_names).idxmax()

        # print('***')
        # print(idx)

        # print('***')
        print(idx.dropna())

        return self.loc[idx.dropna()]

    def filter_index(self, key, *values, action='keep', inplace=True, **kw):

        if action in ('rm', 'remove'):
            action = 'remove'

        assert action in ('remove', 'keep')

        if key not in self.index.names:
            raise ValueError('{} not in index names ()'.format(key, self.index.names))

        if action == 'keep':
            return self.drop(self.index[~self.index.isin(values, level=key)], inplace=inplace)

        print('***', key, values)
        return self.drop(self.index[self.index.isin(values, level=key)], inplace=inplace)

    def get_parsers(self, **kw):
        keep_parser = argparse.ArgumentParser()
        rm_parser = argparse.ArgumentParser()
        for name in self.index.names:
            values, dtype = set_with_nan(self.index.get_level_values(name), return_type=True)
            values_ = ','.join(map(str, values))
            if len(values_) > 50:
                values_ = values_[:47]+'...'

            logger.debug('Adding parser argument --{} of type {} '
                         '({} default values: {})'.format(name, dtype.__name__,
                                                          len(values), values_))

            keep_parser.add_argument('--{}'.format(name), nargs='*',
                                     dest=name,
                                     type=ftype(dtype))

            rm_parser.add_argument('--{}-'.format(name), nargs='*',
                                   dest=name,
                                   type=ftype(dtype))

        rm_parser.add_argument('--last', nargs='?', default=0, const=10, type=int)
        return keep_parser, rm_parser

    def parse_args(self, argv, **kw):

        keep_parser, rm_parser = self.get_parsers()
        keep_args, unknown_args = keep_parser.parse_known_args(argv)
        rm_args, unknown_args = rm_parser.parse_known_args(unknown_args)
        return vars(keep_args), vars(rm_args), unknown_args

    def filter_parse_args(self, argv=None, **kw):

        t0 = time.time()
        kept_values, removed_values, unknown_args = self.parse_args(argv)

        for k, kept in kept_values.items():
            removed = removed_values[k]
            df_len = len(self)
            values_before = set(self.index.get_level_values(k))
            if kept is not None:
                self.drop(self.index[~self.index.isin(kept, level=k)], inplace=True)
            if removed is not None:
                self.drop(self.index[self.index.isin(removed, level=k)], inplace=True)
            values = set(self.index.get_level_values(k))
            logger.debug('Filtering {} {}->{} {}'.format(k, df_len, len(self),
                                                         kept if len(values) < len(values_before) else ''))
        if removed_values['last']:
            self.reorder_index_levels(index_order=['date', 'job'])
            self.drop(self.index[:- --removed_values['last']], inplace=True)

        logger.info('Filtered table of length {} in {:.1f}s'.format(len(self), time.time() - t0))
        self.reorder_index_levels(**kw)
        return unknown_args

    def print(self,
              columns=None,
              show_dropped=True,
              list_values=None, max_length=200,
              na_rep='--',
              float_format='{:.2f}'.format, **kw):

        if len(self) == 0:
            logger.error('Empty table, results are filtered out')
            raise ValueError

        columns = columns or self.columns

        removed_cols = [_ for _ in self.columns if _ not in columns]
        self.drop(removed_cols, axis='columns', inplace=True)

        self.drop(self.index[self.isnull().all(axis=1)], axis=0, inplace=True)

        if len(self) > max_length:
            logger.error('Table too long ({}>{}) '.format(len(self), max_length))
            raise ValueError

        if len(self) == 0:
            logger.error('Empty table, no metrics available')
            raise ValueError

        if list_values:
            try:
                values = set(self.index.get_level_values(list_values))
                for _ in values:
                    print(_)
                return
            except (ValueError, KeyError):
                logger.error('{} not in index'.format(list_values))

        if isinstance(float_format, str):
            float_format = float_format.format

        self.sort_index(inplace=True)
        with pd.option_context("display.date_dayfirst", True, "display.date_yearfirst", False):
            df_str = self.to_string(float_format=float_format, na_rep=na_rep)

        df_width = max(len(_) for _ in df_str.split('\n'))

        print(df_str)

        df_str = ''
        if show_dropped:
            df_str += ''
            df_str += '-' * df_width
            for k, index in self._dropped_index.items():
                v = set_with_nan(index)
                if len(v) > 1:
                    df_str += '\n{:16} [{}]'.format(k, len(v))
                else:
                    df_str += '\n{:16} {}'.format(k, *v)

            print(df_str, file=sys.stdout)

    def to_latex(self, filename=None,
                 columns={'FPR@95': 'fpr', 'AUROC': 'auc'},
                 float_format='{:.3g}', **kw):

        if not filename:
            logger.info('No tex file produced')
            return
        logger.info('Tex file: {}'.format(filename))

        if isinstance(float_format, str):
            float_format = float_format.format

        columns, header = zip(*(t for t in columns.items() if t[1]))
        super().to_latex(filename, float_format=float_format,
                         index_names=False,
                         columns=columns, header=header,
                         escape=True)


if __name__ == '__main__':
    from utils.configdict import ConfigDict
    from utils.logger import set_loggers
    from utils.load import df_results
    import sys

    import argparse

    print(sys.argv)

    sys.exit(0)

    argv = '--load.result_dir ./results/lab-ia/main --ood old_mix'.split()

    argv = None if sys.argv[0] else argv

    config = ConfigDict()

    parser = config.create_parser()

    args, filter_args = parser.parse_known_args(argv)

    config.update(args)

    set_loggers(**config.logger)

    self = df_results(**config.load)
    self.reorder_index_levels(**config.table)

    unknown_args = self.filter_parse_args(parser=parser, argv=filter_args, **config.table)

    print(self.index.names)
    self.sort_index(inplace=True)
    print(self.to_string(**config.table))
