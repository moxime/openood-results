from ..configdict import ConfigDict
from .logger import logger
import time
import re
import sys
import pandas as pd
import numpy as np
import argparse
from pathlib import Path


def inplaceable(func):
    """
    func has to have a inplace kw

    if inplace=False, will first copy self and then execute func on self.copy() with inplace=True
    """

    def modified(self, *a, inplace=False, **kw):
        if not inplace:
            df = self.copy()
            func(df, *a, inplace=True, **kw)
            return df
        kw['inplace'] = True
        return func(self, *a, **kw)

    return modified


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
        self._fullindex_frame = None
        self._fullindex_frame = self.index.to_frame()
        self._fullindex_frame.index = self.index

        self._meaningfull_index = None
        self._meaningfull_index = list(self.index.names)

        if len(self.columns.names) == 1 and not self.columns.name:
            self.columns.name = 'metrics'

    def _attrfrom(self, df):
        self._dropped_index = df._dropped_index.copy()
        self._fullindex_frame = df._fullindex_frame.copy()

        self.result_directory = df.result_directory

    def copy(self, **kw):

        df = type(self)(super().copy(**kw))
        df._attrfrom(self)

        return df

    @property
    def result_directory(self):
        return Path(self._result_directory)

    @property
    def meaningfull_index(self):

        while len(self._meaningfull_index) > 1:
            agg_df = self.groupby(self._meaningfull_index[:-1], dropna=False).count()
            if not (agg_df <= 1).all().all():
                break
            self._meaningfull_index.pop(-1)

        return self._meaningfull_index

    @result_directory.setter
    def result_directory(self, val):
        self._result_directory = None
        self._result_directory = val

    class Subsetter:
        def __init__(self, df, locator, fullindex_locator):
            self.locator = locator
            self.dropped_index = df._dropped_index
            self.fullindex_frame = df._fullindex_frame
            self.fullindex_frame_locator = fullindex_locator
            self._result_directory = df.result_directory

        def __getitem__(self, *vargs, **kwargs):
            df_raw = self.locator.__getitem__(*vargs, **kwargs)
            if not isinstance(df_raw, (pd.Series, pd.DataFrame)):
                return df_raw
            df = ResDF(df_raw)
            df._dropped_index = self.dropped_index.copy()
            df._fullindex_frame = self.fullindex_frame_locator.__getitem__(*vargs, **kwargs)
            df.result_directory = self._result_directory
            return df

        def _getitem_axis(self, *a, **kw):
            return self.locator._getitem_axis(*a, **kw)

        def __setitem__(self, *a, **kw):
            return self.locator.__setitem__(*a, **kw)

    def name(self, ops=[], filters={}, **kw):

        table_name = {}

        for k, kept in filters.get('keep', {}).items():
            if kept:
                table_name[k] = '+'.join(map(str, kept))
            removed = filters.get('removed', {}).get(k)
            if removed:
                table_name[k] = table_name.get(k, '') + ('-' + '-'.join(map(str, removed)))

        opnames = '--'.join(ops)

        if opnames:
            opnames = '--' + opnames

        return '--'.join('{}:{}'.format(k, v) for k, v in table_name.items()) + opnames

    @property
    def loc(self):
        return self.Subsetter(self, super().loc, self._fullindex_frame.loc)

    @inplaceable
    def sort_index(self, *a, **kw):

        self._fullindex_frame.sort_index(*a, **kw)
        super().sort_index(*a, **kw)

    @inplaceable
    def set_index(self, *a, **kw):

        super().set_index(*a, **kw)
        self._fullindex_frame.index = self.index

    @inplaceable
    def reset_index(self, *a, **kw):

        super().reset_index(*a, **kw)

    @inplaceable
    def drop(self, labels=None, **kw):
        if kw.get('axis', 0) in (1, 'columns'):
            return super().drop(labels, **kw)
        self._fullindex_frame.drop(labels, **kw)
        super().drop(labels, **kw)
        self._meaningfull_index = list(self.index.names)

    def _flatten_column(self):

        if not isinstance(self.columns, pd.MultiIndex):
            return

        column_name = '--'.join(self.columns.names)
        self.columns = ['--'.join(map(str, _)) for _ in self.columns]
        self.columns.name = column_name

    def _unflatten_column(self):
        if isinstance(self.columns, pd.MultiIndex):
            return

        colnames = self.columns.name.split('--')
        self.columns = pd.MultiIndex.from_tuples([_.split('--') for _ in self.columns])
        self.columns.names = colnames

    def unstack(self, iname):
        self._unflatten_column()
        d = type(self)(super().unstack(iname))
        d._attrfrom(self)

        d._fullindex_frame = self._fullindex_frame.unstack(iname)
        self._flatten_column()
        d._flatten_column()

        return d

    def stack(self, colname):

        self._unflatten_column()

        df = type(self)(super().stack(colname, future_stack=True))
        df._attrfrom(self)

        df._fullindex_frame = self._fullindex_frame.stack(colname, future_stack=True)

        self._flatten_column()
        df._flatten_column()

        return df

    @property
    def fullindex(self):

        i_frame = self._fullindex_frame.copy()

        while i_frame.columns.nlevels > 1:
            col = i_frame.columns.names[-1]
            i_frame = i_frame.droplevel(col, axis=1).drop(columns=col)
            unique = i_frame.columns[~i_frame.apply(lambda x: x.duplicated(keep=False), axis=1).all()]
            i_frame = i_frame.drop(unique, axis=1)
            i_frame = i_frame.loc[:, ~i_frame.apply(lambda x: x.duplicated(), axis=1).all()]
            i_frame[list(set(unique))] = '<>'

        return pd.MultiIndex.from_frame(i_frame)

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
        fullindex = self.index.copy()
        df._fullindex_frame = fullindex.to_frame()
        df._fullindex_frame.index = df.index

        if self._dropped_index:
            logger.warning('index already dropped returing df.copy()')
            return df
        assert not self._dropped_index

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

        df._fullindex_frame.index = df.index

        df.sort_index(inplace=True)

        df.drop(df.index[df.isnull().all(axis=1)], axis=0, inplace=True)
        df.drop(df.columns[df.isnull().all(axis=0)], axis=1,  inplace=True)

        return df.op(**kw)

    def op(self, ops=[], **kw):

        agg_df = self

        for i, op_arg in enumerate(ops):
            op, args = op_arg.split(':')[0], op_arg.split(':')[1:]

            if op not in ('min', 'max', 'unstack', 'stack'):
                raise NotImplementedError

            if op in ('min', 'max'):
                index_names = agg_df.meaningfull_index
                index_names, last_index = index_names[:-1], index_names[-1]
                column = args[0]

                logger.info('Agg table: {} of {} wrt {}'.format(op, column, last_index))

                ops[i] = '{}:'.format(last_index) + ops[i]

                if op == 'max':
                    idx = agg_df[column].groupby(index_names, dropna=False).idxmax()
                elif op == 'min':
                    idx = agg_df[column].groupby(index_names, dropna=False).idxmin()

                agg_df = agg_df.loc[idx.dropna()]

            if op == 'unstack':
                idx = args[0]
                agg_df = agg_df.unstack(idx)
                logger.info('Unstack {}'.format(idx))

            if op == 'stack':
                col = args[0]
                agg_df = agg_df.stack(col)
                logger.info('Stack {}'.format(col))

        return agg_df

    def _filter_index_by_key(self, key, *values, action='keep', inplace=True, **kw):

        if action in ('rm', 'remove'):
            action = 'remove'

        assert action in ('remove', 'keep')

        if key not in self.index.names:
            raise ValueError('{} not in index names ()'.format(key, self.index.names))

        if action == 'keep':
            return self.drop(self.index[~self.index.isin(values, level=key)], inplace=inplace)

        return self.drop(self.index[self.index.isin(values, level=key)], inplace=inplace)

    def _create_parsers(self, **kw):
        """return two parsers for each index of df multliindex (one
        for keep, one for rmove)

        """

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

    def _parse_args(self, argv, filters={}, **kw):
        """
        update filters with filter args from argv, return unknown args
        """
        keep_parser, rm_parser = self._create_parsers()
        keep_args, unknown_args = keep_parser.parse_known_args(argv)
        rm_args, unknown_args = rm_parser.parse_known_args(unknown_args)
        filters.update(keep={k: v for k, v in vars(keep_args).items() if v is not None},
                       remove={k: v for k, v in vars(rm_args).items() if v is not None})

        return unknown_args

    def filter(self, keep={}, remove={}, **kw):
        for k in set(keep) | set(remove):
            kept = keep.get(k)
            removed = remove.get(k)
            df_len = len(self)
            values_before = set(self.index.get_level_values(k))
            if kept is not None:
                self._filter_index_by_key(k, *kept)
            if removed is not None:
                self._filter_index_by_key(k, *removed, action='remove')
            values = set(self.index.get_level_values(k))
            logger.debug('Filtering {} {}->{} {}'.format(k, df_len, len(self),
                                                         kept if len(values) < len(values_before) else ''))

    def filter_parse_args(self, argv=None, filters={}, **kw):

        t0 = time.time()
        unknown_args = self._parse_args(argv, filters=filters, **kw)
        last = filters.get('remove', {}).pop('last', None)

        # filters has been updated
        self.filter(**filters)

        if last:
            self.reorder_index_levels(index_order=['date', 'job'])
            self.drop(self.index[:-last], inplace=True)

        logger.info('Filtered table of length {} in {:.1f}s'.format(len(self), time.time() - t0))

        self.reorder_index_levels(**kw)

        return unknown_args

    @classmethod
    def concat(cls, dfs, filters={}, **kw):

        dfs = [_.copy() for _ in dfs]
        for df in dfs:
            df.index = df.fullindex

        df = cls(pd.concat(dfs))

        df.result_directory = dfs[0].result_directory

        df.filter(**filters)
        df = df.drop_levels(**kw)

        # df.filter(**filters)
        return df

    def print(self,
              columns=None,
              show_dropped=True,
              list_values=None, max_length=200,
              na_rep='--',
              float_format='{:.3g}'.format,
              name=None,
              subdir='tables',
              **kw):

        if len(self) == 0:
            logger.error('Empty table, results are filtered out')
            raise ValueError

        self = self.copy()

        columns = columns or self.columns

        name = name or self.name(**kw)

        removed_cols = [_ for _ in self.columns if not any(re.match(f, _) for f in columns)]
        self.drop(removed_cols, axis='columns', inplace=True)

        self.drop(self.index[self.isnull().all(axis=1)], axis=0, inplace=True)
        self.drop(self.columns[self.isnull().all(axis=0)], axis=1,  inplace=True)

        if not len(self):
            logger.error('Empty table (no index')
            raise ValueError

        if not len(self.columns):
            logger.error('Empty table (no column)')
            raise ValueError

        (self.result_directory / subdir).mkdir(exist_ok=True)
        self.to_csv(self.result_directory / subdir / (name + '.csv'))

        if len(self) > max_length:
            logger.error('Table too long ({}>{}) '.format(len(self), max_length))
            logger.error('Table index: {}'.format(' '.join(self.index.names)))
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

    parser = config._create_parser()

    args, filter_args = parser.parse_known_args(argv)

    config.update(args)

    set_loggers(**config.logger)

    self = df_results(**config.load)
    self.reorder_index_levels(**config.table)

    unknown_args = self.filter_parse_args(parser=parser, argv=filter_args, **config.table)

    print(self.index.names)
    self.sort_index(inplace=True)
    print(self.to_string(**config.table))
