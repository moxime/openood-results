import pandas as pd
import matplotlib.pyplot as plt

from .logger import logger
from ..scores import has_scores, get_scores


class NoPlotError(ValueError):
    pass


class AxisArray:

    def __init__(self, max_figs=20):

        self.figs = []

        self._last_figkw = {}
        self._iter = iter([])

        self.max_figs = max_figs

    def new_fig(self, nrows=1, ncols=1, nsubplots=None, suptitle=None, **kw):

        self._last_figkw = dict(**kw, nrows=nrows, ncols=ncols, nsubplots=nsubplots, suptitle=suptitle)

        if nsubplots:
            nrows = min(2, nsubplots // 3 + bool(nsubplots % 3))
            ncols = min(nsubplots, 3)

        fig, axis = plt.subplots(squeeze=False, nrows=nrows, ncols=ncols, **kw)

        if suptitle:
            fig.suptitle(suptitle)
        self.figs.append(fig)

        self._iter = iter(axis.flatten())

    def show(self):

        for f in self.figs:
            f.show()

    def plot(self, *a, **kw):
        next(self).plot(*a, **kw)

    def __next__(self):
        try:
            return next(self._iter)
        except StopIteration:
            if len(self.figs) < self.max_figs:
                self.new_fig(**self._last_figkw)
                return next(self)
            raise StopIteration

    def __iter__(self):
        return self


def plot_scores(df, plot=True, plots=[], max_figs=20, wait=True, **kw):

    if not plot or not plots:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot {}'.format(','.join(plots)))

    axes = AxisArray(max_figs=max_figs)

    try:
        if 'hist' in plots:
            plots.remove('hist')
            axes.new_fig(nrows=2, ncols=3, suptitle='Score hist')
            try:
                plot_hist(df, axes=axes, **kw)
            except NoPlotError:
                pass

        if 'phase' in plots:
            plots.remove('phase')
            try:
                plot_phase(df, axes=axes, **kw)
                has_plots = True
            except NoPlotError:
                pass

        if 'boxplots' in plots:
            plots.remove('boxplots')
            try:
                plot_boxplots(df, **kw)
                has_plots = True
            except NoPlotError:
                pass

        for y in plots:
            try:
                plot_x(df, *y.split('-'), axes=axes, **kw)
            except NoPlotError:
                pass

    except StopIteration:
        logger.warning('Tried to make too much figs, stopped at {}'.format(max_figs))

    if not axes.figs:
        logger.info('No plot')
        return
    axes.show()
    if wait:
        input()


def plot_hist(df, max_plots=3, **kw):

    hist_kw = kw.get('hist_params', {})

    if len(has_scores(df)) > max_plots:
        logger.error('table too long ({}>{}), no plot'.format(len(has_scores(df)), max_plots))
        raise NoPlotError

    for idx, idx_str, scores in get_scores(df):
        fig = plt.figure(idx_str)
        ax = fig.gca()
        conf = scores['conf']
        label = scores['label']

        ax.hist(conf[label >= 0], **hist_kw)
        ax.hist(conf[label < 0], **hist_kw)
        fig.show()


def plot_boxplots(df, max_plots=3, **kw):
    raise NoPlotError


def split_by_column_levels(df, n=1):

    keys = df.columns.droplevel(list(range(n, df.columns.nlevels))).unique()
    col_names = df.columns.droplevel(list(range(n, df.columns.nlevels))).names

    if not isinstance(keys, pd.MultiIndex):
        keys = [(k,) for k in keys]

    key_names = {k: '-'.join('{}:{}'.format(c, n) for c, n in zip(col_names, k)) for k in keys}

    return {key_names[key]: df.xs(key, axis=1, level=list(range(n)))
            for key in keys}


def plot_x(df, *columns,
           split_levels=0,
           axes=None,
           subdir='plots',
           logx: False,
           **kw):

    for column in columns:
        if column not in df:
            logger.error('{} is not in table columns'.format(column))
            raise NoPlotError

    x = df.meaningfull_index[-1]

    name, result_directory = '{}:{}--{}'.format(x, '-'.join(columns), df.name), df.result_directory

    gb = df.groupby(df.meaningfull_index, dropna=False)[list(columns)]
    df = gb.mean()

    if isinstance(df.index, pd.MultiIndex):
        df = df.unstack(df.index.names[:-1])

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.reorder_levels([*df.columns.names[1:], df.columns.names[0]])

    # (result_directory / subdir).mkdir(exist_ok=True)
    # csv_file = (result_directory / subdir / name).with_suffix('.csv')
    # df.to_csv(csv_file)

    axes = axes or AxisArray()

    if split_levels:
        dfs = split_by_column_levels(df, n=split_levels)
    else:
        dfs = {'': df}

    axes.new_fig(nsubplots=len(dfs), suptitle='{}:{}'.format(x, '-'.join(columns)))
    for t, df in dfs.items():
        df.plot(ax=next(axes), xlabel=x, title=t, logx=logx)
    logger.debug('Plotting metrics for x={}'.format(x))


def plot_phase(df, max_plots=3, **kw):

    df_scores = has_scores(df).unstack('phase')
    if len(df_scores) > max_plots:
        logger.error('table too long ({}>{}), no plot'.format(len(has_scores(df)), max_plots))
        raise NoPlotError

    for idx, idx_str, scores in get_scores(df, unstack='phase'):
        if any(_ is None for _ in scores.values()):
            continue
        fig = plt.figure(idx_str)
        ax = fig.gca()

        conf_mid = scores['1mid']['conf']
        conf_end = scores['2end']['conf']
        label = scores['1mid']['label']
        ax.scatter(conf_mid, conf_end, s=1, c=label < 0)
        ax.plot([conf_mid.min(), conf_mid.max()], [conf_mid.min(), conf_mid.max()], '--')
        fig.show()
