import pandas as pd
import matplotlib.pyplot as plt

from .logger import logger
from utils.scores import has_scores, get_scores


class NoPlotError(ValueError):
    pass


class AxisArray:

    def __init__(self, max_figs=20):

        self.figs = []

        self._last_figkw = {}
        self._iter = iter([])

        self.max_figs = max_figs

    def new_fig(self, **kw):

        self._last_figkw = kw.copy()
        suptitle = kw.pop('suptitle', None)

        n_subplots = kw.pop('n_subplots', None)

        if n_subplots:
            kw.update(dict(nrows=min(2, n_subplots // 3 + 1),
                           ncols=min(n_subplots, 3)))

        fig, axis = plt.subplots(squeeze=False, **kw)

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

        for x_y in plots:
            try:
                x, y = x_y.split(':')
            except ValueError:
                logger.error('Can not plot {}'.format(x_y))
                continue
            try:
                plot_x(df, x=x, column=y, axes=axes, **kw)
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

    return {'-'.join(map(str, key)): df.xs(key, axis=1, level=list(range(n)))
            for key in keys}


def plot_x(df, x=None, column='auc',
           split_levels=0,
           axes=None,  subdir='plots', **kw):

    name, result_directory = '{}:{}--{}'.format(x, column, df.name), df.result_directory

    if not x:
        raise NoPlotError

    if column not in df:
        logger.error('{} is not in table columns'.format(column))
        raise NoPlotError

    if x not in df.index.names:
        logger.error('{} is not in table index, try to add it with --table.show {}'.format(x, x))
        raise NoPlotError

    idx = list(df.index.names)

    while x != idx[-1]:
        idx.pop(-1)

    gb = df.groupby(idx, dropna=False)[column]
    df_count = gb.count()

    if not (df_count == 1).all():
        logger.error('Change table index order such that there is only one entry per {}'.format(x))
        raise NoPlotError

    df = gb.mean()

    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_product((df.index, ['']), names=(x, ''))

    df = df.unstack(x).T

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.map(lambda col: '-'.join(map(str, col)))

    (result_directory / subdir).mkdir(exist_ok=True)
    csv_file = (result_directory / subdir / name).with_suffix('.csv')
    df.to_csv(csv_file)

    axes = axes or AxisArray()

    if split_levels:
        dfs = split_by_column_levels(df, n=split_levels)
    else:
        dfs = {'': df}

    n_subplots = len(dfs)
    axes.new_fig(nrows=1, ncols=1, suptitle='{}:{}'.format(x, column))

    df.plot(ax=next(axes), xlabel=x, ylabel=column)
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
