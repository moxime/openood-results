import pandas as pd
import matplotlib.pyplot as plt

from .logger import logger
from utils.scores import has_scores, get_scores


class NoPlotError(ValueError):
    pass


def plot_scores(df, plot=True, plots=[], wait=True, **kw):

    if not plot or not plots:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot {}'.format(','.join(plots)))

    if 'phase' not in df.index.names and 'phase' in plots:
        logger.error('Will not plot phase (hidden or unique)')
        plots.remove('phase')

    has_plots = False
    if 'hist' in plots:
        plots.remove('hist')
        try:
            plot_hist(df, **kw)
            has_plots = True
        except NoPlotError:
            pass

    if 'phase' in plots:
        plots.remove('phase')
        try:
            plot_phase(df, **kw)
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
        x_y = x_y.split(':')
        if not len(x_y) == 2:
            logger.error('Can not plot {}'.format(':'.join(x_y)))
            continue
        try:
            plot_x(df, x_y[0], column=x_y[1:], **kw)
            has_plots = True
        except NoPlotError:
            pass

    if not has_plots:
        logger.info('No plot')
    if wait and has_plots:
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


def plot_x(df, x=None, max_plots=3, column=[], **kw):

    if not x:
        raise NoPlotError

    df = has_scores(df).select_dtypes('float')

    if x not in df.index.names:
        logger.error('{} is not in table index, try to add it with --table.show {}'.format(x, x))
        raise NoPlotError

    df = df.unstack(x)

    if isinstance(df, pd.Series):
        df = pd.DataFrame(df).T
        df.index = pd.MultiIndex.from_tuples([('result',)], names=[''])

    for idx, row in df.iterrows():
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx_str = ' '.join('{}:{}'.format(n, i) for n, i in zip(df.index.names, idx))
        logger.debug('Plotting metrics for x={} for {}'.format(x, idx_str))
        fig = plt.figure(idx_str)
        ax = fig.gca()
        row_df = pd.DataFrame(row).unstack(x).T
        no_plot = True
        for c in row_df:
            if column and c not in column:
                continue
            label = c
            series = row_df[c]
            if series.isnull().all():
                continue
            no_plot = False
            ax.plot(row_df.index.get_level_values(x), series, label=label)
        if not no_plot:
            ax.legend()
            fig.show()
        else:
            raise NoPlotError


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
