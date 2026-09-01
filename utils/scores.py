from .table import ResDF
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats import stattools
from scipy import stats

logger = logging.getLogger(__name__)


def has_scores(df, **kw):

    index = df.fullindex
    index = df.index[index.isin([True], level='has_scores')]

    if not len(index):
        logger.error('No scores available')

    if len(index) < len(df):
        logger.warning('Some scores are not available')

    logger.debug('{} results with scores available'.format(len(index)))
    return df.loc[index]


def get_scores(df, unstack=[], **kw):

    df_with_scores = has_scores(df)

    if unstack:
        df_with_scores = df_with_scores.unstack(unstack)
    for idx, results in df_with_scores.iterrows():
        if not isinstance(idx, tuple):
            idx_str = str(idx)
        else:
            idx_str = ' '.join('{}:{}'.format(n, v) for n, v in zip(df.index.names, idx))

        if isinstance(results['SCORES'], (str, Path)):
            yield idx, idx_str, np.load(results['SCORES'])
        else:
            results = dict(results['SCORES'])
            yield idx, idx_str, {_: np.load(results[_]) for _ in results}


def scores_stats(df, compute=True, q=dict(),
                 fisher=True, max_compute=10, **kw):

    funcs = {'mean': np.mean, 'std': np.std,
             'skew': stats.skew, 'kurtosis': stats.kurtosis,
             'medcouple': stattools.medcouple,
             # 'iqr': lambda x: (np.mean(x) - np.quantile(x, 0.1)) / np.std(x)
             'iqr': lambda x: (np.quantile(x, 0.5) - np.quantile(x, 0.1)) / np.std(x)
             }

    if not compute:
        logger.info('No stats calculated')
        return

    if len(has_scores(df)) > max_compute:
        logger.error('table too long ({}>{}), no stats calculated'.format(len(has_scores(df)), max_compute))
        return

    for _ in q:
        if q[_]:
            logger.info('{} quantile for p={}'.format(_, q[_]))

    for stat in funcs:
        if kw.get(stat):
            logger.info('Will calculate {} for {}'.format(stat, ','.join(kw[stat])))

    for idx, _, scores in get_scores(df):
        conf = scores['conf']
        label = scores['label']
        label_ = {'id': label >= 0, 'ood': label < 0}

        c = {_: conf[label_[_]] for _ in label_}

        skip = []
        for _ in label_:
            if not len(label[label_[_]]):
                logger.debug('No {} samples, removed from stats'.format(_))
                skip.append(_)
                continue

            if q.get(_):
                q_ = np.quantile(c[_], q[_])
                df.loc[idx, '{}_Q'.format(_.upper())] = q_
                logger.debug('Quantile calculated for {} [{}]'.format(_, len(c[_])))

            for stat in funcs:
                if _ not in kw.get(stat, []):
                    continue
                if len(c[_]) > 10000 and stat == 'medcouple':
                    logger.debug('Will not calculate medcouple for {} [{}] (too long)'.format(_, len(c[_])))
                    continue
                logger.debug('{} calculated for {} [{}]'.format(stat, _, len(c[_])))
                func = funcs[stat]
                df.loc[idx, '{}_{}'.format(_.upper(), stat.upper())] = func(c[_])

        if fisher and not skip:
            fisher_id_ood = (c['id'].mean() - c['ood'].mean())**2 / (c['id'].var() + c['ood'].var())
            df.loc[idx, 'FISHER'] = fisher_id_ood


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
        i = input()


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


if __name__ == '__main__':

    pass
