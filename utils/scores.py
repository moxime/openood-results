from .table import ResDF
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def has_scores(df, **kw):

    assert 'SCORES' in df

    index = df.index
    index = index[df.fullindex.isin([True], level='has_scores')]

    if not len(index):
        logger.error('No scores available')

    if len(index) < len(df):
        logger.warning('Some scores are not available')

    logger.debug('{} results with scores available'.format(len(index)))
    return df.loc[index]


def get_scores(df, **kw):

    df_with_scores = has_scores(df)
    for idx, results in df_with_scores.iterrows():
        yield idx, np.load(results['SCORES'])


def compute_scores_stats(df, q=dict(), mean=dict(), std=dict(),
                         compute=True, max_compute=10, **kw):

    mean = {_: b for _, b in mean.items() if b}
    std = {_: b for _, b in std.items() if b}

    if (not q and not mean) or not compute:
        logger.info('No stats calculated')
        return

    if len(has_scores(df)) > max_compute:
        logger.error('table too long ({}>{}), no stats calculated'.format(len(has_scores(df)), max_compute))
        return

    for idx, scores in get_scores(df):
        conf = scores['conf']
        label = scores['label']
        label_ = {'id': label > 0, 'ood': label <= 0}

        for _ in q:
            if q.get(_) is None:
                continue
            q_ = np.quantile(conf[label_[_]], q[_])
            df.loc[idx, '{}_Q'.format(_.upper())] = q_

        for _ in mean:
            if not mean.get(_):
                continue
            m = np.mean(conf[label_[_]])
            logger.debug('Mean calculated for {}'.format(_))
            df.loc[idx, '{}_M'.format(_.upper())] = m

        for _ in std:
            if not std.get(_):
                continue
            std = np.std(conf[label_[_]])
            logger.debug('Std calculated for {}'.format(_))
            df.loc[idx, '{}_STD'.format(_.upper())] = std


def plot_scores(df, plot=True, plots=[], max_plots=3, wait=True, id_q=0.05, **kw):

    if not plot or not plots:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot')

    if len(has_scores(df)) > max_plots:
        logger.error('table too long ({}>{}), no plot'.format(len(has_scores(df)), max_plots))
        return

    for idx, scores in get_scores(df):
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx_str = ' '.join('{}:{}'.format(n, v) for n, v in zip(df.index.names, idx))
        fig = plt.figure(idx_str)
        plot_hist(scores, fig.gca())
        fig.show()

    if wait:
        input()


def plot_hist(results, ax=None):

    if ax is None:
        ax = plt.gca()

    conf = results['conf']
    label = results['label']

    ax.hist(conf[label >= 0], bins=100)
    ax.hist(conf[label < 0], bins=100)
