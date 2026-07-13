from .table import ResDF
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


def scores_stats(df, q=dict(), mean=dict(), std=dict(),
                 skew=dict(),
                 compute=True, max_compute=10, **kw):

    mean = {_: b for _, b in mean.items() if b}
    std = {_: b for _, b in std.items() if b}
    skew = {_: b for _, b in skew.items() if b}

    if not (q or mean or skew) or not compute:
        logger.info('No stats calculated')
        return

    if len(has_scores(df)) > max_compute:
        logger.error('table too long ({}>{}), no stats calculated'.format(len(has_scores(df)), max_compute))
        return

    for _ in q:
        logger.info('{} quantile for p={}'.format(_, q[_]))

    for idx, _, scores in get_scores(df):
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
            s = np.std(conf[label_[_]])
            logger.debug('Std calculated for {}'.format(_))
            df.loc[idx, '{}_STD'.format(_.upper())] = s

        for _ in skew:
            if not skew.get(_):
                continue
            c = conf[label_[_]]
            m = np.mean(c)
            s = np.mean((c - m)**3) / np.std(c)**3
            logger.debug('Skew calculated for {}'.format(_))
            df.loc[idx, '{}_SKEW'.format(_.upper())] = s


def plot_scores(df, plot=True, plots=[], wait=True, **kw):

    if not plot or not plots:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot')

    if 'phase' not in df.index.names and 'phase' in plots:
        logger.error('Will not plot phase (hidden or unique)')
        plots.remove('phase')

    has_plots = False
    if 'hist' in plots:
        has_plots |= bool(plot_hist(df, **kw))

    if 'phase' in plots:
        has_plots |= bool(plot_phase(df, **kw))

    if wait and has_plots:
        i = input()


def plot_hist(df, max_plots=3, **kw):

    hist_kw = kw.get('hist_params', {})

    if len(has_scores(df)) > max_plots:
        logger.error('table too long ({}>{}), no plot'.format(len(has_scores(df)), max_plots))
        return

    for idx, idx_str, scores in get_scores(df):
        fig = plt.figure(idx_str)
        ax = fig.gca()
        conf = scores['conf']
        label = scores['label']

        ax.hist(conf[label >= 0], **hist_kw)
        ax.hist(conf[label < 0], **hist_kw)
        fig.show()

    return True


def plot_phase(df, max_plots=3, **kw):

    df_scores = has_scores(df).unstack('phase')
    if len(df_scores) > max_plots:
        logger.error('table too long ({}>{}), no plot'.format(len(has_scores(df)), max_plots))
        return

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

    return True


if __name__ == '__main__':

    pass
