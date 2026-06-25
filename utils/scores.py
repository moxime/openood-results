from .table import ResDF
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
logger = logging.getLogger(__name__)


def plot_scores(df, plot=True, max_plots=3, wait=True, id_q=0.05, **kw):

    if not plot:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot')

    df_len = len(df)
    i = df.index
    df.drop(i[i.isin([False], level='has_scores')], inplace=True)
    df.index = df.index.droplevel('has_scores')
    if not len(df):
        logger.error('Nothing left to plot')
        raise ValueError

    if len(df) < df_len:
        logger.warning('Some scores are not available')

    if len(df) > max_plots:
        logger.error('Kept {}>{} plots, try to add filters'.format(len(df), max_plots))
        raise ValueError

    for idx, results in df.iterrows():
        if not isinstance(idx, tuple):
            idx = (idx,)
        idx_str = ' '.join('{}:{}'.format(n, v) for n, v in zip(df.index.names, idx))
        scores = np.load(results['SCORES'])
        conf = scores['conf']
        label = scores['label']
        q = np.quantile(conf[label >= 0], id_q)
        print(idx_str, len(scores['conf']), 'q={:.1f}'.format(q))
        #        print(df.loc[idx])
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
