from .table import ResDF
import logging
import numpy as np

logger = logging.getLogger(__name__)


def plot_scores(df, plot=True, max_plots=3, **kw):
    if not plot:
        logger.info('Do not plot')
        return
    else:
        logger.info('Tries to plot')

    df_len = len(df)

    i = df.index
    df = df.drop(i[i.isin([False], level='has_scores')])
    if not len(df):
        logger.error('Nothing left to plot')
        raise ValueError

    if len(df) < df_len:
        logger.waring('Some scores are not available')

    if len(df) > max_plots:
        logger.error('Kept {}>{} plots, try to add filters'.format(len(df), max_plots))
        raise ValueError

    for i, results in df.iterrows():

        scores = np.load(results['SCORES'])
        print(list(scores.keys()))
