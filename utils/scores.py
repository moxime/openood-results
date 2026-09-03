import time
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


def get_scores(df, **kw):

    df_with_scores = has_scores(df)

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


def compute_scores_stats(df, compute=True, q=dict(),
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

    t0 = time.time()

    df.sort_index(inplace=True)

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

    logger.debug('Scores stats calculated in {:.2f}s'.format(time.time() - t0))


if __name__ == '__main__':

    from . import df_results

    df = df_results('/tmp/results/', flash=False)

    # df.filter_index('ood', 'val', action='rm')

    index_names = list(df.index.names)

    index_names.remove('epoch')

    df.reorder_index_levels([*index_names, 'eopch'])

    idx = df['AUROC'].groupby(index_names).idxmax()

    print(idx.dropna())
    print(df.drop(index=idx.index).to_string())
