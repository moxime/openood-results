import numpy as np
import pandas as pd
from scipy.signal import medfilt, savgol_filter


def max_or_min_by(df, column, gb_indices, op='max'):

    if op == 'max':
        idx = df[column].groupby(list(gb_indices), dropna=False).idxmax()

    elif op == 'min':
        idx = df[column].groupby(list(gb_indices), dropna=False).idxmin()

    return df.loc[idx.dropna()]


def max_cutoff(df, column, gb_indices, cutoff=0.97):
    gb = df[column].groupby(list(gb_indices), dropna=False)

    idx = gb.apply(lambda s: s.index[np.flatnonzero(s.to_numpy() >= cutoff * s.max())[0]])

    return df.loc[idx.dropna()]


def smooth(df, column, gb_indices, n=3, **kw):

    def smooth_col(x):
        return x.rolling(n, center=True, min_periods=1, win_type='hamming').mean()

    def median(x):
        return medfilt(x, kernel_size=n)

    def savgol(x):
        return savgol_filter(x, window_length=n, polyorder=2)

    df[column + '~'] = df[column].groupby(list(gb_indices), dropna=False).transform(smooth_col)

    return df


def mean(df, gb_indices, lost_str='--', avg_str='<:>', **kw):

    mean_cols = df.select_dtypes(include="number").columns
    out = df[mean_cols].groupby(list(gb_indices), dropna=False).mean()
    out = out.reindex(columns=df.columns)

    remaining_indices = [_ for _ in df.index.names if _ not in gb_indices]
    avged_index = remaining_indices[0]
    out[avged_index] = avg_str

    if remaining_indices[1:]:
        out[remaining_indices[1:]] = lost_str

    out = type(df)(out.set_index(remaining_indices, append=True))  # type(df) is ResDF

    out._attrfrom(df)

    out._fullindex_frame = out._fullindex_frame.groupby(level=gb_indices).first()
    out._fullindex_frame.index = out.index

    fi_cols = out._fullindex_frame.columns
    for c in fi_cols[fi_cols.get_loc(avged_index):]:
        if len(out._fullindex_frame[c].unique()) > 1:
            out._fullindex_frame[c] = lost_str
    out._fullindex_frame[avged_index] = avg_str
    return out


def do_op(df, *a, op='min', **kw):

    if op in ('min', 'max'):
        return max_or_min_by(df, *a, op=op)

    if op == 'max_cutoff':
        return max_cutoff(df, *a)

    if op == 'smooth':
        return smooth(df, *a, **kw)

    if op == 'mean':
        return mean(df, *a, **kw)


if __name__ == '__main__':

    import numpy as np
    import matplotlib.pyplot as plt
    from . import ResDF
    import random
    import string

    op = None
    op = 'smooth'
    op = 'mean'
    op = 'max'
    op = 'max_cutoff'

    L = list('XY')
    n_epochs = 10

    if op == 'smooth':
        L = list('XY')
        n_epochs = 50
    elif op == 'mean':
        L = list('X')
        L = list('XY')
        n_epochs = 5

    index = pd.MultiIndex.from_product((L, ['a', 'b'], range(n_epochs)), names=('L', 'm', 'e'))

    T = 1000
    alpha = 0.001
    e = index.get_level_values('e')
    v = np.cos(2 * np.pi * (e - T/4) / T) + alpha * np.random.randn(len(index))
    df = pd.DataFrame.from_dict({'v': v})
    df.index = index

    df['l'] = random.choices(string.ascii_lowercase, k=len(index))
    df.set_index('l', append=True, inplace=True)

    df['l_'] = random.choices(string.ascii_lowercase, k=len(index))
    df.set_index('l_', append=True, inplace=True)

    df['u'] = 'A'
    df.set_index('u', append=True, inplace=True)

    df['scores'] = random.choices(string.ascii_lowercase, k=len(index))
    df = ResDF(df)
    df.result_directory = '/tmp'

    df = df.drop_levels(hide=['l_'])

    if op == 'smooth':
        df_ = smooth(df, 'v', ['L', 'm'], n=10)
        df_.plot()
        plt.show(block=False)

    elif op == 'mean':
        df_ = mean(df, 'v', ['L', 'm'])

    elif op == 'max':
        df_ = max_or_min_by(df, 'v', ['L', 'm'])

    elif op == 'max_cutoff':
        df_ = max_cutoff(df, 'v', ['L', 'm'], cutoff=0.5)
