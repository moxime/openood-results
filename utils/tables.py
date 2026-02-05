from collections import defaultdict
import numpy as np
import pandas as pd
from utils.load import fetch_results


def concatenate(*dfs, index_fill_values={}, **kw):

    index_dict = defaultdict(list)
    for df in dfs:
        for name in df.index.names:
            index_dict[name].append(df.index.names.index(name))

    for _ in index_dict:
        index_dict[_] = np.exp(index_dict[_]).mean()
    # print(dict(index_dict))
    sorted_index = sorted(index_dict, key=index_dict.get)

    df_ = []
    for df in dfs:
        index = df.index

        if not (isinstance(index, pd.MultiIndex)):
            df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])

        index_frame = df.index.to_frame()

        for c in sorted_index:
            if c not in index_frame.columns:
                index_frame[c] = index_fill_values.get(c)

        df.index = pd.MultiIndex.from_frame(index_frame[sorted_index])

        df_.append(df)

    return pd.concat(df_)


def df_results(csv_**kw):

    df = concatenate(*fetch_results(**kw), **kw)


if __name__ == '__main__':
    from configs.configdict import ConfigDict

    df = df_results(**ConfigDict())

    print(len(df))
    print(df.tail().to_string())
