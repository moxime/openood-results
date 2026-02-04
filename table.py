from pathlib import Path
import pandas as pd
from utils import fetch_results


def filter_df(df, **kw):

    i_ = ~df.isin([])

    for k, v in kw.items():

        i_ = i_ & df.index.isin(v, level=k)


if __name__ == '__main__':

    import argparse
    import sys

    index_order = ['dataset', 'ood_dataset', 'network',
                   'preprocessor', 'pipeline', 'evaluator',
                   'postprocessor', 'ood', 'epoch']

    parser = argparse.ArgumentParser()

    parser.add_argument('path')
    args, filtering_args = parser.parse_known_args(None if sys.argv[0] else ['~/openood/results'])

    path = Path(args.path).expanduser()

    df = pd.concat([*fetch_results(path)])

    if 'epoch' in df.index.names:

        last_epoch = df.index.get_level_values('epoch').max()
        i_ = df.index.isin([last_epoch], level='epoch')
        df = df.iloc[i_]

    dropped_index = {}
    for name in df.index.names:

        # if only one value
        values = set(df.index.get_level_values(name))
        if len(values) == 1:
            df.index = df.index.droplevel(name)
            dropped_index[name] = next(iter(values))

    ordered_index = sorted(df.index.names, key=index_order.index)
    df = df.reorder_levels(ordered_index)

    print(df.to_string())
    print(dropped_index)
