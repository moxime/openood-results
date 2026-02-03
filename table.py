import yaml
import os
from pathlib import Path
import pandas as pd


metrics_columns = ('FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC')


def read_csv(path, name='ood.csv'):

    path = Path(path)

    if path.is_dir():

        path = path / name

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != '.csv':
        raise ValueError('.csv expected, got {}'.format(path.suffix))

    df = pd.read_csv(path)

    index_labels = df.columns[~df.columns.isin(metrics_columns)]

    df.set_index(list(index_labels), inplace=True, append=False)

    if df.index.name:
        df.index.rename('ood', inplace=True)
    else:
        df.index.rename({'dataset': 'ood'}, inplace=True)

    return df


def fetch_results(directory='./results', csv_name='ood.csv', yml_name='config.yml'):

    d = Path(directory)

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)

    for csv_file in d.glob('**/{}'.format(csv_name)):

        yml_file = csv_file.parent / yml_name

        try:
            with open(yml_file) as f:
                c = yaml.load(f, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files
        except FileNotFoundError:
            c = {'dataset': {'name': 'unknown'}}

        df = read_csv(csv_file)

        for k in c:
            if isinstance(c[k], dict) and 'name' in c[k]:
                df[k] = c[k]['name']
                df.set_index(k, append=True, inplace=True)

        yield df


if __name__ == '__main__':

    import argparse
    import sys

    index_order = ['dataset', 'ood_dataset', 'network',
                   'preprocessor', 'pipeline', 'evaluator',
                   'postprocessor', 'ood', 'epoch']

    parser = argparse.ArgumentParser()

    parser.add_argument('path')
    args = parser.parse_args(None if sys.argv[0] else ['~/openood/results'])

    path = Path(args.path).expanduser()

    df = pd.concat([*fetch_results(path)])

    if 'epoch' in df.index.names:

        last_epoch = df.index.get_level_values('epoch').max()
        i_ = df.index.isin([last_epoch], level='epoch')
        df = df.iloc[i_]

    ordered_index = sorted(df.index.names, key=index_order.index)
    df = df.reorder_levels(ordered_index)

    print(df.to_string())
