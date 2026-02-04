import yaml
from pathlib import Path
import pandas as pd
from utils.config import sample_config

METRICS_COLUMNS = ('FPR@95', 'AUROC', 'AUPR_IN', 'AUPR_OUT', 'ACC')
OOD_CSV = 'ood.csv'
CONFIG_YML = 'config.yml'


def read_csv(path, name=OOD_CSV):

    path = Path(path)

    if path.is_dir():
        path = path / name

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != '.csv':
        raise ValueError('.csv expected, got {}'.format(path.suffix))

    df = pd.read_csv(path)

    index_labels = df.columns[~df.columns.isin(METRICS_COLUMNS)]

    df.set_index(list(index_labels), inplace=True, append=False)

    if df.index.name:
        df.index.rename('ood', inplace=True)
    else:
        df.index.rename({'dataset': 'ood'}, inplace=True)

    return df


def load_config(path, name=CONFIG_YML):

    path = Path(path)

    if path.is_dir():
        path = path / name

    if path.suffix != '.yml':
        raise ValueError('.yml expected, got {}'.format(path.suffix))

    with open(path) as f:
        c = yaml.load(f, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files

    return c


def df_exp(path, csv_name=OOD_CSV, yml_name=CONFIG_YML, key_dict={}, index_dict={}, index_order=[]):
    path = Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)

    df = read_csv(path, name=csv_name)

    try:
        config = load_config(path, name=yml_name)
    except FileNotFoundError:
        config = {}

    params = dict(sample_config(config, key_dict))

    for k, v in params.items():
        df[k] = v

    df.set_index(list(params), append=True, inplace=True)

    return df


def fetch_results(directory='./results', csv_name=OOD_CSV, yml_name=CONFIG_YML):

    d = Path(directory)

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)

    for csv_file in d.glob('**/{}'.format(csv_name)):

        df = read_csv(csv_file)
        try:
            c = load_config(csv_file.parent)
        except FileNotFoundError:
            c = {'dataset': {'name': 'unknown'}}

        for k in c:
            if isinstance(c[k], dict) and 'name' in c[k]:
                df[k] = c[k]['name']
                df.set_index(k, append=True, inplace=True)

        yield df


if __name__ == '__main__':

    import sys

    with open('utils/keys.yml') as f:
        config_dict = yaml.load(f, Loader=yaml.SafeLoader)

    key_dict = config_dict['config_parser']

    path = sys.argv[-1]
    if not path:
        path = '/tmp'

    df = df_exp(path, key_dict=key_dict)

    print(df.head().to_string())
