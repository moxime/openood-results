import yaml
from pathlib import Path
import pandas as pd
from configs.config import default_parse_config

OOD_CSV = 'ood.csv'
CONFIG_YML 'config.yml'

METRICS_COLUMNS = default_parse_config['csv']['columns']
CSV_HEADER = default_parse_config['csv']['header']
PARAMS_KEYS = default_parse_config['params']


def read_csv(path, name=OOD_CSV, index_dict=CSV_HEADER, metrics=METRICS_COLUMNS, **kw):

    path = Path(path)

    if path.is_dir():
        path = path / name

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != '.csv':
        raise ValueError('.csv expected, got {}'.format(path.suffix))

    df = pd.read_csv(path)

    index_labels = df.columns[~df.columns.isin(list(metrics))]

    df.set_index(list(index_labels), inplace=True, append=False)

    if len(df.index.names) == 1:
        if df.index.name in index_dict:
            pass
            df.index.rename(index_dict[df.index.name], inplace=True)
    else:
        df.index.rename(index_dict, inplace=True)

    return df


class ConfigLoader(yaml.SafeLoader):
    pass


def config_as_dict(loader, node):
    return loader.construct_mapping(node, deep=True)


ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/new:openood.utils.config.Config",
                             config_as_dict)


def load_raw_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=ConfigLoader)

    return data['state']


def load_config(path, name=CONFIG_YML, **kw):

    path = Path(path)

    if path.is_dir():
        path = path / name

    if path.suffix != '.yml':
        raise ValueError('.yml expected, got {}'.format(path.suffix))

    c = load_raw_config(path)
    # with open(path) as f:
    #    c = yaml.load(f, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files

    return c


def sample_config(config, key_dict=PARAMS_KEYS, **kw):
    """sample config dict wrt a key_dict

    --config is a config dict (loaded from config.yml)


    -- key_dict is a dict-like tree

    Return: a dict on the form key_name: val if

    k1 : k2: key_name is in key_dict and k1: k2: key_val is in
    config

    """

    if isinstance(key_dict, dict):
        for k in key_dict:
            if k in config:
                yield from sample_config(config[k], key_dict=key_dict[k])

        return

    if isinstance(config, dict):
        for k in config:
            yield from sample_config(config[k], key_dict='{}_{}'.format(key_dict, k))

        return

    yield key_dict, config


def fetch_results(directory='./results', ood_csv=OOD_CSV, config_yml=CONFIG_YML, **kw):

    d = Path(directory)

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)

    for csv_file in d.glob('**/{}'.format(ood_csv)):

        df = read_csv(csv_file, **kw)
        try:
            c = load_config(csv_file.parent, name=config_yml)
        except FileNotFoundError:
            c = {'dataset': {'name': 'unknown'}}

        c = dict(sample_config(c, **kw))

        for k in c:
            df[k] = c[k]
            df.set_index(k, append=True, inplace=True)

        yield df


if __name__ == '__main__':

    df = fetch_results()
