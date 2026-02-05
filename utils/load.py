import yaml
from pathlib import Path
import pandas as pd

OOD_CSV = 'ood.csv'
CONFIG_YML = 'config.yml'
CONFIG_KEYS = {'dataset': {'name': 'set'}, 'postprocessor': {'name': 'method'}}


def read_csv(path, ood_csv=OOD_CSV, csv_index={'dataset': 'ood', 'epoch': 'epoch'},  **kw):

    path = Path(path)

    if path.is_dir():
        path = path / ood_csv

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != '.csv':
        raise ValueError('.csv expected, got {}'.format(path.suffix))

    df = pd.read_csv(path)

    index_labels = df.columns[df.columns.isin(list(csv_index))]

    df.set_index(list(index_labels), inplace=True, append=False)

    if len(df.index.names) == 1:
        if df.index.name in csv_index:
            df.index.rename(csv_index[df.index.name], inplace=True)
    else:
        df.index.rename(csv_index, inplace=True)

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


def load_config(path, config_yml=CONFIG_YML, **kw):

    path = Path(path)

    if path.is_dir():
        path = path / config_yml

    if path.suffix != '.yml':
        raise ValueError('.yml expected, got {}'.format(path.suffix))

    c = load_raw_config(path)
    # with open(path) as f:
    #    c = yaml.load(f, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files

    return c


def sample_config(parsed_config, config_keys=CONFIG_KEYS, **kw):
    """sample config dict wrt a key_dict

    --config is a config dict (loaded from config.yml)


    -- key_dict is a dict-like tree

    Return: a dict on the form key_name: val if

    k1 : k2: key_name is in key_dict and k1: k2: key_val is in
    config

    """
    if isinstance(config_keys, dict):
        for k in config_keys:
            if k in parsed_config:
                yield from sample_config(parsed_config[k], config_keys=config_keys[k])

        return

    if isinstance(parsed_config, dict):
        for k in parsed_config:
            yield from sample_config(parsed_config[k], config_keys='{}_{}'.format(config_keys, k))

        return

    yield config_keys, parsed_config


def df_exp(path, **kw):
    path = Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)

    df = read_csv(path, **kw)

    try:
        config = load_config(path, **kw)
    except FileNotFoundError:
        config = {}

    params = dict(sample_config(config, **kw))

    for k, v in params.items():
        df[k] = v

    df.set_index(list(params), append=True, inplace=True)

    return df


def fetch_results(directory='./results', **kw):

    d = Path(directory)

    ood_csv = kw.get('ood_csv', OOD_CSV)

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError(d)

    for csv_file in d.glob('**/{}'.format(ood_csv)):

        df = read_csv(csv_file, **kw)
        try:
            c = load_config(csv_file.parent, **kw)
        except FileNotFoundError:
            c = {'dataset': {'name': 'unknown'}}

        c = dict(sample_config(c, **kw))

        for k in c:
            df[k] = c[k]
            df.set_index(k, append=True, inplace=True)

        yield df


if __name__ == '__main__':
    from configs.configdict import ConfigDict

    df_ = list(fetch_results(**ConfigDict()))

    index_names = set(sum([list(df.index.names) for df in df_], start=[]))
