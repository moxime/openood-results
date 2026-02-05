import logging
import time
import yaml
from pathlib import Path
import pandas as pd

OOD_CSV = 'ood.csv'
CONFIG_YML = 'config.yml'
CONFIG_KEYS = {'dataset': {'name': 'set'}, 'postprocessor': {'name': 'method'}}


logger = logging.getLogger(__name__)


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
    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])
    df.index.rename(csv_index, inplace=True)

    return df


class ConfigLoader(yaml.SafeLoader):
    pass


def config_as_dict(loader, node):
    return loader.construct_mapping(node, deep=True)


ConfigLoader.add_constructor("tag:yaml.org,2002:python/object/new:openood.utils.config.Config",
                             config_as_dict)


def _load_raw_config(path):
    with open(path) as f:
        data = yaml.load(f, Loader=ConfigLoader)

    return data['state']


def load_config(path, config_yml=CONFIG_YML, **kw):

    path = Path(path)

    if path.is_dir():
        path = path / config_yml

    if path.suffix != '.yml':
        raise ValueError('.yml expected, got {}'.format(path.suffix))

    c = _load_raw_config(path)
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
    if parsed_config is not None:
        yield config_keys, parsed_config


def df_exp(path, **kw):

    path = Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)

    df = read_csv(path, **kw)

    logger.debug('Found a csv in {}'.format(path))

    try:
        config = load_config(path, **kw)
        logger.debug('Found a config file in {}'.format(path))
    except FileNotFoundError:
        config = {'dataset': {'name': 'unknown'}}
        logger.debug('Did not find a config file in {}, default one is used'.format(path))

    parsed_config = dict(sample_config(config, **kw))

    for k, v in parsed_config.items():
        df[k] = v
        df.set_index(k, append=True, inplace=True)

    return df


def fetch_results(results_directory='./results', **kw):

    d = Path(results_directory)

    try:
        yield df_exp(d, **kw)
    except FileNotFoundError:
        pass

    for s in [_ for _ in d.iterdir() if _.is_dir()]:

        yield from fetch_results(results_directory=s, **kw)


if __name__ == '__main__':

    import time
    from pathlib import Path
    p = Path('/tmp/config.yml')
    p_ = p.parent / (p.name + '_raw'+p.suffix)

    c = load_config(p)

    with open(p_, 'w') as f:
        yaml.safe_dump(c, f)

    N = 10

    for p in (p, p_):
        t0 = time.time()
        for _ in range(N):
            with open(p) as f:
                c = yaml.load(f, Loader=ConfigLoader)
        t = time.time() - t0

        print(p, '{:.3f}ms'.format(t/N*1e3))
