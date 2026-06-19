import logging
import os
import time
import yaml
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from .table import ResDF

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

    if 'epoch' in df:
        epochs = max(df['epoch'])
        df['phase'] = df['epoch'].map(lambda e: {0: '0start', epochs // 4: '1mid', epochs: '2end'}.get(e))

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
    date = pd.Timestamp(os.path.getmtime(path), unit="s", tz='Europe/Paris')
    c['exp_date'] = date
    # date = pd.Timestamp(os.path.getctime(path), unit="s")
    # c['create_date'] = date
    #    c = yaml.load(f, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files

    return c


def save_config(path, c, config_keys=CONFIG_YML, **kw):

    saved_conf = dict(state_dict=c)
    # TBF


def sample_config(parsed_config, key, **config_keys):
    """sample config dict wrt a key_dict

    --config is a config dict (loaded from config.yml)


    -- key_dict is a dict-like tree

    Return: a dict on the form key_name: val if

    k1 : k2: key_name is in key_dict and k1: k2: key_val is in
    config

    """

    assert key is None or not config_keys

    if parsed_config is None:
        return

    if config_keys:
        for k, v in config_keys.items():
            if k in parsed_config:
                if isinstance(v, dict):
                    yield from sample_config(parsed_config[k], None, **v)
                elif isinstance(v, (tuple, list)):
                    for _ in v:
                        yield from sample_config(parsed_config[k], _)
                else:
                    yield from sample_config(parsed_config[k], v)
        return

    if isinstance(parsed_config, dict):
        for k in parsed_config:
            yield from sample_config(parsed_config[k], '{}_{}'.format(key, k))
        return

    yield key, parsed_config


def df_exp(path, root='./results', **kw):
    """
    kw[load] forwarded to read_csv, load_config,
    kw[config_keys] forwarded to sample_config

    """
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

    parsed_config = dict(sample_config(config, None, **kw['config_keys']))

    parsed_config['path'] = path

    for k, v in parsed_config.items():
        if isinstance(v, list):
            v = '-'.join(sorted(v))
        df[k] = v
    df.set_index(list(parsed_config), append=True, inplace=True)
    logger.debug('df ({}) filled up with {} indexes'.format(len(df), len(parsed_config)))
    return df


def fetch_results(results_directory='./results', root=None, **kw):
    """
    kw forwarded to df_exp
    """
    d = Path(results_directory)
    if root is None:
        root = d

    try:
        yield df_exp(d, root=root, **kw)
    except FileNotFoundError:
        for s in [_ for _ in d.iterdir() if _.is_dir()]:
            yield from fetch_results(results_directory=s, root=root, **kw)


def df_results(df_columns={'FPR@95': 'fpr', 'AUROC': 'auc'},
               parse_dates=['date'], flash=False, **kw):
    """

    """
    t0 = time.time()
    res_dir = kw.get('results_directory')
    csv_path = Path(res_dir) / 'table.csv'

    if flash:
        try:
            df = ResDF(pd.read_csv(csv_path, parse_dates=parse_dates))
            df.set_index([_ for _ in df.columns if _ not in df_columns], inplace=True)
        except FileNotFoundError:
            logger.warning('Flash df is true but {} does not exist, will fetch results'.format(csv_path))
            flash = False

    if not flash:

        logger.info('Looking for results in {}'.format(res_dir))
        list_of_dfs = list(fetch_results(**kw))
        logger.info('Found {} results in {:.1f}s'.format(len(list_of_dfs), time.time() - t0))
        df = concatenate_df(*list_of_dfs, **kw)
        df.to_csv(csv_path)
        logger.info('Table saved in {}'.format(csv_path))

    removed_cols = [_ for _ in df.columns if not df_columns.get(_)]

    df.drop(removed_cols, axis='columns', inplace=True)

    df.rename(columns=df_columns, inplace=True)

    t0 -= time.time()

    logger.info('Loaded {} lines in {:.1f}s'.format(len(df), -t0))
    return df


def concatenate_df(*dfs, index_fill_values={}, **kw):

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

    return ResDF(pd.concat(df_))


if __name__ == '__main__':

    import time
    import sys
    from pathlib import Path
    p = Path('/tmp/config.yml')
    c = load_config(p)

    yaml.dump(c, stream=sys.stdout,
              default_flow_style=False,
              sort_keys=False,
              indent=2)
