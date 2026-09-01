import logging
import os
import yaml
import time
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from .table import ResDF

OOD_CSV = 'ood.csv'
CONFIG_YML = 'config.yml'
CONFIG_KEYS = {'dataset': {'name': 'set'}, 'postprocessor': {'name': 'method'}}


logger = logging.getLogger(__name__)


class DeleledRes(Exception):
    pass


def read_csv(path, ood_csv=OOD_CSV, csv_index={'dataset': 'ood', 'epoch': 'epoch'},
             ignore_deleted=False, **kw):

    path = Path(path)

    if path.is_dir():

        if (path / 'deleted').exists() and not ignore_deleted:
            raise DeleledRes(path)

        path = path / ood_csv

    if not path.exists():
        raise FileNotFoundError(path)

    if path.suffix != '.csv':
        raise ValueError('.csv expected, got {}'.format(path.suffix))

    df = pd.read_csv(path)

    index_labels = df.columns[df.columns.isin(list(csv_index))]
    df.set_index(list(index_labels), inplace=True, append=False)

    scores_dict = score_paths(path.parent)
    for i, s in scores_dict.items():
        df.loc[i, 'SCORES'] = s

    if 'epoch' in df.index.names:
        epochs_ = df.index.get_level_values('epoch')
        epochs = max(epochs_)
        df['phase'] = epochs_.map(lambda e: {0: '0start', epochs // 4: '1mid', epochs: '2end'}.get(e))
        df.set_index('phase', append=True, inplace=True)

    if not isinstance(df.index, pd.MultiIndex):
        df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])
    df.index.rename(csv_index, inplace=True)

    df.drop(df.index[df.isnull().all(axis=1)], inplace=True)

    df['has_scores'] = ~df['SCORES'].isna()
    df.set_index('has_scores', append=True, inplace=True)

    return df


def score_paths(path):

    def tryint(s):

        try:
            return int(s)
        except ValueError:
            return s

    path = Path(path)

    return {(*[], *map(tryint, _.parent.name.split('-')[1:]), _.stem): _  # .relative_to(path)
            for _ in path.glob('**/*.npz')}


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
    c['path'] = path
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


def df_exp(path, root='./results', config_keys={'foo': 'bar'}, **kw):
    """

    """
    path = Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)

    df = read_csv(path, **kw)
    df_len = len(df)

    logger.debug('Found a csv of len {} in {}'.format(df_len, path))

    try:
        config = load_config(path, **kw)
        logger.debug('Found a config file in {}'.format(path))
    except FileNotFoundError:
        config = {'dataset': {'name': 'unknown'}}
        logger.debug('Did not find a config file in {}, default one is used'.format(path))

    parsed_config = dict(sample_config(config, None, **config_keys))

    for k, v in parsed_config.items():
        if isinstance(v, list):
            v = '-'.join(map(str, sorted(v)))
        df[k] = v
    df.set_index(list(parsed_config), append=True, inplace=True)

    try:
        df['path'] = path.relative_to(root)
    except ValueError:
        df['path'] = path
    df.set_index('path', append=True, inplace=True)
    logger.debug('df ({}) filled up with {} indexes'.format(len(df), len(parsed_config)))
    return df


def fetch_results(result_directory='./results', root=None, **kw):
    """
    kw forwarded to df_exp
    """
    d = Path(result_directory)
    if root is None:
        root = d

    try:
        yield df_exp(d, root=root, **kw)
    except FileNotFoundError:
        for s in [_ for _ in d.iterdir() if _.is_dir()]:
            yield from fetch_results(result_directory=s, root=root, **kw)
    except DeleledRes:
        pass


def df_results(result_directory='./results', parse_dates=['date'], flash=False, **kw):
    """

    """
    t0 = time.time()
    csv_path = Path(result_directory) / 'table.csv'

    if flash:

        try:
            df = ResDF(pd.read_csv(csv_path, parse_dates=parse_dates, low_memory=False))
            i = list(df).index('/')
            df.set_index(list(df)[:i], inplace=True)
            df.drop('/', axis=1, inplace=True)
            logger.info('Flashed table from {}'.format(csv_path))
        except FileNotFoundError:
            logger.warning('Flash df is true but {} does not exist, will fetch results'.format(csv_path))
            flash = False
        except ValueError:
            logger.warning('Flash df is true but "/" col does not exist, will fetch results'.format(csv_path))
            flash = False

    if not flash:

        logger.info('Walking through {} for results'.format(result_directory))
        list_of_dfs = list(fetch_results(result_directory=result_directory, **kw))
        logger.info('Found {} results in {:.1f}s'.format(len(list_of_dfs), time.time() - t0))
        df = concatenate_df(*list_of_dfs, **kw)
        df.insert(0, '/', None)
        df.to_csv(csv_path)
        logger.info('Table saved in {}'.format(csv_path))
        df.drop('/', axis=1, inplace=True)

    t0 -= time.time()

    logger.info('Loaded {} lines in {:.1f}s'.format(len(df), -t0))

    df.result_directory = None
    df.result_directory = result_directory

    return df


def concatenate_df(*dfs, index_fill_values={}, **kw):

    index_dict = defaultdict(list)
    for df in dfs:
        for name in df.index.names:
            index_dict[name].append(df.index.names.index(name))

    for _ in index_dict:
        index_dict[_] = np.exp(index_dict[_]).mean()
    sorted_index = sorted(index_dict, key=index_dict.get)

    df_ = []
    for df in dfs:
        index = df.index

        if not (isinstance(index, pd.MultiIndex)):
            df.index = pd.MultiIndex.from_arrays([df.index], names=[df.index.name])

        index_frame = df.index.to_frame()

        for c in sorted_index:
            if c not in index_frame.columns:
                index_frame[c] = index_fill_values.get(c, pd.NA)

        df.index = pd.MultiIndex.from_frame(index_frame[sorted_index])

        df_.append(df)

    return ResDF(pd.concat(df_))


if __name__ == '__main__':

    import time
    import sys
    from pathlib import Path

    print(sys.argv[1])
    df = df_results(sys.argv[1])

    print('Index\n'+'\n'.join(df.index.names))
    print('Cols\n' + '\n'.join(df.columns))

    print(df.to_string())
