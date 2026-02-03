import yaml
import openood
from openood.utils.config import Config
import os
from pathlib import Path


class ConfigFilter():

    pass


def fetch_configs(directory='./results', name='config.yml', **kw):

    d = Path(directory)

    if not d.exists() or not d.is_dir():
        raise FileNotFoundError

    for f in d.glob('**/{}'.format(name)):

        with open(f) as f_:
            c = yaml.load(f_, Loader=yaml.UnsafeLoader)  # DANGEROUS on untrusted files
            c['path'] = f.parent
            yield c


if __name__ == '__main__':

    config_keys = ('dataset', 'network', 'ood_dataset', 'pipeline',
                   'evaluator', 'preprocessor', 'postprocessor')

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('d')

    args = parser.parse_args()

    for c in fetch_configs(args.d, yield_path=True):
        print('*' * 80)
        print(c['path'], len(c), '\n',  *c)
        for k in config_keys:
            print('***** {} *****'.format(k))
            print(c[k]['name'])
