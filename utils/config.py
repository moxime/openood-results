import argparse
import yaml

PARAMS_YML = 'configs/params.yml'


def sample_config(config, key_dict, prefix=''):
    """sample config dict wrt a key_dict

    --config is a config dict (loaded from config.yml)


    -- key_dict is a dict-like tree with leaves as key

    Return: a dict on the form key_name: val if

    k1 : k2: key: key_name is in key_dict and k1: k2: key_val is in
    config

    """

    if isinstance(key_dict, dict):
        for k in key_dict:
            if k in config:
                yield from sample_config(config[k], key_dict[k], prefix=prefix)

        return

    if isinstance(config, dict):
        for k in config:
            yield from sample_config(config[k], '{}_{}'.format(key_dict, k))

        return

    yield prefix + key_dict, config


def add_filter_args_to_parser(parser, key_dict, *parsed_configs):

    if 'key' in key_dict:
        key_name = key_dict['key']
        types = set(type(config.get(key_name)) for config in parsed_configs)
        aliases = ['-' + _ if len(_) == 1 else '--' + _ for _ in key_dict.get('alias', [])]

        type = None
        if len(types) == 1:
            type = next(iter(types))

        parser.add_argument('--' + key_name, *aliases, type=type)
        return

    if 'key_prefix' in key_dict:
        return

    for k in key_dict:
        add_filter_args_to_parser(parser, key_dict[k], *parsed_configs)


if __name__ == '__main__':

    from utils.load import load_config
    import argparse
    import time

    cdir = 'results/ash'
    cdir = '/tmp/bogus'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', default=cdir)
    args = parser.parse_args()

    cdir = args.d

    t = time.time()
    config = load_config(cdir)
    print('{:.0f} ms'.format((time.time() - t)*1e3))
    with open(PARAMS_YML) as f:
        key_dict = yaml.load(f, Loader=yaml.SafeLoader)

    d = dict(sample_config(config, key_dict))

    print(d)

    parser = argparse.ArgumentParser()

    # add_filter_args_to_parser(parser, key_dict, d)
