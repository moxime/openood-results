import logging
import time
import yaml
from pathlib import Path
import argparse
from argparse import Namespace
from importlib.resources import files
import sys
import os
import pandas as pd
from functools import partialmethod

try:
    config_root = Path(__file__).parent.parent / 'configs'
except NameError:
    config_root = Path('configs')

KEYS_YML = config_root / 'config_keys.yml'
CSV_YML = config_root / 'table.yml'
MAIN_YML = config_root / 'main.yml'


logger = logging.getLogger(__name__)


class ConfigDict(dict):

    def __init__(self, /, *a, **kw):

        super().__init__()

        if '_registering_default' not in kw:
            for yml_file in (MAIN_YML, CSV_YML, KEYS_YML):
                self._update(0, **yaml.load(open(yml_file), Loader=yaml.SafeLoader))

        self.update(*a, **kw)

    def __repr__(self, prefix='', indent=2):

        r = []
        for k, v in self.items():
            if isinstance(v, type(self)):
                r.append('{}{}:'.format(prefix, k))
                r.append(v.__repr__(indent=indent, prefix=prefix + indent * ' '))
            else:
                r.append('{}{}: {}'.format(prefix, k, str(v)))
        return '\n'.join(r)

    def shallowupdate(self, /, *a, **kw):
        self._update(1, *a, **kw)

    def update(self, /, *a, **kw):
        self._update(-1, *a, **kw)

    @classmethod
    def fromargs(cls, args, config=None, **kw):
        if args is not None:
            return cls.fromargs(None, config, **kw)
        if config is None:
            config = cls(_registering_default=False)
            return cls.fromargs(None, config, **kw)

        for k, v in kw.items:
            pass

    def _update_with_dotkeys(self, **kw):

        for k, v in kw.items():
            k_ = k.split('.')
            if len(k_) == 1:
                self.update(**{k: v})
                continue
            self[k_[0]]._update_with_dotkeys(**{'.'.join(k_[1:]): v})

    def _update(self, /, depth, *a, _registering_default=False, **kw):

        if a:
            assert len(a) == 1 and isinstance(a[0], (Namespace, dict)), '{},{}'.format(depth, a)

            if isinstance(a[0], dict):
                return self._update(depth, _registering_default=_registering_default, **a[0])
            # a[0] is args
            return self._update_with_dotkeys(__regiistering_defaults=_registering_default,
                                             **a[0].__dict__)
        for k, v in kw.items():

            if isinstance(v, dict):
                v['_registering_default'] = False
                if depth == 1 or k not in self:
                    super().update({k: type(self)(**v)})
                else:
                    self[k]._update(depth-1, **v)
                continue

            try:
                path = Path(v).resolve(strict=True)
                is_yml = path.suffix == '.yml'
            except (FileNotFoundError, TypeError):
                is_yml = False
            if is_yml:
                with open(path) as f:
                    self._update(depth-1, **{k: yaml.load(f, Loader=yaml.SafeLoader)})
                continue
            super().update({k: v})

    def subdict(self, k, prefix=None):

        if prefix is None:
            prefix = '{}_'.format(k)
        return type(self)(_registering_default=False,
                          **{'{}{}'.format(prefix, _): self[k][_] for _ in self[k]})

    def create_parser(self, parser=None, prefix=[], exclude=['config_keys'], include=None, aliases='aliases'):

        if isinstance(aliases, str):
            exclude.append(aliases)
            aliases = self[aliases]

        if include is None:
            include = set(self)

        include = include - set(exclude)

        if not parser:
            parser = argparse.ArgumentParser()

        for k, v in self.items():
            if k not in include:
                continue
            if isinstance(v, type(self)):
                v.create_parser(parser=parser, prefix=prefix + [k], aliases=aliases)
            else:
                arg_name = '.'.join(prefix + [k])
                arg_alias = []
                if arg_name in aliases:
                    arg_alias = aliases[arg_name]
                    if not isinstance(arg_alias, list):
                        arg_alias = [arg_alias]
                args = ['--{}'.format(arg_name), *arg_alias]
                logger.debug('{} ({})'.format(','.join(args), type(v)))
                if isinstance(v, bool):
                    parser.add_argument(*args, action='store_true', default=v)
                    arg_name_neg = 'no-' + arg_name
                    if arg_name_neg in aliases:
                        arg_alias = aliases[arg_name_neg]
                        if not isinstance(arg_alias, list):
                            arg_alias = [arg_alias]
                    args = ['--{}'.format(arg_name_neg), *arg_alias]
                    parser.add_argument(*args, action='store_false', dest=arg_name, default=v)
                else:
                    argtype = str if v is None else type(v)
                    parser.add_argument(*args, type=argtype, default=v)

        return parser

    def setup(self):

        pass
        # # time is in s from epoch, convert to ascii localtime
        # fmt = {'time': lambda x: time.asctime(time.localtime(x))}

        # index_types = self.pop('index_types', {})
        # formatters = {_: fmt[index_types[_]] for _ in index_types}

        # print(formatters)
        # pd.DataFrame.to_string = partialmethod(pd.DataFrame.to_string, formatters=formatters)


if __name__ == '__main__':
    c = ConfigDict()
    from .logger import set_loggers

    parser = c.create_parser(exclude=['config_keys'])

    args = parser.parse_args()

    c.update(args)
    set_loggers(**c)

    print(c)
