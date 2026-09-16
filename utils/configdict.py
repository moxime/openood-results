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
    default_config_root = Path(__file__).parent.parent / 'configs'
except NameError:
    default_config_root = Path('configs')


logger = logging.getLogger(__name__)


class NoneArgType:

    def __repr__(self):
        return 'NoneArg'


NoneArg = NoneArgType()


class ConfigDict(dict):

    def __init__(self, /, *a, config_root=None, **kw):

        super().__init__()

        if config_root == 'default':
            config_root = default_config_root
        try:
            config_files = Path(config_root).glob('*.yml')
        except TypeError:
            config_files = []

        self.update(*config_files, *a, **kw)

    def __repr__(self, prefix='', indent=2):

        r = []
        for k, v in self.items():
            if isinstance(v, type(self)):
                r.append('{}{}:'.format(prefix, k))
                r.append(v.__repr__(indent=indent, prefix=prefix + indent * ' '))
            else:
                r.append('{}{}: {}'.format(prefix, k, str(v)))
        return '\n'.join(r)

    def copy(self):

        config = type(self)()
        config.update(self)
        return config

    def shallowupdate(self, /, *a, **kw):
        self._update(0, *a, **kw)

    def update(self, /, *a, **kw):
        self._update(-1, *a, **kw)

    def __getattr__(self, a):
        if a in self:
            return self.__getitem__(a)
        return super().__getattribute__(a)

    def __setattr__(self, k, val):
        if hasattr(super(), k):
            return super().__setattr__(k, val)
        return self.__setitem__(k, val)

    def _update_with_dotkeys(self, /, **kw):

        for k, v in kw.items():
            k_ = k.split('.')
            if len(k_) == 1:
                self.update(**{k: v})
                continue
            self[k_[0]]._update_with_dotkeys(**{'.'.join(k_[1:]): v})

    def _update(self, /, depth, *a, **kw):

        for arg in a:
            if isinstance(arg, dict):
                self._update(depth, **arg)
                continue
            if isinstance(arg, Namespace):
                self._update_with_dotkeys(**arg.__dict__)
                continue
            if isinstance(arg, (Path, str)) and Path(arg).suffix == '.yml':
                self._update(depth, **yaml.load(open(arg), Loader=yaml.SafeLoader))
                continue
            raise TypeError('Can not update config with {}'.format(type(arg)))

        for k, v in kw.items():

            if isinstance(v, dict):
                if not depth or k not in self:
                    super().update({k: type(self)(**v)})
                else:
                    self[k]._update(depth-1, **v)
                continue

            if v is not NoneArg:
                super().update({k: v})

    def _create_parser(self, parser=None, prefix=[], exclude=None, aliases=None):

        def generic_type(v):

            for t in (int, float, str):
                try:
                    return t(v)
                except ValueError:
                    pass

        if aliases is None:
            aliases = self.get('args', {}).get('aliases', {})

        if exclude is None:
            exclude = self.get('args', {}).get('exclude', {})

        if not parser:
            parser = argparse.ArgumentParser()

        for k, v in self.items():
            arg_name = '.'.join(prefix + [k])
            arg_name_neg = '.'.join(prefix + ['no_' + k])
            if arg_name in exclude:
                continue

            if isinstance(v, type(self)):
                v._create_parser(parser=parser, prefix=prefix + [k], exclude=exclude, aliases=aliases)
                continue

            arg_alias = []
            if arg_name in aliases:
                arg_alias = aliases[arg_name]
                if not isinstance(arg_alias, list):
                    arg_alias = [arg_alias]
            args = ['--{}'.format(arg_name), *arg_alias]
            logger.debug('{} ({})'.format(','.join(args), type(v)))

            if isinstance(v, bool):
                parser.add_argument(*args, action='store_true', default=NoneArg)
                if arg_name_neg in aliases:
                    arg_alias = aliases[arg_name_neg]
                    if not isinstance(arg_alias, list):
                        arg_alias = [arg_alias]
                args = ['--{}'.format(arg_name_neg), *arg_alias]
                parser.add_argument(*args, action='store_false', dest=arg_name, default=NoneArg)
                continue

            if isinstance(v, list):
                argtype = type(v[0]) if v else generic_type
                nargs = '*'
                extend_arg = '--{}+'.format(arg_name)
            else:
                argtype = generic_type if v is None else type(v)
                nargs = None
                extend_arg = None
            parser.add_argument(*args, type=argtype, nargs=nargs, default=v, metavar=k.upper())
            if extend_arg:
                parser.add_argument(extend_arg, dest=arg_name, type=argtype, nargs='*', action='extend')

        return parser

    def parse_args(self, argv=None, **kw):

        parser = self._create_parser(**kw)
        args, unknown_args = parser.parse_known_args(argv)
        self.update(args)

        return unknown_args


if __name__ == '__main__':

    c = ConfigDict(config_root='configs')
    from .logger import set_loggers

    parser = c._create_parser()

    args = parser.parse_args()

    c.update(args)
    # set_loggers(**c.logger)

    print(c)
