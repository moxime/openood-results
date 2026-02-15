import logging
import yaml
import pathlib
import argparse
from argparse import Namespace

KEYS_YML = 'configs/config_keys.yml'
CSV_YML = 'configs/table.yml'
MAIN_YML = 'configs/main.yml'


logger = logging.getLogger(__name__)


class ConfigDict(dict):

    def __init__(self, *a, **kw):

        super().__init__()

        if '_registering_default' not in kw:
            for yml_file in (MAIN_YML, CSV_YML, KEYS_YML):
                self._update(0, **yaml.load(open(yml_file), Loader=yaml.SafeLoader))

        self.deepupdate(*a, **kw)

    def __repr__(self, prefix='', indent=2):

        r = []
        for k, v in self.items():
            if isinstance(v, type(self)):
                r.append('{}{}:'.format(prefix, k))
                r.append(v.__repr__(indent=indent, prefix=prefix + indent * ' '))
            else:
                r.append('{}{}: {}'.format(prefix, k, str(v)))
        return '\n'.join(r)

    def update(self, *a, **kw):
        self._update(1, *a, **kw)

    def deepupdate(self, *a, **kw):
        self._update(-1, *a, **kw)

    def _update(self, depth, *a, _registering_default=False, **kw):

        if a:
            assert len(a) == 1 and isinstance(a[0], (Namespace, dict)), '{},{}'.format(depth, a)

            if isinstance(a[0], dict):
                return self._update(depth, _registering_default=_registering_default, **a[0])
            return self._update(depth, __regiistering_defaults=_registering_default, **a[0].__dict__)
        for k, v in kw.items():

            if isinstance(v, dict):
                v['_registering_default'] = False
                if depth == 1 or k not in self:
                    super().update({k: type(self)(**v)})
                else:
                    self[k]._update(depth-1, **v)
                continue

            try:
                path = pathlib.Path(v).resolve(strict=True)
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

    def create_parser(self, parser=None, prefix=[], exclude=None, include=None):

        if include is None:
            include = set(self)

        if exclude is not None:
            include = include - set(exclude)

        if not parser:
            parser = argparse.ArgumentParser()

        for k, v in self.items():
            if k not in include:
                continue
            if isinstance(v, type(self)):
                v.create_parser(parser=parser, prefix=prefix + [k])
            else:
                arg = '--{}'.format('.'.join(prefix + [k]))
                parser.add_argument(arg, type=type(v), default=v)

        return parser


if __name__ == '__main__':
    c = ConfigDict()

    print(c)
    parser = c.create_parser(exclude=['config_keys'])

    args, _ = parser.parse_known_args()

    c.deepupdate(args)

    print(c)
