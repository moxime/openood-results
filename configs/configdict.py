import logging
import yaml
import pathlib
import argparse

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

    def __repr__(self):

        return 'Conf{}'.format(super().__repr__())

    def update(self, *a, **kw):

        d = dict(*a, **kw)

        self._update(1, **d)

    def deepupdate(self, *a, **kw):

        d = dict(*a, **kw)
        self._update(-1, **d)

    def _update(self, depth, _registering_default=False, **kw):

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

    def create_parser(self, parser=None, prefix=[]):

        if not parser:
            parser = argparse.ArgumentParser()
        for k, v in self.items():

            if isinstance(v, type(self)):
                yield from self[k].create_parser(prefix=prefix + [k])

            yield prefix + [k]


if __name__ == '__main__':
    c = ConfigDict()

    for _ in c.create_parser():
        print(_)
