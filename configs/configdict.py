import yaml
import pathlib

KEYS_YML = 'configs/config_keys.yml'
CSV_YML = 'configs/csv.yml'
FILES_YML = 'configs/files.yml'


class ConfigDict(dict):

    def __init__(self, *a, **kw):

        super().__init__()

        if '_registering_default' not in kw:
            self._update(0, files=FILES_YML, csv=CSV_YML, keys=KEYS_YML)

        self.deepupdate(*a, **kw)

    def update(self, *a, **kw):

        d = dict(*a, **kw)

        self._update(1, **d)

    def deepupdate(self, *a, **kw):

        d = dict(*a, **kw)
        self._update(-1, **d)

    def _update(self, depth, _registering_default=False, **kw):

        for k, v in kw.items():

            #            print(k, depth)
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


if __name__ == '__main__':
    print('**** default')
    c = ConfigDict(foo={'baz': 1, 'bar': 2})
    print(c['foo'], c['files'])

    print('**** deep update')
    c.deepupdate(foo={'baz': 2}, files={'ood_csv': 'foo.csv'})
    print(c['foo'], c['files'])

    print('**** update')
    c.update(files={'ood_csv': 'foo.csv'}, foo={'baz': 2})
    print(c['foo'], c['files'])
