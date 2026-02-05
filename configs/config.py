import yaml

PARAMS_YML = 'configs/params.yml'
CSV_YML = 'configs/csv.yml'


class Config(dict):

    def update_with_yml(self, **kw):

        for k, path in kw.items():
            with open(path) as f:
                self.update({k: yaml.load(f, Loader=yaml.SafeLoader)})


default_parse_config = Config()
default_parse_config.update_with_yml(csv=CSV_YML, params=PARAMS_YML)
