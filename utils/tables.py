from pathlib import Path
from utils.load import OOD_CSV, CONFIG_YML, read_csv, load_config, CSV_YML
from utils.config import sample_config, PARAMS_YML


def df_exp(path, csv_name=OOD_CSV, yml_name=CONFIG_YML, key_dict={}, index_dict={}, index_order=[]):
    path = Path(path)

    if not path.exists() or not path.is_dir():
        raise FileNotFoundError(path)

    df = read_csv(path, name=csv_name)

    try:
        config = load_config(path, name=yml_name)
    except FileNotFoundError:
        config = {}

    params = dict(sample_config(config, key_dict))

    for k, v in params.items():
        df[k] = v

    df.set_index(list(params), append=True, inplace=True)

    return df


if __name__ == '__main__':

    import yaml
    from utils.load import test_load_from_term

    with open(PARAMS_YML) as f:
        key_dict = yaml.load(f, Loader=yaml.SafeLoader)

    path = test_load_from_term('results/ortho')[0].f

    df = df_exp(path, key_dict=key_dict)

    print(df.head().to_string())
