from pathlib import Path


def concatenate(*dfs):
    all_indexes = set(sum([list(df.index.names) for df in dfs], start=[]))

    print(*all_indexes)


if __name__ == '__main__':
    from configs.configdict import ConfigDict
    from utils.load import fetch_results

    concatenate(*fetch_results(**ConfigDict()))
    concatenate(*fetch_results())
