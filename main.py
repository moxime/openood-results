from .utils import ConfigDict, set_loggers, df_results, create_filter_parser
import argparse
import sys

print('***', __name__)
if __name__ == "__main__" and __package__ is None:
    import os
    import sys

    try:
        pkg_dir = os.path.dirname(os.path.abspath(__file__))  # .../myproject
        print(__file__)
        parent = os.path.dirname(pkg_dir)                     # .../
        sys.path.insert(0, parent)
        __package__ = 'openood-results'
    except NameError:
        print(__package__)

print('***', __package__)


def main():
    argv = '--results_dir ./results/lab-ia filter --epoch 200 --set cifar100'

    argv = None if sys.argv[0] else argv.split()

    config = ConfigDict()

    parser = argparse.ArgumentParser()
    parser = config.create_parser()

    subparsers = parser.add_subparsers()

    parser_filter = subparsers.add_parser('filter', help='table filter help')

    args, _ = parser.parse_known_args(argv)

    config.update(args)
    set_loggers(**config)

    df, dropped_index = df_results(**config)

    create_filter_parser(df, parser=parser_filter, **config)

    parser.parse_args(argv, namespace=args)

    # print('FARGS\n', args)

    for k in df.index.names:
        df = df.iloc[df.index.isin(vars(args)['filter.{}'.format(k)], level=k)]

    for n in df.index.names:
        values = set(df.index.get_level_values(n))
        if len(values) == 1:
            df.index = df.index.droplevel(n)
            dropped_index[n] = next(iter(values))

    df.sort_index(inplace=True)
    print(df.to_string())
    print('='*20)
    # print(df.index.names)
    print(ConfigDict(dropped_index, _registering_default=False))


if __name__ == '__main__':

    main()
