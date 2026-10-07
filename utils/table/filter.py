import logging
from abc import ABC, abstractmethod, abstractclassmethod
import re
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


interval_regexp = re.compile('([^\.]*)\.{2,}([^\.]*)')


class WrongTypeError(ValueError):
    pass


class Region(object):

    def __init__(self, *a, strict=False, negation=False, rtype='discrete'):

        assert rtype in ('nan', 'discrete', 'interval')

        self.rtype = rtype
        self.bounds = a
        self.strict = strict

    @classmethod
    def from_string(cls, s):

        s = s.strip()

        if not s or s.lower() in ('nan', 'none'):
            return cls(rtype='nan')

        try:
            a = map(float, s.split(' '))
            return cls(*a, rtype='discrete')
        except ValueError:
            pass

        try:
            m = interval_regexp.match(s)
            if not m:
                raise ValueError

            # m.groups() is the interval bounds. if empty, +/- inf
            a = (float(_) if _ else np.inf * (i-0.5) for i, _ in enumerate(m.groups()))
            return cls(*a, rtype='interval')
        except ValueError:
            pass

        raise ValueError('Did not manage to parse {} as a region'.format(s))

    def is_in(self, val):

        if self.rtype == 'nan':
            return val in (None, np.nan)

        if self.rtype == 'discrete':
            return val in self.bounds

        if self.rtype == 'interval':
            if self.strict and val in self.bounds:
                return False
            # print(val, '>=', self.bounds[0], ':', val >= self.bounds[0],
            #       '<=', self.bounds[1], ':', val <= self.bounds[1])
            return (val >= self.bounds[0]) and (val <= self.bounds[1])

    def __call__(self, val):

        return self.is_in(val)

    def __eq__(self, val):

        if not isinstance(val, type(self)):
            return False

        for attr in ('rtype', 'bounds', 'strict'):
            if getattr(self, attr) != getattr(val, attr):
                return False

        return True

    def __hash__(self):

        return hash(tuple((self.bounds, self.rtype)))

    def __repr__(self):

        if self.rtype == 'nan':
            return 'nan'

        join = '...' if self.rtype == 'interval' else ', '
        bounds = '()' if self.rtype == 'interval' and self.strict else '[]'

        if len(self.bounds) == 1:
            bounds = ['', '']

        return bounds[0] + join.join(map(str, self.bounds)) + bounds[-1]

    def copy(self):

        return type(self)(*self.bounds, rtype=self.rtype)


class Filter(ABC):

    def __init__(self, *a, ftype=str, negation=False, wrongtype=None, **kw):

        logger.debug('ConfigDict init of type {} with ftype {}'.format(type(self), ftype))
        self._type = ftype
        self._negation = negation
        self._wrongtype = wrongtype

    def __call__(self, val):

        if isinstance(val, pd.Index):

            return val.map(lambda x: self(x))

        if not self._type:
            return self.match(val) ^ self._negation

        if NoneFilter()(val):
            return self._negation

        try:
            return self.match(self._type(val)) ^ self._negation
        except ValueError:
            logger.debug('{} is not of type {} hence returning {}'.format(val,
                                                                          str(self._type),
                                                                          self._negation))
            return self._negation

    @abstractmethod
    def match(self, val):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractclassmethod
    def _from_string(cls, s, **kw):
        raise NotImplementedError

    @property
    def is_true(self):
        return False

    @classmethod
    def from_string(cls, s, **kw):

        has_neg, string = re.match('(^\s*not\s+|\s*!\s*)?\s*(.*)', s).groups()

        filter = cls._from_string(string, **kw)
        filter._negation = filter._negation ^ bool(has_neg)

        return filter

    def neg(self):

        filter = self.copy()
        filter._negation = not self._negation

        return filter

    def __repr__(self):

        return '!' if self._negation else ''

    def __or__(self, val):

        if isinstance(val, str):
            return self | IndexFilter(val)
        return OrFilter(self, val)

    def __and__(self, val):

        if isinstance(val, str):
            return self & IndexFilter(val)
        return AndFilter(self, val)

    def __neg__(self):
        return self.neg()

    def __sub__(self, val):

        if isinstance(val, str):
            return self & (-IndexFilter(val))
        return self & (-val)

    def __add__(self, val):

        return self & val

    def __hash__(self):
        return 0

    def __eq__(self, val):

        if not isinstance(val, type(self)):
            return False

        if not self._negation == val._negation:
            return False

        return self._is_same(val)


class TrueFilter(Filter):

    @property
    def is_true(self):
        return True

    def match(self, val):
        return True

    @classmethod
    def _from_string(cls):
        return cls()

    def copy(self):
        f = type(self)()
        f._negation = self._negation
        return f

    def _is_same(self, other):

        return True

    def __repr__(self):

        return 'False' if self._negation else 'True'


class NoneFilter(Filter):

    def __init__(self, **kw):

        super().__init__(ftype=None, **kw)

    def match(self, val):
        if val is pd.NA:
            return True
        return val in (None, np.nan)

    def copy(self):

        return type(self)(negation=self._negation)

    @classmethod
    def _from_string(cls, s):
        return cls()

    def _is_same(self, other):
        return True

    def __repr__(self):

        return 'not null' if self._negation else 'null'


class AndFilter(Filter):

    def __new__(cls, *filters, identity=TrueFilter(), **kw):

        filters = cls._clean_filters(identity, *filters)

        if len(set(filters)) == 1:
            return filters.pop()

        return super().__new__(cls)

    def __init__(self, *filters, identity=TrueFilter(), **kw):

        assert not kw.get('negation')
        super().__init__(self, ftype=None)
        filters = self._clean_filters(identity, *filters)
        if len(filters) == 1:
            return
        self.filters = filters
        self._op_str = '&'
        self._op = all
        self._no_filters_returns = False

    @classmethod
    def _clean_filters(cls, identity, *filters):

        filters_ = [list(f.filters) if type(f) is cls and not f._negation else [f]
                    for f in filters]
        filters = sum(filters_, [])

        if identity.neg() in filters:
            return set([identity.neg()])

        while identity in filters:
            filters.remove(identity)

        if not filters:
            return set([identity])
        return set(filters)

    def match(self, val):
        if not self.filters:
            return self._no_filters_returns

        return self._op(f(val) for f in self.filters)

    def copy(self):
        return type(self)(*(f.copy() for f in self.filters), negation=self._negation)

    def neg(self):

        return OrFilter(*(f.neg() for f in self.filters))

    def __repr__(self):

        def sub(f): return '({})'.format(f) if isinstance(f, AndFilter) else '{}'.format(f)
        s = '{}'.format(self._op_str).join(sub(f) for f in self.filters)

        if self._negation:
            return '!({})'.format(s)

        return s

    @classmethod
    def _from_string(cls, s):
        raise NotImplementedError

    def _is_same(self, other):

        return self.filters == other.filters


class OrFilter(AndFilter):

    def __new__(cls, *filters, **kw):

        return super().__new__(cls, *filters, identity=TrueFilter().neg(), **kw)

    def __init__(self, *filters, **kw):

        super().__init__(*filters, identity=TrueFilter().neg(), **kw)

        self._op_str = '|'
        self._op = any
        self._no_filters_returns = True

    def neg(self):

        return AndFilter(*(f.neg() for f in self.filters))


class StringFilter(Filter):

    def __init__(self, regex, **kw):

        logger.debug('Creating a filter of type {} with regex {}'.format(type(self), regex))
        super().__init__(ftype=str, **kw)
        self.regex = regex

    def match(self, val):
        return bool(re.fullmatch(self.regex, val))

    @classmethod
    def _from_string(cls, s, **kw):

        logger.debug('Parsing {} to make it a regex'.format(s))
        regex = '|'.join(s.replace('*', '.*').split())
        return cls(regex, **kw)

    def copy(self):
        return type(self)(self.regex, negation=self._negation)

    def __repr__(self):

        return super().__repr__() + self.regex

    def _is_same(self, other):

        return self.regex == other.regex


class FloatFilter(Filter):

    def __init__(self, region, **kw):

        logger.debug('Creating a filter of type {} with region {}'.format(type(self), region))

        super().__init__(ftype=float, **kw)

        assert isinstance(region, Region)
        self.region = region

    def match(self, val):
        return self.region(val)

    @classmethod
    def _from_string(cls, s, **kw):
        region = Region.from_string(s)
        return cls(region, **kw)

    def __repr__(self):
        return super().__repr__() + repr(self.region)

    def copy(self):
        return type(self)(region=self.region.copy(), negation=self._negation)

    def _is_same(self, other):

        return self.region == other.region


def _filter_from_string(s, **kw):

    s = re.sub('\s*&\s*', '&', s)
    s = re.sub('\s*\|\s*', '|', s)

    if len(s.split()) > 1:
        return OrFilter(*(_filter_from_string(_s) for _s in s.split()))

    # assert '&' not in s or '|' not in s

    if '|' in s:
        return OrFilter(*(_filter_from_string(_s) for _s in s.split('|')))

    if '&' in s:
        return AndFilter(*(_filter_from_string(_s) for _s in s.split('&')))

    s = s.strip()

    if s in ('', '!'):
        return NoneFilter.from_string(s)

    try:
        return FloatFilter.from_string(s)
    except ValueError:
        return StringFilter.from_string(s)


class IndexFilter(Filter):

    def __new__(cls, s):

        f = _filter_from_string(s)

        if isinstance(f, AndFilter) and len(f.filters) == 1:

            return -f.filters.pop() if f._negation else f.filters.pop()

        return f


if __name__ == '__main__':

    import sys
    import utils.logger
    import argparse
    from functools import reduce
    from operator import or_, and_

    def test_float():
        for s in ('..', '1 3 4', 'not 1 2', '1..3', 'not ..3', '1', 'not 1'):
            f = FloatFilter.from_string(s)
            print('{:20}'.format(str(f)), end=': ')
            for x in (2, np.inf, 1, 0):
                print('{}: {}'.format(x, f(x)), end=', ')
            print()

    def test_string():

        for s in ('foo', 'foo*', 'not foo*', 'foo bar', '*bar'):
            f = StringFilter.from_string(s)
            print(f)
            for x in ('foo foobar foobaz bar bazbar'.split()):
                print('{}: {}'.format(x, f(x)), end=', ')
            print()

    # test_float()
    # test_string()

    parser = argparse.ArgumentParser()

    parser.add_argument('-f', action='append', default=[], nargs='*')
    parser.add_argument('-f-', action='append', dest='fn', default=[], nargs='*')
    parser.add_argument('s', nargs='*', default=[None])

    argv = '-f a|b -f 1..3'
    argv = '-f a|b -f 1..3 -f- foo*'
    argv = '-f bar baz foo |1..4'
    argv = '-f a|b c'
    argv = '-f a&d b c'
    argv = '-f *bar -f- foo* -f a&b c'
    args = parser.parse_args(argv.split() if not sys.argv[0] else None)

    fpos = TrueFilter()
    for f in args.f:
        fpos |= f

    fneg = TrueFilter()
    for f in args.f:
        fpos |= f

    fpos = reduce(and_,
                  (reduce(or_, (IndexFilter(_) for _ in f), -TrueFilter()) if f else NoneFilter()
                   for f in args.f),
                  TrueFilter())

    fneg = reduce(and_,
                  (reduce(or_, (IndexFilter(_).neg() for _ in f), -TrueFilter()) if f else -NoneFilter()
                   for f in args.fn),
                  TrueFilter())

    print(fpos + fneg)

    for s in args.s:
        try:
            s = float(s)
        except ValueError:
            pass
        print('{}: {}'.format(s, (fpos + fneg)(s)))
