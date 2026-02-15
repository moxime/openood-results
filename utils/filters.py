import logging
from abc import ABC, abstractmethod, abstractclassmethod
import numpy as np
import re

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
            return (val >= self.bounds[0]) and (val <= self.bounds[1])

    def __call__(self, val):

        return self.is_in(val)

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

        if self._type and not isinstance(val, self._type):
            logger.debug('{} is not of type {} hence returning {}'.format(val,
                                                                          str(self._type),
                                                                          self._negation))
            return self._negation

        return self.match(val) ^ self._negation

    @abstractmethod
    def match(self, val):
        raise NotImplementedError

    @abstractmethod
    def copy(self):
        raise NotImplementedError

    @abstractclassmethod
    def _from_string(cls, s, **kw):
        raise NotImplementedError

    @classmethod
    def from_string(cls, s, **kw):

        has_neg, string = re.match('(^\s*not\s+|!\s*)?\s*(.*)', s).groups()

        filter = cls._from_string(string, **kw)
        filter._negation = filter._negation ^ bool(has_neg)

        return filter

    def neg(self):

        filter = self.copy()
        filter._negation = not self._negation

        return filter

    def __repr__(self):

        return '!' if self._negation else ''


class AndFilter(Filter):

    def __init__(self, *filters, **kw):

        super().__init__(self, ftype=None)
        self.filters = filters
        self._op = '&'

    def match(self, val):
        if not self.filters:
            return False

        return all(f(val) for f in self.filters)

    def copy(self):
        return type(self)(*(f.copy() for f in self.filters), negation=self._negation)

    def __repr__(self):

        s = self._op.join(repr(f) for f in self.filters)

        if self._negation:
            return '!({})'.format(s)

        return s

    @classmethod
    def _from_string(cls, s):
        raise NotImplementedError


class OrFilter(AndFilter):

    def __init__(self, *filters, **kw):

        super().__init__(*filters, **kw)

        self.filters = filters
        self._op = '|'

    def match(self, val):
        if not self.filters:
            return True

        return any(f(val) for f in self.filters)

    def copy(self):
        return type(self)(*(f.copy() for f in self.filters), negation=self._negation)

    def __repr__(self):

        s = '|'.join(repr(f.neg()) for f in self.filters)

        if not self._negation:
            return '!({})'.format(s)

        return s


class NoneFilter(Filter):

    def __init__(self, **kw):

        super().__init__(ftype=None, **kw)

    def match(self, val):
        return val in (None, np.nan)

    def copy(self):

        return type(self(negation=self._negation))

    @classmethod
    def _from_string(cls, s):
        return cls()

    def __repr__(self):

        return super().__repr__() + 'none'


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


class ConfigFilter(Filter):

    def __init__(self, filter, **kw):

        super().__init__(ftype=filter._type, **kw)
        assert isinstance(filter, Filter), type(filter)

        self._filter = filter
        self._type = filter._type

        assert not self._type or self._type == self._filter._type

    def __repr__(self):
        return super().__repr__() + repr(self._filter)

    def copy(self):
        return type(self)(filter=self._filter, negation=self._negation)

    def match(self, val):
        return self._filter.match(val)

    @classmethod
    def _from_string(cls, s, **kw):

        assert '&' not in s or '|' not in s

        if '&' in s:
            return AndFilter(*(cls._from_string(_s) for _s in s.split('&')))

        if '|' in s:
            return OrFilter(*(cls._from_string(_s) for _s in s.split('|')))

        if not s.strip():
            return cls(NoneFilter())

        try:
            return cls(FloatFilter._from_string(s))
        except ValueError:
            pass
        return cls(StringFilter._from_string(s))


if __name__ == '__main__':

    import utils.logger

    def test_float():
        for s in ('..', '1 3 4', 'not 1 2', '1..3', 'not ..3', '1', 'not 1'):
            f = FloatFilter.from_string(s)
            print('{:10}'.format(str(f)), end=': ')
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

    sfilters = [ConfigFilter.from_string(_) for _ in ('foo*', '*bar', '1..3')]

    andfilter = AndFilter(*sfilters)
    orfilter = OrFilter(*sfilters)
