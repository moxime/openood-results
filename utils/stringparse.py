class Token(str):

    OPERATORS = '&|'
    NEG = '!'

    SPECIAL = OPERATORS + NEG

    def __new__(cls, s, level=0):
        return str.__new__(cls, s)

    def __init__(self, s, level=0):
        self.level = level
        self._isspecial = self in self.SPECIAL

    def __repr__(self):
        return super().__repr__() + ('({})'.format(self.level) if self.level > -1 else '')

    def __add__(self, other):

        if other.level != self.level:
            raise ValueError
        return Token(super().__add__(other), level=self.level)

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, val):
        self._level = val

    @property
    def isspecial(self):
        return self._isspecial


class Tokens(list):

    def __init__(self, string):

        super().__init__()
        self.current_level = 0

        for c in string:
            self.append(c)

        assert self.current_level == 0

    def append(self, letter):

        if letter == '(':
            self.current_level += 1
            return
        if letter == ')':
            self.current_level -= 1
            return

        token = Token(letter, level=self.current_level)

        if not self:
            super().append(token)
            return

        if token.isspecial or self[-1].isspecial:
            super().append(token)
            return

        if token.level != self[-1].level:
            super().append(token)
            return

        self[-1] += token


class Tree:

    def __init__(self, op, *subtrees):

        self.op = op
        self.subtrees = subtrees

    @classmethod
    def from_tokens(tokens):

        min_level = min(tokens, key=lambda x: x.level)
        i_ = [i for i, _ in enumerate(tokens) if i == min_level]

    def __repr__(self, prefix=''):

        firstline = '{}{}\n'.format(prefix, self.op)
        subtrees = '\n'.join([repr(subtree, prefix=prefix+'   ') for subtree in self.subtrees])

        return firstline + subtrees
