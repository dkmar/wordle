from collections import UserList
from enum import IntEnum
from itertools import product


class Pattern:
    ALL_PATTERNS = tuple(map(''.join, product('⬛🟨🟩', repeat=5)))

    _trans_table_str_to_pattern = str.maketrans(
        dict.fromkeys('_bBwW', '⬛') |
        dict.fromkeys('yYmM', '🟨') |
        dict.fromkeys('gGcC', '🟩')
    )

    _trans_table_pattern_to_str = str.maketrans('⬛🟨🟩', 'BYG')

    @staticmethod
    def from_str(s: str):
        """
        [_BW] ⬛
        [YM]  🟨
        [GC]  🟩
        """
        return s.translate(Pattern._trans_table_str_to_pattern)

    @staticmethod
    def to_str(pat: str):
        return pat.translate(Pattern._trans_table_pattern_to_str)


class Status(IntEnum):
    Grey = 0
    Yellow = 1
    Green = 2

    def __str__(self):
        match self:
            case Status.Grey:
                return '⬛'
            case Status.Yellow:
                return '🟨'
            case Status.Green:
                return '🟩'

    @classmethod
    def from_char(cls, ch: str):
        match ch:
            case 'B' | '_' | 'w':
                return Status.Grey
            case 'Y' | 'm':
                return Status.Yellow
            case 'G' | 'c':
                return Status.Green


# class Pattern(UserList):
#     def __init__(self, initial_data=None):
#         super().__init__()
#         if initial_data:
#             self.extend(initial_data)
#         else:
#             self.data = [None] * 5
#
#     def __str__(self):
#         return ''.join(map(str, self.data))
#
#     def to_int(self) -> int:
#         code = 0
#         for status in self.data:
#             code = code * 3 + status.value
#         return code
#
#     @staticmethod
#     def all_patterns():
#         return range(3 ** 5)
#
#     @classmethod
#     def from_int(cls, code: int):
#         pattern = cls()
#         for i in reversed(range(5)):
#             pattern[i] = Status(code % 3)
#             code //= 3
#         return pattern
#
#     @classmethod
#     def from_str(cls, s: str):
#         pattern = cls(Status.from_char(ch) for ch in s)
#         return pattern
