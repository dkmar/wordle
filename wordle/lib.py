from itertools import product


class Pattern:
    """
    The Pattern class implements functions and literals related to guess feedback representation.
    Wrong, Miss, Correct <=> Black, Yellow, Green
    """
    SQUARES = '⬛🟨🟩'
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
        if len(s) != 5:
            raise ValueError

        res = s.translate(Pattern._trans_table_str_to_pattern)
        if any(ch not in Pattern.SQUARES for ch in res):
            raise ValueError

        return res


    @staticmethod
    def to_str(pat: str):
        return pat.translate(Pattern._trans_table_pattern_to_str)
