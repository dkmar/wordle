import pytest
from wordle.solver import Game, Pattern

# test data from https://github.com/yukosgiti/wordle-tests
PATTERN_TEST_DATA = 'tests/testdata/tests.txt'


def read_test_data(file_path: str):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            answer, guess, feedback = line.split(',')
            yield answer, guess, feedback


@pytest.mark.parametrize('answer,guess,feedback', read_test_data(PATTERN_TEST_DATA))
def test_grade_guess(answer, guess, feedback):
    """
    tests.txt
        aaaaa,aaaaa,ccccc
        aaaaa,aaaab,ccccw
        aaaaa,aaaba,cccwc
        aaaaa,aaabb,cccww

    answer,guess,feedback
    where the feedback pattern is coded as:
        c == correct == green
        w == wrong == black/grey
        m == misplaced == yellow
    """
    fb = Game(answer).grade_guess(guess)
    assert fb.pattern == Pattern.from_str(feedback), f'Failed for test ({answer}, {guess}, {feedback})'


def test_patterns_manual():
    """
    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    fb = Game('TABOO').grade_guess('POOCH')
    assert fb.pattern == Pattern.from_str('_YY__')
    fb = Game('OTHER').grade_guess('POOCH')
    assert fb.pattern == Pattern.from_str('_Y__Y')
