import pytest
from wordle.solver import Game, Pattern, Feedback

# test data from https://github.com/yukosgiti/wordle-tests
PATTERN_TEST_DATA = 'tests/testdata/tests.txt'


def read_test_data(file_path: str):
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            answer, guess, feedback = line.split(',')
            yield answer, guess, feedback


@pytest.mark.parametrize('answer,guess,feedback', read_test_data(PATTERN_TEST_DATA))
def test_feedback_matching(answer, guess, feedback):
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
    pat = Pattern.from_str(feedback)
    
    fb1 = Game(answer).grade_guess(guess)
    assert fb1.pattern == pat, f'Failed for test1 ({answer}, {guess}, {feedback})'

    fb2 = Feedback(guess, pat)
    assert fb2.matchesWord(answer), f'Failed for test2 ({answer}, {guess}, {feedback})'


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

def test_matches_word_small():
    """
    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    fb = Game('TABOO').grade_guess('POOCH')
    assert fb.matchesWord('TABOO')
    assert not fb.matchesWord('OTHER')

    fb = Game('OTHER').grade_guess('POOCH')
    assert fb.matchesWord('OTHER')
    assert not fb.matchesWord('TABOO')

