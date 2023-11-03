from wordle.lib import Pattern, Status
from collections.abc import Mapping
import numpy as np
import itertools

with open('wordle/data/relevant_words.txt', 'r') as f:
    words = map(str.strip, f)
    RELEVANT_WORDS = tuple(map(str.upper, words))

GUESSES = RELEVANT_WORDS
ANSWERS = RELEVANT_WORDS
guess_index = {guess: np.uint16(i)
               for i, guess in enumerate(GUESSES)}

# ALL_PATTERNS = tuple(map(Pattern.from_int, Pattern.all_patterns()))
ALL_PATTERNS = tuple(map(''.join, itertools.product('⬛🟨🟩', repeat=5)))

pattern_index = {pattern: np.uint8(i)
                 for i, pattern in enumerate(ALL_PATTERNS)}


def grade_guess(guess: str, answer: str) -> str:
    """
    What should feedback look like?
    return indices of greens and yellows.

    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    feedback = ['⬛'] * 5
    used = 0
    # label greens
    for i, (ch, ans) in enumerate(zip(guess, answer)):
        if ch == ans:
            feedback[i] = '🟩'
            used |= (1 << i)

    # label yellows
    for i, (ch, fb) in enumerate(zip(guess, feedback)):
        if fb == '🟩':
            continue

        # TODO: see if it even makes sense to usr find() instead of a generator like we do in matchesWord
        j = -1
        while (j := answer.find(ch, j + 1)) != -1:
            if not used & (1 << j):
                feedback[i] = '🟨'
                used |= (1 << j)
                break
        else:
            feedback[i] = '⬛'

    return ''.join(feedback)

# def grade_guess(guess: str, answer: str) -> str:
#     """
#     What should feedback look like?
#     return indices of greens and yellows.
#
#     POOCH TABOO
#     _YY__
#
#     POOCH OTHER
#     _Y__Y
#     """
#     feedback = Pattern()
#     used = 0
#     # label greens
#     for i, (ch, ans) in enumerate(zip(guess, answer)):
#         if ch == ans:
#             feedback[i] = Status.Green
#             used |= (1 << i)
#
#     # label yellows
#     for i, (ch, fb) in enumerate(zip(guess, feedback)):
#         if fb == Status.Green:
#             continue
#
#         # TODO: see if it even makes sense to usr find() instead of a generator like we do in matchesWord
#         j = -1
#         while (j := answer.find(ch, j + 1)) != -1:
#             if not used & (1 << j):
#                 feedback[i] = Status.Yellow
#                 used |= (1 << j)
#                 break
#         else:
#             feedback[i] = Status.Grey
#
#     return str(feedback)

def guess_feedback(guess: str, answers: tuple[str] = ANSWERS, pattern_id: Mapping[str, np.uint8] = pattern_index):
    return [pattern_id[grade_guess(guess, answer)] for answer in answers]

FeedbackType = np.dtype((np.uint8, len(ANSWERS)))
feedback = np.fromiter(map(guess_feedback, GUESSES), dtype=FeedbackType, count=len(GUESSES))

def score(guess: str) -> np.float64:
    guess_id = guess_index[guess]
    fb = feedback[guess_id]

    _, freqs = np.unique(fb, return_counts=True)
    probabilities = freqs / fb.shape[0]
    information = -np.log2(probabilities)
    expected_info = probabilities * information

    return expected_info.sum()

evals = []
for guess in GUESSES[:100]:
    s = score(guess)
    evals.append((s, guess))

evals.sort(reverse=True)
for s, g in evals:
    print(g, s)