import numpy.typing

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


# from collections import Counter, defaultdict
# answer_counts = {answer: Counter(answer) for answer in ANSWERS}
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
#     feedback = ['⬛'] * 5
#     counts = answer_counts[answer]
#
#     # label greens
#     for i, (ch, ans) in enumerate(zip(guess, answer)):
#         if ch == ans:
#             feedback[i] = '🟩'
#             counts[ans] -= 1
#
#     # label yellows
#     for i, (ch, ans) in enumerate(zip(guess, answer)):
#         if ch != ans and counts.get(ch, 0) > 0:
#             feedback[i] = '🟨'
#             counts[ch] -= 1
#
#     for (ch, fb) in zip(guess, feedback):
#         if fb != '⬛':
#             counts[ch] += 1
#
#     return ''.join(feedback)

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
        if fb == '⬛':
            j = answer.find(ch)
            while j != -1:
                if not used & (1 << j):
                    feedback[i] = '🟨'
                    used |= (1 << j)
                    break

                j = answer.find(ch, j + 1)

    return ''.join(feedback)


def feedbacks_for_guess(guess: str, answers: tuple[str] = ANSWERS, pattern_id: Mapping[str, np.uint8] = pattern_index):
    return [pattern_id[grade_guess(guess, answer)] for answer in answers]


# feedback: feedback for each answer if we used this guess
# guess_id -> [feedback pattern for answer in ANSWERS]
FeedbackType = np.dtype((np.uint8, len(ANSWERS)))
guess_feedbacks = np.fromiter(map(feedbacks_for_guess, GUESSES), dtype=FeedbackType, count=len(GUESSES))


def score(guess_id: np.uint16, feedbacks: np.ndarray[FeedbackType]) -> np.float64:
    fbs = feedbacks[guess_id]

    _, feedback_freqs = np.unique(fbs, return_counts=True)
    probabilities = feedback_freqs / fbs.shape[0]
    information = -np.log2(probabilities)
    # expected info. aka (probabilities * information).sum()
    return probabilities.dot(information)


def actual_info_from_guess(guess: str, feedback: str, possible_words: np.ndarray) -> np.float64:
    # entropy
    guess_id = guess_index[guess]
    fbs = guess_feedbacks[guess_id, possible_words]

    pattern_id = pattern_index[feedback]
    freq = np.count_nonzero(fbs == pattern_id)
    probability = freq / fbs.shape[0]
    return -np.log2(probability)


def get_possible_words():
    return np.arange(len(ANSWERS))


def refine_possible_words(possible_words: np.ndarray, guess: str, feedback: str):
    guess_id = guess_index[guess]
    pattern_id = pattern_index[feedback]

    # current subset of answers
    subset = guess_feedbacks[guess_id, possible_words]
    next_possible_words = possible_words[subset == pattern_id]
    return next_possible_words


def best_guesses(possible_words: np.ndarray, k: int = 10):
    feedbacks = guess_feedbacks[:, possible_words]

    scores = np.array([score(guess_id, feedbacks) for guess_id in possible_words])
    best_ids = scores.argsort()[::-1]

    res = []
    for ind in best_ids[:k]:
        guess = GUESSES[possible_words[ind]]
        res.append((guess, scores[ind]))

    return res

# evals = []
# for guess in GUESSES:
#     s = score(guess, answer_feedbacks=guess_feedbacks)
#     evals.append((s, guess))
#
# evals.sort(reverse=True)
# for s, g in evals[:25]:
#     print(g, s)
