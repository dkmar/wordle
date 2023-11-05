from concurrent.futures import ProcessPoolExecutor
from typing import Mapping
from os import cpu_count
from math import ceil
from functools import partial
import numpy as np


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


def feedbacks_for_guess(guess: str, answers: tuple[str], pattern_id: Mapping[str, np.uint8]) -> list[np.uint8]:
    return [pattern_id[grade_guess(guess, answer)] for answer in answers]


def compute_guess_feedbacks_array(guesses: tuple[str, ...],
                                  answers: tuple[str, ...],
                                  pattern_index: Mapping[str, np.uint8]):
    # FeedbackType = np.dtype((np.uint8, len(answers)))
    compute_feedbacks_for_guess = partial(feedbacks_for_guess, answers=answers, pattern_id=pattern_index)
    num_workers = cpu_count() or 1

    with ProcessPoolExecutor(num_workers) as executor:
        chunk_size = ceil(len(guesses) / num_workers)
        return np.fromiter(
            executor.map(compute_feedbacks_for_guess, guesses, chunksize=chunk_size),
            dtype=(np.uint8, len(answers)),
            count=len(guesses)
        )