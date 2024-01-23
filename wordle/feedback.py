from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from typing import Mapping
from os import cpu_count
from math import ceil
from functools import partial
from pathlib import Path
from wordle.types import FeedbackIndexType
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


def feedbacks_for_guess(guess: str, answers: Iterable[str], pattern_id: Mapping[str, int]) -> tuple[int, ...]:
    return tuple(
        pattern_id[grade_guess(guess, answer)]
        for answer in answers
    )


def compute_guess_feedbacks_array(guesses: Sequence[str],
                                  answers: Sequence[str],
                                  pattern_index: Mapping[str, int]) -> np.ndarray:
    # FeedbackType = np.dtype((np.uint8, len(answers)))
    compute_feedbacks_for_guess = partial(feedbacks_for_guess, answers=answers, pattern_id=pattern_index)
    num_workers = cpu_count() or 1

    with ProcessPoolExecutor(num_workers) as executor:
        chunk_size = ceil(len(guesses) / num_workers)
        return np.fromiter(
            executor.map(compute_feedbacks_for_guess, guesses, chunksize=chunk_size),
            dtype=(FeedbackIndexType, len(answers)),
            count=len(guesses)
        )


def get_guess_feedbacks_array(guesses: Sequence[str],
                              answers: Sequence[str],
                              pattern_index: Mapping[str, int],
                              file_path: Path) -> np.ndarray:
    try:
        guess_feedbacks_array = np.load(file_path)
    except (OSError, ValueError) as e:
        guess_feedbacks_array = compute_guess_feedbacks_array(guesses, answers, pattern_index)
        np.save(file_path, guess_feedbacks_array)

    return guess_feedbacks_array
