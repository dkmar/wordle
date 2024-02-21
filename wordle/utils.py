import numpy as np
from wordle.types import WordIndexArray


def feedback_inverted_index(feedbacks, feedback_counts):
    # feedbacks = self.guess_feedbacks_array[guess_id]

    # inverted index. pattern to words
    indices = np.argsort(feedbacks)
    # feedback_freqs = np.bincount(feedbacks)
    answers_matching_pattern = np.split(indices, np.cumsum(feedback_counts)[:-1])

    return answers_matching_pattern


def lexmax(*keys: np.ndarray) -> int:
    """
    Lexicographical version of argmax. Intended for 1D arrays.

    Example:
        keys: a, b
        where a = [3, 3, 1]
              b = [5, 0, 1]

        returns 0 (the index of the max, (3,5))
    """
    candidates: np.ndarray | None = None
    for key in keys:
        if candidates is None:
            candidates = np.where(key == key.max())[0]
        else:
            subset = key[candidates]
            best = np.where(subset == subset.max())[0]
            candidates = candidates[best]

        if candidates.size == 1:
            break

    return candidates[0]


def filter_possible_words(guess_feedbacks_array: np.ndarray,
                          possible_words: WordIndexArray,
                          guess_id: int, feedback_id: int) -> WordIndexArray:
    # current subset of answers
    possible_feedbacks = guess_feedbacks_array[guess_id, possible_words]
    next_possible_words = possible_words[possible_feedbacks == feedback_id]
    return next_possible_words
