import numpy as np
from wordle.lib import Pattern
from wordle.utils import filter_possible_words
_NUM_PATTERNS = len(Pattern.ALL_PATTERNS)


def get_feedback_distribution(guess_feedbacks_array: np.ndarray,
                              guess_id: int,
                              possible_answers: np.ndarray[np.int16]) -> np.ndarray:
    possible_feedbacks = guess_feedbacks_array[guess_id, possible_answers]
    return np.bincount(possible_feedbacks, minlength=_NUM_PATTERNS)


def entropy(guess_feedbacks_array: np.ndarray, guess_id: int,
            possible_answers: np.ndarray[np.int16]) -> np.float64:
    feedbacks = guess_feedbacks_array[guess_id, possible_answers]

    # patterns, num_answers_for_patterns = np.unique(feedbacks, return_counts=True)
    bins = np.bincount(feedbacks, minlength=_NUM_PATTERNS)
    num_answers_for_patterns = bins[bins > 0]

    answer_dist = num_answers_for_patterns / feedbacks.size
    information = np.log2(answer_dist)
    return -answer_dist.dot(information)


def entropy_level2(guess_feedbacks_array: np.ndarray, guess_id: int,
                   possible_guesses: np.ndarray[np.int16], possible_answers: np.ndarray[np.int16]) -> float:
    feedbacks = guess_feedbacks_array[guess_id, possible_answers]
    bins = np.bincount(feedbacks, minlength=_NUM_PATTERNS)
    unique_feedbacks = bins.nonzero()[0]
    entropies = []

    for feedback_id in unique_feedbacks:
        next_possible_guesses = filter_possible_words(guess_feedbacks_array, possible_guesses,
                                                      guess_id, feedback_id)
        next_possible_answers = filter_possible_words(guess_feedbacks_array, possible_answers,
                                                      guess_id, feedback_id)
        ent = max(entropy(guess_feedbacks_array, guess_id, next_possible_answers)
                  for guess_id in next_possible_guesses)
        entropies.append(ent)

    p = bins[unique_feedbacks] / feedbacks.size
    return p.dot(np.array(entropies))


def partitions(guess_feedbacks_array: np.ndarray, guess_id: int, possible_answers: np.ndarray[np.int16]) -> int:
    # partitions
    feedbacks = guess_feedbacks_array[guess_id, possible_answers]
    bins = np.bincount(feedbacks, minlength=_NUM_PATTERNS)
    return np.count_nonzero(bins)


def partitions_and_max(guess_feedbacks_array: np.ndarray, guess_id: int,
                       possible_answers: np.ndarray[np.int16]) -> tuple[int, int]:
    # partitions
    feedbacks = guess_feedbacks_array[guess_id, possible_answers]
    bins = np.bincount(feedbacks, minlength=_NUM_PATTERNS)
    bins = bins[bins > 0]
    return bins.size, bins.max()


def partitions_level2(guess_feedbacks_array: np.ndarray, guess_id: int,
                      possible_guesses: np.ndarray[np.int16], possible_answers: np.ndarray[np.int16]) -> float:
    feedbacks = guess_feedbacks_array[guess_id, possible_answers]
    bins = np.bincount(feedbacks, minlength=_NUM_PATTERNS)
    possible_feedbacks = bins.nonzero()[0]

    total_parts = 0
    num_buckets = possible_feedbacks.size

    for feedback_id in possible_feedbacks:
        next_possible_guesses = filter_possible_words(guess_feedbacks_array, possible_guesses,
                                                      guess_id, feedback_id)
        next_possible_answers = filter_possible_words(guess_feedbacks_array, possible_answers,
                                                      guess_id, feedback_id)
        parts = max(partitions(guess_feedbacks_array, guess_id, next_possible_answers)
                    for guess_id in next_possible_guesses)
        total_parts += parts

    return total_parts / num_buckets


def pillar_aware_heuristic(guess_feedbacks_array: np.ndarray,
                           pillars_of_doom: np.ndarray[np.int16],
                           possible_guesses: np.ndarray[np.int16],
                           possible_answers: np.ndarray[np.int16]
                           ) -> tuple[np.ndarray, ...]:
    num_partitions = np.array([partitions(guess_feedbacks_array, guess_id, possible_answers)
                               for guess_id in possible_guesses])

    can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)

    possible_pillars = np.intersect1d(possible_answers, pillars_of_doom, assume_unique=True)
    if possible_pillars.size == 0 or possible_answers.size <= 10:
        return num_partitions, can_be_answer

    num_pillar_partitions = np.array([partitions(guess_feedbacks_array, guess_id, possible_pillars)
                                      for guess_id in possible_guesses])
    is_pillar = np.isin(possible_guesses, pillars_of_doom, assume_unique=True)
    # TODO: revisit this penalty? the concept is we reduce the pillar partition contributions for pillar guesses
    # we need to retain the best guesses within as small a pool as possible (the k=20).
    pillar_penalty = np.mean(num_pillar_partitions) / np.log10(possible_answers.size)

    return num_partitions + (num_pillar_partitions - pillar_penalty * is_pillar), can_be_answer


def basic_heuristic(guess_feedbacks_array: np.ndarray,
                    possible_guesses: np.ndarray[np.int16],
                    possible_answers: np.ndarray[np.int16]
                    ) -> tuple[np.ndarray, ...]:
    num_partitions = np.array([partitions(guess_feedbacks_array, guess_id, possible_answers)
                               for guess_id in possible_guesses])
    can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)
    return num_partitions, can_be_answer

def basic_heuristic_ents(guess_feedbacks_array: np.ndarray,
                         possible_guesses: np.ndarray[np.int16],
                         possible_answers: np.ndarray[np.int16]
                         ) -> tuple[np.ndarray, ...]:
    ents = np.array([entropy(guess_feedbacks_array, guess_id, possible_answers)
                     for guess_id in possible_guesses])
    can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)
    return ents, can_be_answer


def bluebrown(ent):
    # ent = 5.614710
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2 ** (-ent) + 2 * (1 - 2 ** (-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5

def expected_score_from_remaining_entropy(rent):
    # log2(16) = 4
    # 2^-4 = 0.0625 = 1/16
    p = 2 ** (-rent)
    floor = p + (1 - p) * 2

    return floor + 0.135474 * rent

def basic_heuristic2(guess_feedbacks_array: np.ndarray,
                     possible_guesses: np.ndarray[np.int16],
                     possible_answers: np.ndarray[np.int16]
                    ) -> tuple[np.ndarray, ...]:
    ents = np.array([entropy(guess_feedbacks_array, guess_id, possible_answers)
                     for guess_id in possible_guesses])
    can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)

    remaining_entropy = np.log2(possible_answers.size)
    p = (1 / possible_answers.size) * can_be_answer
    # remaining_guesses = 1.149569 + 0.247628 * (remaining_entropy - ents)
    # remaining_guesses = bluebrown(remaining_entropy - ents)
    # next_remaining_entropy = remaining_entropy - ents
    # remaining_guesses = 1.127971 + 0.568010 * np.log2(next_remaining_entropy,
    #                                                   out=np.zeros_like(next_remaining_entropy),
    #                                                   where=(next_remaining_entropy > 0))
    # remaining_guesses = np.exp(0.130605 + 0.135474 * (remaining_entropy - ents))
    remaining_guesses = expected_score_from_remaining_entropy(remaining_entropy - ents)
    exp_guesses = p + (1 - p) * (1 + remaining_guesses)

    # print(remaining_entropy, ents.max(), exp_guesses.min())
    return -exp_guesses, can_be_answer




