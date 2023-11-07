from wordle.feedback import compute_guess_feedbacks_array
import numpy as np
import itertools

with open('wordle/data/relevant_words.txt', 'r') as f:
    words = map(str.strip, f)
    RELEVANT_WORDS = tuple(map(str.upper, words))

ANSWERS = RELEVANT_WORDS
GUESSES = RELEVANT_WORDS
guess_index = {guess: np.uint16(i)
               for i, guess in enumerate(GUESSES)}

pattern_index = {pattern: np.uint8(i)
                 for i, pattern in enumerate(ALL_PATTERNS)}


try:
    guess_feedbacks_array = np.load('wordle/data/guess_feedbacks_array.npy')
except (OSError, ValueError) as e:
    guess_feedbacks_array = compute_guess_feedbacks_array(GUESSES, ANSWERS, pattern_index)
    np.save('wordle/data/guess_feedbacks_array.npy', guess_feedbacks_array)


def score(guess_id: np.uint16, feedbacks: np.ndarray[np.uint8]) -> np.float64:
    fbs = feedbacks[guess_id]

    _, feedback_freqs = np.unique(fbs, return_counts=True)
    probabilities = feedback_freqs / fbs.shape[0]
    information = -np.log2(probabilities)
    # expected info. aka (probabilities * information).sum()
    return probabilities.dot(information)


def actual_info_from_guess(guess: str, feedback: str, possible_words: np.ndarray) -> np.float64:
    # entropy
    guess_id = guess_index[guess]
    fbs = guess_feedbacks_array[guess_id, possible_words]

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
    subset = guess_feedbacks_array[guess_id, possible_words]
    next_possible_words = possible_words[subset == pattern_id]
    return next_possible_words


def best_guesses(possible_words: np.ndarray, k: int = 10):
    feedbacks = guess_feedbacks_array[:, possible_words]

    scores = np.array([score(guess_id, feedbacks) for guess_id in possible_words])
    best_ids = scores.argsort()[::-1]

    res = []
    for ind in best_ids[:k]:
        guess = GUESSES[possible_words[ind]]
        res.append((guess, scores[ind]))

    return res



# if __name__ == '__main__':
#     # feedback: feedback for each answer if we used this guess
#     # guess_id -> [feedback pattern for answer in ANSWERS]
#     # FeedbackType = np.dtype((np.uint8, len(ANSWERS)))
#     # guess_feedbacks = np.fromiter(map(feedbacks_for_guess, GUESSES), dtype=FeedbackType, count=len(GUESSES))
#
#     try:
#         guess_feedbacks_array = np.load('wordle/data/guess_feedbacks_array.npy')
#     except (OSError, ValueError) as e:
#         guess_feedbacks_array = compute_guess_feedbacks_array(GUESSES, ANSWERS, pattern_index)
#         np.save('wordle/data/guess_feedbacks_array.npy', guess_feedbacks_array)
#
#     pa = np.arange(len(ANSWERS))
#     for guess, score in best_guesses(pa):
#         print(f'\t{guess}: {score}')

# evals = []
# for guess in GUESSES:
#     s = score(guess, answer_feedbacks=guess_feedbacks)
#     evals.append((s, guess))
#
# evals.sort(reverse=True)
# for s, g in evals[:25]:
#     print(g, s)