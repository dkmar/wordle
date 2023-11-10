from typing import Mapping

from wordle.feedback import get_guess_feedbacks_array
from wordle.lib import Pattern
import numpy as np
''' Gameplay (Hard Mode)
1. Initial state has the target word.
2. We play up to six rounds wherein we can make a guess and get feedback.
The best guess is based on our knowledge, which is our cumulative feedback.
We want to score guess candidates.
The score of a guess =
    sum of (pattern probability * pattern information)
Our guess candidates consist of the remaining wordset.

Given our feedback, I want to return a top k of guess candidates

We play a word from the wordset.
We get feedback, this feedback prunes the wordset.
We want to choose the next word
- score candidates by how much they would prune the wordset
    - maybe lets have the wordset be a list of words
    - and every pruned wordset be a list of indices for candidate words
    OK this doesn't work cause you're using the answer to score candidates...
    which is cheating lol. We need to estimate the value of a word from all
    possible feedbacks we could get from using it.
- keep the top k candidates (for reporting)
repeat by playing a candidate
'''

with open('wordle/data/relevant_valid_words.txt', 'r') as f:
    words = map(str.strip, f)
    RELEVANT_WORDS = tuple(map(str.upper, words))

ANSWERS = RELEVANT_WORDS
GUESSES = RELEVANT_WORDS


def get_word_frequencies(word_index: Mapping[str, int]) -> np.ndarray:
    with open('wordle/data/word_freqs_2019_valid.txt', 'r') as f:
        lines = map(str.strip, f)
        freqs = [0] * len(word_index)
        for word, count in map(str.split, lines):
            word = word.upper()
            if word in word_index:
                i = word_index[word]
                freqs[i] = int(count)

        return np.array(freqs)

def entropy_to_expected_score(ent):
    """
    Based on a regression associating entropies with typical scores
    from that point forward in simulated games, this function returns
    what the expected number of guesses required will be in a game where
    there's a given amount of entropy in the remaining possibilities.
    """
    # Assuming you can definitely get it in the next guess,
    # this is the expected score
    min_score = 2**(-ent) + 2 * (1 - 2**(-ent))

    # To account for the likely uncertainty after the next guess,
    # and knowing that entropy of 11.5 bits seems to have average
    # score of 3.5, we add a line to account
    # we add a line which connects (0, 0) to (3.5, 11.5)
    return min_score + 1.5 * ent / 11.5

class Wordle:
    def __init__(self):
        self.guesses = GUESSES
        self.answers = ANSWERS
        self.word_index = {guess: i
                           for i, guess in enumerate(self.guesses)}

        self.pattern_index = {pattern: i
                              for i, pattern in enumerate(Pattern.ALL_PATTERNS)}
        self.guess_feedbacks_array = get_guess_feedbacks_array(self.guesses, self.answers, self.pattern_index)
        # self.word_freqs = get_word_frequencies(self.word_index)
        word_freqs = get_word_frequencies(self.word_index)
        self.word_freqs = word_freqs
        self.relative_word_freqs = word_freqs / word_freqs.sum()

        # weird 3b1b H0. the entropy of just the relative word frequencies?
        from scipy.stats import entropy
        self.H0 = entropy(self.relative_word_freqs, base=2, axis=0)

    def score(self, guess_id: np.uint16, possible_words) -> np.float64:
        fbs = self.guess_feedbacks_array[guess_id, possible_words]

        unique_patterns, feedback_freqs = np.unique(fbs, return_counts=True)
        scaled_freq_of_feedbacks = feedback_freqs / fbs.shape[0]

        # pattern_freqs = []
        # candidate_freqs = self.word_freqs[possible_words]
        # for pat in unique_patterns:
        #     ind = np.where(fbs == pat)
        #     pat_freq = candidate_freqs[ind].sum()
        #     pattern_freqs.append(pat_freq)
        #
        # scaled_freq_of_feedbacks = np.array(pattern_freqs) / candidate_freqs.sum()

        # p_is_word = candidate_freqs / candidate_freqs.sum()
        #
        # probabilities = scaled_freq_of_feedbacks * p_is_word
        information = -np.log2(scaled_freq_of_feedbacks)
        # try:
        #     np.seterr(all='raise')
        #     information = -np.log2(scaled_freq_of_feedbacks)
        # except FloatingPointError as e:
        #     breakpoint()
        # expected info. aka (probabilities * information).sum()
        return scaled_freq_of_feedbacks.dot(information)

    def score2(self, guess_id: int, possible_words, H1):
        fbs = self.guess_feedbacks_array[guess_id, possible_words]

        unique_patterns, feedback_freqs = np.unique(fbs, return_counts=True)
        scaled_freq_of_feedbacks = feedback_freqs / fbs.shape[0]

        # (pat_relative_freq, second_guess, H2s)
        res = []
        # candidate_freqs = self.word_freqs[possible_words]
        for pat, pat_relative_freq in zip(unique_patterns, scaled_freq_of_feedbacks):
            # ind = np.where(fbs == pat)

            sub_possible_words = self.refine_wordset(possible_words, guess_id, pat)
            sub_best_guess, sub_entropy = self.best_guess2(sub_possible_words)
            # sub_best_guess, sub_entropy = self.best_guess3(sub_possible_words, H1)


            res.append((pat_relative_freq, sub_best_guess, sub_entropy))

        return res
        # scaled_freq_of_feedbacks = np.array(pattern_freqs) / candidate_freqs.sum()
        #
        # # p_is_word = candidate_freqs / candidate_freqs.sum()
        # #
        # # probabilities = scaled_freq_of_feedbacks * p_is_word
        # information = -np.log2(scaled_freq_of_feedbacks)
        # # try:
        # #     np.seterr(all='raise')
        # #     information = -np.log2(scaled_freq_of_feedbacks)
        # # except FloatingPointError as e:
        # #     breakpoint()
        # # expected info. aka (probabilities * information).sum()
        # return scaled_freq_of_feedbacks.dot(information)

    def actual_info_from_guess(self, guess: str, feedback: str, possible_words: np.ndarray) -> np.float64:
        # entropy
        guess_id = self.word_index[guess]
        fbs = self.guess_feedbacks_array[guess_id, possible_words]

        pattern_id = self.pattern_index[feedback]
        freq = np.count_nonzero(fbs == pattern_id)
        probability = freq / fbs.shape[0]
        return -np.log2(probability)

    def get_possible_words(self):
        return np.arange(len(ANSWERS))

    def refine_wordset(self, possible_words: np.ndarray, guess_id: int, feedback_id: int):
        # current subset of answers
        subset = self.guess_feedbacks_array[guess_id, possible_words]
        next_possible_words = possible_words[subset == feedback_id]
        return next_possible_words

    def refine_possible_words(self,possible_words: np.ndarray, guess: str, feedback: str):
        guess_id = self.word_index[guess]
        pattern_id = self.pattern_index[feedback]

        # current subset of answers
        subset = self.guess_feedbacks_array[guess_id, possible_words]
        next_possible_words = possible_words[subset == pattern_id]
        return next_possible_words

    def best_guess(self, possible_words: np.ndarray) -> int:
        scores = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        # exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        #
        # candidate_freqs = self.word_freqs[possible_words]
        # p_is_word = candidate_freqs / candidate_freqs.sum()
        #
        # scores = exp_info + p_is_word
        # scores = p_is_word * np.log2(possible_words.size) + (1 - p_is_word) * (exp_info)

        guess_id = possible_words[scores.argmax()]
        return guess_id

    def best_guess2(self, possible_words: np.ndarray) -> tuple[int, np.float64]:
        entropies = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        # exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        #
        # candidate_freqs = self.word_freqs[possible_words]
        # p_is_word = candidate_freqs / candidate_freqs.sum()
        #
        # scores = exp_info + p_is_word
        # scores = p_is_word * np.log2(possible_words.size) + (1 - p_is_word) * (exp_info)
        i = entropies.argmax()
        guess_id = possible_words[i]
        return guess_id, entropies[i]

    def best_guess3(self, possible_words: np.ndarray, H1) -> tuple[int, np.float64]:
        entropies = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        # exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        #
        # candidate_freqs = self.word_freqs[possible_words]
        # p_is_word = candidate_freqs / candidate_freqs.sum()
        #
        # scores = exp_info + p_is_word
        # scores = p_is_word * np.log2(possible_words.size) + (1 - p_is_word) * (exp_info)
        probs = self.relative_word_freqs[possible_words]
        expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(self.H0 - H1 - entropies))
        i = expected_scores.argmin()
        guess_id = possible_words[i]
        return guess_id, entropies[i]

    def best_guess_3b1b(self, possible_words: np.ndarray, extra_depth=False) -> int:
        entropies = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        probs = self.relative_word_freqs[possible_words]
        expected_scores = probs + (1 - probs) * (1 + entropy_to_expected_score(self.H0 - entropies))

        if not extra_depth:
            guess_id = possible_words[expected_scores.argmin()]
            return guess_id

        best = np.argsort(expected_scores)
        k = 10
        # push up the rest
        expected_scores += 1
        for i in best[:10]:
            '''
            for every possible feedback pattern given we guess i as guess1, consider the best next guess
            '''
            guess1_entropy = entropies[i]
            guess_id = possible_words[i]
            p = probs[i]
            # (pat_relative_freq, second_guess given pat, H2s aka entropy given pat)
            dist, second_guesses, H2s = list(zip(*self.score2(guess_id, possible_words, guess1_entropy)))

            # sub_possible_words = self.refine_wordset(possible_words, guess_id, self.guess_feedbacks_array[guess_id, ])
            expected_scores[i] = sum((
                1 * p,
                2 * (1 - p) * sum(p2 * self.relative_word_freqs[guess2] for p2, guess2 in zip(dist, second_guesses)),
                (1 - p) * (2 + sum(
                    p2 * (1 - self.relative_word_freqs[guess2]) * entropy_to_expected_score(self.H0 - guess1_entropy - guess2_entropy)
                    for p2, guess2, guess2_entropy in zip(dist, second_guesses, H2s)
                ))
            ))

        guess_id = possible_words[expected_scores.argmin()]
        return guess_id

    def best_guesses(self, possible_words: np.ndarray, k: int = 10):
        # feedbacks = guess_feedbacks_array[:, possible_words]

        exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        candidate_freqs = self.word_freqs[possible_words]
        p_is_word = candidate_freqs / candidate_freqs.sum()
        scores = exp_info + p_is_word
        # scores = p_is_word + (1 - p_is_word) * exp_info

        best_ids = scores.argsort()[::-1]

        res = []
        for ind in best_ids[:k]:
            guess = self.guesses[possible_words[ind]]
            res.append((guess, scores[ind]))

        return res

# class Feedback:
#     def __init__(self, guess: str, pattern: Pattern):
#         self.guess = guess
#         self.pattern = pattern
#
#     def __repr__(self):
#         return self.guess + '\n' + str(self.pattern)
#
#     def matchesWord(self, word: str) -> bool:
#         used = 0
#
#         for i, (g, fb, w) in enumerate(zip(self.guess, self.pattern, word)):
#             if fb == Status.Green:
#                 if g != w:
#                     return False
#                 used |= (1 << i)
#
#         for i, (g, fb, w) in enumerate(zip(self.guess, self.pattern, word)):
#             if fb == Status.Green:
#                 continue
#
#             if g == w:
#                 return False
#
#             for j, ch in enumerate(word):
#                 if ch == g and not used & (1 << j):
#                     if fb == Status.Grey:
#                         # should be yellow
#                         return False
#
#                     used |= (1 << j)
#                     break
#             else:
#                 if fb == Status.Yellow:
#                     # should be grey
#                     return False
#
#         return True
#
#
# class Game:
#     def __init__(self, answer: str):
#         self.answer = answer.upper()
#
#     def grade_guess(self, guess: str) -> Feedback:
#         """
#         What should feedback look like?
#         return indices of greens and yellows.
#
#         POOCH TABOO
#         _YY__
#
#         POOCH OTHER
#         _Y__Y
#         """
#         answer, guess = self.answer, guess.upper()
#         feedback = Pattern()
#         used = 0
#         # label greens
#         for i, (ch, ans) in enumerate(zip(guess, answer)):
#             if ch == ans:
#                 feedback[i] = Status.Green
#                 used |= (1 << i)
#
#         # label yellows
#         for i, (ch, fb) in enumerate(zip(guess, feedback)):
#             if fb == Status.Green:
#                 continue
#
#             # TODO: see if it even makes sense to usr find() instead of a generator like we do in matchesWord
#             j = -1
#             while (j := answer.find(ch, j + 1)) != -1:
#                 if not used & (1 << j):
#                     feedback[i] = Status.Yellow
#                     used |= (1 << j)
#                     break
#             else:
#                 feedback[i] = Status.Grey
#
#         return Feedback(guess, feedback)
#
#
# class Solver:
#     def __init__(self, wordset: list[str]):
#         self.wordset = wordset
#         self.history = []
#
#     def solved(self) -> bool:
#         return len(self.wordset) == 1
#
#     def play(self, guess: str, feedback: str):
#         fb = Feedback(guess.upper(), Pattern.from_str(feedback))
#         self.wordset = list(filter(fb.matchesWord, self.wordset))
#
#         self.history.append(fb)
#         return fb
#
#     def suggestions(self):
#         return Evaluation.topk(self.wordset)
#         # for word, score in Evaluation.topk(self.wordset):
#         #     print(f'{word}: {score}')
#
#
# class Evaluation:
#     ALL_PATTERNS = tuple(map(Pattern.from_int, Pattern.all_patterns()))
#     @staticmethod
#     def score(wordset: list[str], guess: str) -> float:
#         total_remaining = len(wordset)
#         expected_info = 0.0
#         for pattern in Evaluation.ALL_PATTERNS:
#             feedback = Feedback(guess, pattern)
#             matches = sum(map(feedback.matchesWord, wordset))
#             # matches = 0
#             # for word in wordset:
#             #     if feedback.matchesWord(word):
#             #         matches += 1
#
#             if matches > 0:
#                 pattern_probability = matches / total_remaining
#                 information = -math.log2(pattern_probability)
#                 expected_info += pattern_probability * information
#
#         return expected_info
#
#     @staticmethod
#     def topk(wordset: list[str], k: int = 10):
#         evals = []
#         for word in tqdm(wordset):
#             score = Evaluation.score(wordset, word)
#             evals.append((score, word))
#
#         top = heapq.nlargest(k, evals)
#         for score, word in top:
#             yield word, score


# wordset: list[str]
# with open('wordle/data/relevant_words.txt', 'r') as f:
#     words = map(str.strip, f)
#     wordset = list(map(str.upper, words))

# TODO test int conversions
# p = Pattern([Status.Grey] * 5)
# # p[0] = Status.Green
# print(p, p.to_int(), Pattern.from_int(p.to_int()))
# p = Pattern([Status.Green] * 5)
# # p[-2] = Status.Yellow
# print(p, p.to_int(), Pattern.from_int(p.to_int()))
#
# start = "CRANE"
# answer = 'SPLAT'
# game = Game(answer)
# startInfo = game.grade_guess(start)
# print(startInfo)

# wordset = list(filter(startInfo.matchesWord, wordset))
# relevant_words: list[str]
# with open('wordle/data/relevant_words.txt', 'r') as f:
#     words = map(str.strip, f)
#     relevant_words = list(map(str.upper, words))

# print('BIOTA', Evaluation.score('BIOTA'))
# print('SPLAT', Evaluation.score('SPLAT'))
# relevant_words = list(filter(startInfo.matchesWord, relevant_words))
# nextInfo = game.grade_guess('TAILS')
# relevant_words = list(filter(nextInfo.matchesWord, relevant_words))
# evals = []
# for word in tqdm(relevant_words):
#     s = Evaluation.score(word)
#     evals.append((s, word))
#
# topk = heapq.nlargest(10, evals)
# for s, w in topk:
#     print(f'{w}: {s}')

# print(Game('TABOO').grade_guess('POOCH'))
# print(Game('OTHER').grade_guess('POOCH'))
# print(Game('SPLAT').grade_guess('SPLAT'))
#
# info = Game('TABOO').grade_guess('_OO__')
# res = list(filter(info.matchesWord, wordset))
# print(len(res))

# for code in Pattern.all_patterns():
#     pat = Pattern.from_int(code)
#     print(pat)

# evals = []
# for guess in relevant_words:
#     s = Evaluation.score(relevant_words, guess)
#     evals.append((s, guess))
#
# evals.sort(reverse=True)
# for s, g in evals[:25]:
#     print(g, s)


# TODO add pytest and some cases
# TODO decide on scoring / evaluation
# TODO should probably delineate between pattern and feedback (word, pattern)

