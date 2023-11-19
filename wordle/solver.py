import functools
from typing import Mapping

from wordle.feedback import get_guess_feedbacks_array
from wordle.lib import Pattern
import numpy as np

from wordle.utils import feedback_inverted_index, lexmax
from wordle.solutiontree import SolutionTree

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
with open('wordle/data/allowed_except_relevant.txt', 'r') as f:
    words = map(str.strip, f)
    OTHER_WORDS = tuple(map(str.upper, words))

with open('wordle/data/updated_words.txt', 'r') as f:
    words = map(str.strip, f)
    RELEVANT_WORDS = tuple(map(str.upper, words))

with open('wordle/data/wordlist_nyt20230701_hidden', 'r') as f:
    words = map(str.strip, f)
    HIDDEN_ANSWERS = tuple(map(str.upper, words))

ANSWERS = HIDDEN_ANSWERS
GUESSES = RELEVANT_WORDS + OTHER_WORDS


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


class Wordle:
    guesses = GUESSES
    answers = ANSWERS
    word_index = {guess: i
                  for i, guess in enumerate(guesses)}

    patterns = Pattern.ALL_PATTERNS
    pattern_index = {pattern: i
                     for i, pattern in enumerate(patterns)}


    word_freqs = get_word_frequencies(word_index)
    scaled_word_freqs = word_freqs / word_freqs[:len(answers)].min()
    relative_word_freqs = word_freqs / word_freqs.sum()

    def __init__(self):
        self.guess_feedbacks_array = get_guess_feedbacks_array(self.guesses, RELEVANT_WORDS, self.pattern_index)
        # self.word_freqs = get_word_frequencies(self.word_index)
        # self.relative_word_freqs = self.word_freqs / self.word_freqs.sum()

    def entropy(self, guess_id: int, possible_answers) -> np.float64:
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]

        patterns, num_answers_for_patterns = np.unique(feedbacks, return_counts=True)

        answer_dist = num_answers_for_patterns / feedbacks.size
        information = -np.log2(answer_dist)
        return answer_dist.dot(information)

    def score(self, guess_id: int, possible_answers) -> np.float64:
        ent = self.entropy(guess_id, possible_answers)
        return ent
        # return self.score3(guess_id, possible_answers)
        # return self.score4(guess_id, possible_words)

    def score2(self, guess_id: int, possible_words) -> np.float64:
        feedbacks = self.guess_feedbacks_array[guess_id, possible_words]

        patterns, num_answers_for_patterns = np.unique(feedbacks, return_counts=True)
        answers_matching_pattern = feedback_inverted_index(feedbacks, num_answers_for_patterns)
        cum_freqs = [self.word_freqs[possible_words[ind]].sum()
                     for ind in answers_matching_pattern]

        total_freq = self.word_freqs[possible_words].sum()
        feedback_probability = np.array(cum_freqs) / total_freq
        # information = -np.log2(feedback_probability)

        feedback_perc = num_answers_for_patterns / feedbacks.size
        information = -np.log2(feedback_perc)
        return feedback_probability.dot(information)

    def score3(self, guess_id: int, possible_words) -> np.float64:
        ent = self.entropy(guess_id, possible_words)
        p = self.word_freqs[guess_id] / self.word_freqs[possible_words].sum()
        return p + ent

    def score4(self, guess_id: int, possible_words) -> np.float64:
        # partitions
        feedbacks = self.guess_feedbacks_array[guess_id, possible_words]
        p = self.word_freqs[guess_id] / self.word_freqs[possible_words].sum()
        return p + np.unique(feedbacks).size

    def partitions(self, guess_id: int, possible_answers):
        # partitions
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]
        return np.unique(feedbacks).size

    def partitions_and_max(self, guess_id: int, possible_answers):
        # partitions
        # feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]
        # pats, pat_freqs = np.unique(feedbacks, return_counts=True)
        # return pats.size, np.max(pat_freqs)
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]
        bincounts = np.bincount(feedbacks)
        parts = bincounts[bincounts > 0]

        return parts.size, parts.max()



    def score5(self, guess_id: int, possible_words):
        feedbacks = self.guess_feedbacks_array[guess_id, possible_words]
        patterns, pattern_sizes = np.unique(feedbacks, return_counts=True)
        p = self.word_freqs[guess_id] / self.word_freqs[possible_words].sum()

        return p + patterns.size, np.max(pattern_sizes)

    # def partitions_deep(self, guess_id: int, possible_words) -> np.float64:
    #     feedbacks = self.guess_feedbacks_array[guess_id, possible_words]
    #     patterns = np.unique(feedbacks)
    #     for pat in patterns:
    #         next_possible_words = self.refine_wordset(possible_words, guess_id, pat)
    #

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

    def refine_possible_words(self, possible_words: np.ndarray, guess: str, feedback: str):
        guess_id = self.word_index[guess]
        pattern_id = self.pattern_index[feedback]

        # current subset of answers
        subset = self.guess_feedbacks_array[guess_id, possible_words]
        next_possible_words = possible_words[subset == pattern_id]
        return next_possible_words

    def best_guess2(self, possible_guesses, possible_answers) -> int:
        # scores = np.array([self.score(guess_id, possible_words) for guess_id in possible_words],
        #                   dtype=[('score', np.float64), ('max_bucket_size', int)])
        ent = np.array([self.entropy(guess_id, possible_answers) for guess_id in possible_guesses])
        # parts = np.array([self.partitions(guess_id, possible_answers) for guess_id in possible_guesses])
        p_is_answer = self.word_freqs[possible_guesses] / self.word_freqs[possible_guesses].sum()
        # i = np.lexsort((parts, p_is_answer))[-1]
        # i = np.argmax(parts + p_is_answer)
        i = np.argmax(ent + p_is_answer)
        return possible_guesses[i]

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

    def best_guesses(self, possible_words: np.ndarray, k: int = 10):
        # feedbacks = guess_feedbacks_array[:, possible_words]
        scores = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])

        # exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        # candidate_freqs = self.word_freqs[possible_words]
        # p_is_word = candidate_freqs / candidate_freqs.sum()
        # scores = exp_info + p_is_word
        # scores = p_is_word + (1 - p_is_word) * exp_info

        best_ids = scores.argsort()[::-1]

        res = []
        for ind in best_ids[:k]:
            guess = self.guesses[possible_words[ind]]
            res.append((guess, scores[ind]))

        return res


class Game(Wordle):
    def __init__(self, answer: str, hard_mode: bool = True):
        super().__init__()
        from wordle.feedback import grade_guess
        self.grade_guess = functools.partial(grade_guess, answer=answer)
        self.answer = answer
        self.possible_guesses = np.arange(
            len(self.guesses) if not hard_mode
            else len(RELEVANT_WORDS)
        )
        self.possible_answers = np.arange(len(self.answers))
        self.hard_mode = hard_mode
        self.history = []
        self.history = {}

    def play(self, guess: str, feedback: None | str = None) -> str:
        pg, pa = self.possible_guesses, self.possible_answers
        if feedback is None:
            feedback = self.grade_guess(guess)

        self.possible_answers = self.refine_possible_words(pa, guess, feedback)

        if self.hard_mode:
            self.possible_guesses = self.refine_possible_words(pg, guess, feedback)
            # self.possible_guesses = self.possible_answers

        self.history[guess] = feedback

        return feedback
    def with_guesses(self, *guesses: str):
        assert self.answer != ''

        for guess in guesses:
            self.play(guess)

        return self

    def score_guess(self, guess: str):
        guess_id = self.word_index[guess]
        ent = self.entropy(guess_id, self.possible_answers)
        parts, max_part = self.partitions_and_max(guess_id, self.possible_answers)
        return ent, parts, max_part

    def top_guesses(self, k: int = 10):
        pg, pa = self.possible_guesses, self.possible_answers

        if pa.size == 1:
            guess = self.guesses[pa[0]]
            return [(guess, 0.0, 1, 1, self.scaled_word_freqs[pa[0]], True)]

        ents = np.array([self.entropy(guess_id, pa) for guess_id in pg])
        parts_maxpart = [self.partitions_and_max(guess_id, pa) for guess_id in pg]
        parts, max_part_size = np.array(parts_maxpart).T
        can_be_answer = np.isin(pg, pa, assume_unique=True)

        # exp_info = np.array([self.score(guess_id, possible_words) for guess_id in possible_words])
        candidate_freqs = self.relative_word_freqs[pg]
        # p_is_word = candidate_freqs / candidate_freqs.sum()
        # scores = exp_info + p_is_word
        # scores = p_is_word + (1 - p_is_word) * exp_info
        # best_ids = np.argsort(parts + can_be_answer * 0.1)[::-1]
        best_ids = np.lexsort((candidate_freqs, can_be_answer, parts))[::-1]
        res = []
        for ind in best_ids[:k]:
            guess = self.guesses[pg[ind]]
            res.append(
                (guess, ents[ind], parts[ind], max_part_size[ind], self.scaled_word_freqs[ind], can_be_answer[ind]))

        return res

    # def best_guess_cheat(self):
    #     if len(self.history) == 1 and self.history[0] == 'SLATE' and self.grade_guess('SLATE') == '⬛⬛🟨⬛🟨':
    #         return 'PAGLE'
    #     else:
    #         return self.best_guess()

    def best_guess(self) -> str:
        pg, pa = self.possible_guesses, self.possible_answers

        if pa.size == 1:
            return self.guesses[pa[0]]

        # ent, i = max((self.entropy(guess_id, pa), guess_id) for guess_id in pg)
        # ents = np.array([self.entropy(guess_id, pa)
        #                 for guess_id in pg])
        parts, max_part_size = np.array([self.partitions_and_max(guess_id, pa)
                                         for guess_id in pg]).T
        # parts = np.array([self.partitions(guess_id, pa)
        #                   for guess_id in pg]).T
        # word_freqs = self.word_freqs[self.possible_guesses]
        # p_is_answer = word_freqs / word_freqs.sum()
        # i = np.lexsort((parts, p_is_answer))[-1]
        can_be_answer = np.isin(pg, pa, assume_unique=True)
        # i = np.lexsort((parts, can_be_answer))[-1]
        # i = np.argmax(parts + p_is_answer)
        # i = np.argmax(ent + p_is_answer)
        # keys = np.fromiter(zip(parts, can_be_answer, -max_part_size),
        #                    dtype='i,b,i')
        # i = np.argmax(keys)
        i = lexmax(parts, can_be_answer, -max_part_size)
        # i = np.argmax(parts + can_be_answer * 0.5 - max_part_size / max_part_size.max() * 0.5)
        # i = np.argmax(parts + can_be_answer * 0.5)
        # i = np.argmax(ent)
        # return self.guesses[pg[i]]
        return self.guesses[pg[i]]

    def best_guess_id(self, pg, pa) -> int:
        if pa.size == 1:
            return pa[0]

        # ent, i = max((self.entropy(guess_id, pa), guess_id) for guess_id in pg)
        # ents = np.array([self.entropy(guess_id, pa)
        #                 for guess_id in pg])
        # parts, max_part_size = np.array([self.partitions_and_max(guess_id, pa)
        #                                  for guess_id in pg]).T
        parts = np.array([self.partitions(guess_id, pa)
                          for guess_id in pg])
        # word_freqs = self.word_freqs[self.possible_guesses]
        # p_is_answer = word_freqs / word_freqs.sum()
        # i = np.lexsort((parts, p_is_answer))[-1]
        # can_be_answer = np.isin(pg, pa, assume_unique=True)
        # i = np.lexsort((parts, can_be_answer))[-1]
        # i = np.argmax(parts + p_is_answer)
        # candidate_freqs = self.relative_word_freqs[pg]
        # i = np.argmax(ent + p_is_answer)

        # if not self.hard_mode or (possible_pillars := self.get_possible_pillars()).size == 0:
        #     i = lexmax(parts, can_be_answer, -max_part_size, candidate_freqs)
        #     return pg[i]


        # pillar_parts = np.array([self.partitions(guess_id, possible_pillars) for guess_id in pg])
        # pillar_parts, ppart_size_x = np.array([self.partitions_and_x(guess_id, possible_pillars)
        #                                        for guess_id in pg]).T
        # keys = np.fromiter(zip(parts, can_be_answer, -max_part_size),
        #                    dtype='i,b,i')
        # i = np.argmax(keys)
        # if possible_pillars.size / pa.size > 0.25:
        #     i = lexmax(~is_bait, parts + pillar_parts, can_be_answer, -max_part_size, candidate_freqs)
        # else:
        # i = lexmax(parts + pillar_parts - ppart_size_x + can_be_answer, -max_part_size, candidate_freqs)

        # i = np.argmax(parts + can_be_answer * 0.5 - max_part_size / max_part_size.max() * 0.5)
        # i = np.argmax(parts + can_be_answer * 0.5)
        # i = np.argmax(ent)
        # return self.guesses[pg[i]]
        i = np.argmax(parts)
        return pg[i]

    from typing import Iterable
    def with_limited_answers(self, words: Iterable[str]):
        word_ids = (self.word_index[word] for word in words)
        self.possible_answers = np.array(sorted(word_ids))
        return self


    def map_solutions(self, starting_word: str = 'SLATE') -> SolutionTree:
        guess_id = self.word_index[starting_word]
        possible_guesses = self.possible_guesses
        possible_answers = self.possible_answers

        return self._map_solutions(possible_guesses, possible_answers, guess_id)


    def _map_solutions(self,
                       possible_guesses: np.ndarray[int],
                       possible_answers: np.ndarray[int],
                       given_guess_id: int | None = None) -> SolutionTree:

        if possible_answers.size == 1:
            answer = self.answers[possible_answers[0]]
            return SolutionTree(answer, True)

        guess_id = given_guess_id if (given_guess_id is not None) else self.best_guess_id(possible_guesses, possible_answers)
        tree = SolutionTree(self.guesses[guess_id])

        feedback_ids = np.unique(self.guess_feedbacks_array[guess_id, possible_answers])
        for feedback_id in feedback_ids:
            feedback = self.patterns[feedback_id]
            if feedback == '🟩🟩🟩🟩🟩':
                tree.is_answer = True
            else:
                next_possible_guesses = self.refine_wordset(possible_guesses, guess_id, feedback_id)
                next_possible_answers = self.refine_wordset(possible_answers, guess_id, feedback_id)
                tree[feedback] = self._map_solutions(next_possible_guesses, next_possible_answers)

        return tree

# def cmp_scoring():
#     tg1 = w.top_guesses(score_fn=w.entropy, k=25)
#     tg2 = w.top_guesses(score_fn=w.score3, k=25)
#     tg3 = w.top_guesses(score_fn=w.score4, k=25)
#
#     for (g1, s1), (g2, s2), (g3, s3) in zip(tg1, tg2, tg3):
#         out = f'{g1} {s1:.2f} {g2} {s2:.2f} {g3} {s3:.2f}'
#         print(out)
#     print()
#
# w = Game('BAKER')
# w.play('SLATE')
# cmp_scoring()
# w.play('RACED')
# cmp_scoring()

#
# w = Game('ABASE')
# w.play('SLATE')
# # w.score_guess('UKASE')
# w.score_guess('PHASE')
# w.score_guess('ABASE')

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
