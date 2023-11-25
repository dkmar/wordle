import functools
from collections.abc import Callable, Mapping, Iterable

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

def get_pillars_of_doom(word_index: Mapping[str, int]) -> np.ndarray:
    with open('wordle/data/pillars_of_doom.txt', 'r') as f:
        lines = map(str.strip, f)
        words = map(str.upper, lines)
        return np.array([word_index[word]
                         for word in words if word in word_index], dtype=np.int16)


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

pillars_of_doom = get_pillars_of_doom(word_index)


def refine_possible_words(possible_words: np.ndarray[np.int16],
                          guess_feedbacks_array: np.ndarray,
                          guess_id: int, feedback_id: int) -> np.ndarray[np.int16]:
    # current subset of answers
    subset = guess_feedbacks_array[guess_id, possible_words]
    next_possible_words = possible_words[subset == feedback_id]
    return next_possible_words


class Game:
    def __init__(self,
                 guess_feedbacks_array: np.ndarray,
                 answer: str = '', hard_mode: bool = True):
        self.guess_feedbacks_array = guess_feedbacks_array
        self.answer = answer
        self.possible_guesses = np.arange(len(guesses), dtype=np.int16)
        self.possible_answers = np.arange(len(answers), dtype=np.int16)
        self.hard_mode = hard_mode
        self.history = {}

        from wordle.feedback import grade_guess
        self.grade_guess = functools.partial(grade_guess, answer=answer)

    # builders
    def with_guesses(self, *guesses: str):
        assert self.answer != ''

        for guess in guesses:
            self.play(guess)

        return self

    def with_limited_answers(self, words: Iterable[str]):
        word_ids = (word_index[word] for word in words)
        self.possible_answers = np.array(sorted(word_ids))
        return self

    def play(self, guess: str, feedback: None | str = None) -> str:
        if feedback is None:
            assert len(self.answer) == 5
            feedback = self.grade_guess(guess)

        guess_id = word_index[guess]
        feedback_id = pattern_index[feedback]

        self.possible_answers = refine_possible_words(self.possible_answers, self.guess_feedbacks_array,
                                                      guess_id, feedback_id)
        if self.hard_mode:
            self.possible_guesses = refine_possible_words(self.possible_guesses, self.guess_feedbacks_array,
                                                          guess_id, feedback_id)

        self.history[guess] = feedback

        return feedback


class WordleSolver:
    def __init__(self, hard_mode: bool = False):
        guess_feedbacks_array = get_guess_feedbacks_array(guesses, guesses, pattern_index)
        self.guess_feedbacks_array = guess_feedbacks_array
        self.hard_mode = hard_mode
        self.game = None
        self.optimal = False
        self.solution_tree = None

    def for_answer(self, answer: str):
        self.new_game(answer)
        return self

    def with_optimal_tree(self, starting_word: str):
        if not self.game:
            self.new_game('')

        self.optimal = True
        self.solution_tree = self.map_solutions(starting_word, find_optimal=True)
        return self


    def new_game(self, answer: str):
        self.game = Game(self.guess_feedbacks_array,
                         answer=answer,
                         hard_mode=self.hard_mode)


    def entropy(self, guess_id: int, possible_answers: np.ndarray[np.int16]) -> np.float64:
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]

        # patterns, num_answers_for_patterns = np.unique(feedbacks, return_counts=True)
        bins = np.bincount(feedbacks, minlength=len(patterns))
        num_answers_for_patterns = bins[bins > 0]

        answer_dist = num_answers_for_patterns / feedbacks.size
        information = np.log2(answer_dist)
        return -answer_dist.dot(information)

    def partitions(self, guess_id: int, possible_answers: np.ndarray[np.int16]) -> int:
        # partitions
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]
        bins = np.bincount(feedbacks, minlength=len(patterns))
        return np.count_nonzero(bins)

    def partitions_and_max(self, guess_id: int, possible_answers: np.ndarray[np.int16]) -> tuple[int, int]:
        # partitions
        feedbacks = self.guess_feedbacks_array[guess_id, possible_answers]
        bins = np.bincount(feedbacks, minlength=len(patterns))
        bins = bins[bins > 0]
        return bins.size, bins.max()

    def pillar_aware_heuristic(self,
                               possible_guesses: np.ndarray[np.int16],
                               possible_answers: np.ndarray[np.int16]
    ) -> tuple[np.ndarray, ...]:
        partitions = np.array([self.partitions(guess_id, possible_answers)
                               for guess_id in possible_guesses])

        can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)

        possible_pillars = np.intersect1d(possible_answers, pillars_of_doom, assume_unique=True)
        if possible_pillars.size == 0 or possible_answers.size <= 10:
            return partitions, can_be_answer

        pillar_partitions = np.array([self.partitions(guess_id, possible_pillars)
                                      for guess_id in possible_guesses])
        can_be_pillar = np.isin(possible_guesses, pillars_of_doom, assume_unique=True)
        # TODO: revisit this penalty? the concept is we reduce the pillar partition contributions for pillar guesses
        # we need to retain the best guesses within as small a pool as possible (the k=20).
        pillar_penalty = np.mean(pillar_partitions) / np.log10(possible_answers.size)

        return partitions + pillar_partitions - (pillar_penalty * can_be_pillar), can_be_answer

    def basic_heuristic(self,
                        possible_guesses: np.ndarray[np.int16],
                        possible_answers: np.ndarray[np.int16]
    ) -> tuple[np.ndarray, ...]:
        partitions = np.array([self.partitions(guess_id, possible_answers)
                               for guess_id in possible_guesses])
        can_be_answer = np.isin(possible_guesses, possible_answers, assume_unique=True)
        return partitions, can_be_answer

    def best_guess(self) -> str:
        if self.optimal:
            tree = self.solution_tree
            history = self.game.history
            curr = tree
            for guess, feedback in history.items():
                curr = curr[feedback]

            return curr.guess

        game = self.game
        scorefn = self.pillar_aware_heuristic if self.hard_mode else self.basic_heuristic
        guess_id = self._best_guesses(game.possible_guesses, game.possible_answers,
                                      scorefn=scorefn,
                                      k=1)[0]

        return guesses[guess_id]

    def top_guesses_info(self, k: int = 10) -> list:
        '''
        guess, "score", entropy, partitions, max partition size, pillar partitions, scaled word freq, can be answer?
        '''
        game = self.game
        possible_guesses, possible_answers = game.possible_guesses, game.possible_answers

        scorefn = self.pillar_aware_heuristic if self.hard_mode else self.basic_heuristic
        top_ids = self._best_guesses(possible_guesses, possible_answers,
                                     scorefn=scorefn,
                                     k=k)

        keys = scorefn(top_ids, possible_answers)
        inds = np.lexsort(keys[::-1])[::-1]
        top_ids = top_ids[inds]

        ents = np.array([self.entropy(guess_id, possible_answers) for guess_id in top_ids])
        partitions, max_partition = np.array([self.partitions_and_max(guess_id, possible_answers)
                                              for guess_id in top_ids]).T
        can_be_answer = np.isin(top_ids, possible_answers, assume_unique=True)

        possible_pillars = np.intersect1d(possible_answers, pillars_of_doom, assume_unique=True)
        pillar_partitions = np.array([self.partitions(guess_id, possible_pillars)
                                      for guess_id in top_ids])

        top_guesses = (guesses[gid] for gid in top_ids)
        info = zip(top_guesses, keys[0][inds], ents, partitions, max_partition, pillar_partitions, scaled_word_freqs[top_ids], can_be_answer)
        return [guess_info for guess_info in info]


    def _best_guesses(self,
                      possible_guesses: np.ndarray[np.int16],
                      possible_answers: np.ndarray[np.int16],
                      scorefn: Callable[[np.ndarray[np.int16], np.ndarray[np.int16]], tuple[np.ndarray, ...]],
                      k: int = 20
    ) -> np.ndarray:
        if possible_answers.size == 1:
            return possible_answers

        if possible_guesses.size <= k:
            return possible_guesses

        keys = scorefn(possible_guesses, possible_answers)

        if k == 1:
            i = lexmax(*keys) if len(keys) > 1 else np.argmax(keys[0])
            return possible_guesses[[i]]

        key = np.fromiter(zip(*keys), dtype='f,b') if len(keys) > 1 else keys[0]
        inds = np.argpartition(key, -k)[-k:]
        return possible_guesses[inds]




    def map_solutions(self, starting_word: str = '', find_optimal: bool = False) -> SolutionTree:
        possible_guesses = self.game.possible_guesses
        possible_answers = self.game.possible_answers

        if starting_word:
            guess_id = word_index[starting_word]
            if find_optimal:
                return self._map_solutions_optimal({}, possible_guesses, possible_answers, guess_id)
            else:
                return self._map_solutions_greedy(possible_guesses, possible_answers, guess_id)
        elif find_optimal:
            return self._map_solutions_optimal({}, possible_guesses, possible_answers)
        else:
            return self._map_solutions_greedy(possible_guesses, possible_answers)


    def _map_solutions_greedy(self,
                              possible_guesses: np.ndarray[np.int16],
                              possible_answers: np.ndarray[np.int16],
                              given_guess_id: int | None = None) -> SolutionTree:

        if possible_answers.size == 1:
            answer = answers[possible_answers[0]]
            return SolutionTree(answer, True)

        guess_id = given_guess_id if (given_guess_id is not None) \
            else self._best_guesses(possible_guesses, possible_answers, self.pillar_aware_heuristic, 1)
        tree = SolutionTree(guesses[guess_id])

        feedback_ids = np.unique(self.guess_feedbacks_array[guess_id, possible_answers])
        for feedback_id in feedback_ids:
            feedback = patterns[feedback_id]
            if feedback == '🟩🟩🟩🟩🟩':
                tree.is_answer = True
            else:
                next_possible_guesses = refine_possible_words(possible_guesses, self.guess_feedbacks_array,
                                                              guess_id, feedback_id)
                next_possible_answers = refine_possible_words(possible_answers, self.guess_feedbacks_array,
                                                              guess_id, feedback_id)
                tree[feedback] = self._map_solutions_greedy(next_possible_guesses, next_possible_answers)

        return tree

    def _map_solutions_optimal(self,
                               memo: dict,
                               possible_guesses: np.ndarray[np.int16],
                               possible_answers: np.ndarray[np.int16],
                               given_guess_id: int | None = None) -> SolutionTree:

        if possible_answers.size == 1:
            answer = answers[possible_answers[0]]
            return SolutionTree(answer, True)

        # check cache. 3,862,686 cache hits lol
        key = given_guess_id or (hash(possible_guesses.data.tobytes()),
                                 hash(possible_answers.data.tobytes()))
        if entry := memo.get(key):
            return entry

        best_tree = None
        best_guess_ids = [given_guess_id] if given_guess_id is not None \
            else self._best_guesses(possible_guesses, possible_answers, self.pillar_aware_heuristic, 20)

        for guess_id in best_guess_ids:
            tree = SolutionTree(guesses[guess_id])

            feedback_ids = np.bincount(self.guess_feedbacks_array[guess_id, possible_answers]).nonzero()[0]
            for feedback_id in feedback_ids:
                feedback = patterns[feedback_id]
                if feedback == '🟩🟩🟩🟩🟩':
                    tree.is_answer = True
                else:
                    next_possible_guesses = refine_possible_words(possible_guesses, self.guess_feedbacks_array,
                                                                  guess_id, feedback_id)
                    next_possible_answers = refine_possible_words(possible_answers, self.guess_feedbacks_array,
                                                                  guess_id, feedback_id)
                    tree[feedback] = self._map_solutions_optimal(memo, next_possible_guesses, next_possible_answers)

            if best_tree is None or tree.cmp_key < best_tree.cmp_key:
                best_tree = tree

        memo[key] = best_tree
        return best_tree

def best_starts():
    solver = WordleSolver().for_answer('')
    for start in 'SLATE', 'TARSE', 'LEAST', 'CRANE', 'SALET', 'LEAPT', 'STEAL':
        tree = solver.map_solutions(start, find_optimal=True)
        cnts = tree.answer_depth_distribution

        s = '{}: {} total, {:.3f} avg, {} worst, {} fails'.format(
            start,
            tree.total_guesses,
            tree.total_guesses / tree.answers_in_tree,
            tree.max_guess_depth,
            sum(cnts[d] for d in range(7, tree.max_guess_depth + 1))
        )
        print(s)

# best_starts()
# solver = WordleSolver('AGING', hard_mode=True, optimal=True)