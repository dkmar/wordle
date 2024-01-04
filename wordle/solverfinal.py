import dataclasses
from dataclasses import dataclass
from collections.abc import Callable, Mapping, Iterable
from pathlib import Path
from typing import Optional

import numpy as np

import wordle.feedback
import wordle.heuristics as heuristics
from wordle.feedback import get_guess_feedbacks_array
from wordle.lib import Pattern
from wordle.solutiontree import SolutionTree
from wordle.utils import lexmax

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

'''
<datasets>
<Config>
- hard mode
- answer set
<Game>
<Solver>
'''

DATA_DIR = Path.cwd() / 'wordle' / 'data'
# original set of answers (prior to NYT acquisition) [2315]
# (this is commonly used for benchmarks)
ORIGINAL_HIDDEN_ANSWERS_PATH = DATA_DIR / 'original_hidden_answers.txt'
# all hidden answers past and present. [3171]
ALL_HIDDEN_ANSWERS_PATH = DATA_DIR / 'cat_hidden_answers.txt'
# all guesses allowed by wordle. [14855]
ALLOWED_WORDS_PATH = DATA_DIR / 'allowed_words.txt'
# curated set of "human" words allowed by wordle.
HUMAN_WORDS_PATH = DATA_DIR / 'relevant_words.txt'
# word frequencies from google ngrams (2019-)
WORD_FREQS_PATH = DATA_DIR / 'word_freqs_2019_valid.txt'
# pillars of doom. words matching patterns that trouble hard mode. (eg LIGHT, FIGHT, MIGHT, ...)
PILLARS_OF_DOOM_PATH = DATA_DIR / 'pillars_of_doom.txt'
# array of feedback patterns for (guess, answer) pairs
GUESS_FEEDBACKS_PATH = DATA_DIR / 'guess_feedbacks_array.npy'


def read_words_from_file(words_file: Path) -> tuple[str]:
    with open(words_file, 'r') as f:
        words = map(str.strip, f)
        return tuple(map(str.upper, words))


def get_word_frequencies(word_index: Mapping[str, int]) -> np.ndarray:
    with open(WORD_FREQS_PATH, 'r') as f:
        lines = map(str.strip, f)
        freqs = [0] * len(word_index)
        for word, count in map(str.split, lines):
            word = word.upper()
            if word in word_index:
                i = word_index[word]
                freqs[i] = int(count)

        return np.array(freqs)


def get_index_for_words(words_file: Path, word_index: Mapping[str, int]) -> np.ndarray[np.int16]:
    words = read_words_from_file(words_file)
    return np.array([word_index[word]
                     for word in words if word in word_index], dtype=np.int16)


@dataclass
class Game:
    """
    What defines a game?
    - The answer if we know it.
    - The set of all legal guesses
    - The set of all possible (probable, really) answers
    - The guesses we've made thus far and their associated feedbacks.
    """
    answer: str
    possible_guesses: np.ndarray[np.int16]
    possible_answers: np.ndarray[np.int16]
    history: dict[str, str]


class WordleSolver:
    def __init__(self,
                 hard_mode: bool = False,
                 use_original_answer_list: bool = False):
        self.words = tuple(read_words_from_file(ALLOWED_WORDS_PATH))
        self.word_index = {guess: i
                           for i, guess in enumerate(self.words)}
        self.word_freqs = get_word_frequencies(self.word_index)
        self.pillars_of_doom = get_index_for_words(PILLARS_OF_DOOM_PATH, self.word_index)

        self.patterns = Pattern.ALL_PATTERNS
        self.pattern_index = {pattern: i
                              for i, pattern in enumerate(self.patterns)}

        self.guess_feedbacks_array = get_guess_feedbacks_array(self.words, self.words, self.pattern_index,
                                                               GUESS_FEEDBACKS_PATH)
        self.possible_guesses = np.arange(len(self.words), dtype=np.int16)
        self.possible_answers = get_index_for_words(
            ALL_HIDDEN_ANSWERS_PATH if not use_original_answer_list else ORIGINAL_HIDDEN_ANSWERS_PATH,
            self.word_index
        )

        self.game = Game('', self.possible_guesses, self.possible_answers, {})

        self.hard_mode = hard_mode
        self.optimal = False
        self.solution_tree: Optional[SolutionTreeView] = None
        self.grade_guess = wordle.feedback.grade_guess

    def for_answer(self, answer: str):
        self.new_game(answer)
        return self

    def with_optimal_tree(self, starting_word: str, read_cached: bool = False):
        self.optimal = True

        # import pickle
        # path = Path(f'wordle/data/{starting_word}_tree.pickle')
        # if read_cached and path.exists():
        #     with open(path, 'rb') as f:
        #         self.solution_tree = pickle.load(f)
        # else:
        #     self.solution_tree = self.map_solutions(starting_word, find_optimal=True)
        #     with open(path, 'wb') as f:
        #         pickle.dump(self.solution_tree, f)
        #
        # return self
        solution_tree = self.map_solutions(starting_word, find_optimal=True)
        self.solution_tree = SolutionTreeView(self, solution_tree)
        return self

    def new_game(self, answer: str):
        self.game = Game(answer, self.possible_guesses, self.possible_answers, {})

    def filter_words(self, possible_words: np.ndarray[np.int16],
                     guess_id: int, feedback_id: int) -> np.ndarray[np.int16]:
        # current subset of answers
        subset = self.guess_feedbacks_array[guess_id, possible_words]
        next_possible_words = possible_words[subset == feedback_id]
        return next_possible_words

    def play(self, guess: str, feedback: Optional[str] = None) -> str:
        game = self.game
        if feedback is None:
            assert len(game.answer) == 5
            feedback = self.grade_guess(guess, game.answer)

        guess_id = self.word_index[guess]
        feedback_id = self.pattern_index[feedback]

        game.possible_answers = self.filter_words(game.possible_answers, guess_id, feedback_id)
        if self.hard_mode:
            game.possible_guesses = self.filter_words(game.possible_guesses, guess_id, feedback_id)

        game.history[guess] = feedback

        return feedback

    def best_guess(self) -> str:
        if self.optimal:
            tree = self.solution_tree
            history = self.game.history
            curr = tree
            for guess, feedback in history.items():
                curr = curr[feedback]

            return curr.guess

        game = self.game
        keys = heuristics.basic_heuristic(self.guess_feedbacks_array, game.possible_guesses, game.possible_answers)
        i = lexmax(*keys)
        # print('\n', -keys[0][i], keys[1][i], '\n')
        guess_id = game.possible_guesses[i]
        # guess_id = self.best_guesses(game.possible_guesses, game.possible_answers, k=1)[0]

        return self.words[guess_id]

    def top_guesses_info(self, k: int = 10) -> list:
        '''
        guess, "score", entropy, partitions, max partition size, pillar partitions, scaled word freq, can be answer?
        '''
        game = self.game
        possible_guesses, possible_answers = game.possible_guesses, game.possible_answers

        top_ids = self._best_guesses(possible_guesses, possible_answers, k=k)
        # top_ids = self.best_guesses(possible_guesses, possible_answers, k=k)

        if self.hard_mode:
            keys = heuristics.pillar_aware_heuristic(self.guess_feedbacks_array, self.pillars_of_doom,
                                                     top_ids, possible_answers)
        else:
            keys = heuristics.basic_heuristic(self.guess_feedbacks_array, top_ids, possible_answers)

        inds = np.lexsort(keys[::-1])[::-1]
        top_ids = top_ids[inds]

        ents = np.array([heuristics.entropy(self.guess_feedbacks_array, guess_id, possible_answers) for guess_id in top_ids])
        # ents = np.array([heuristics.entropy_level2(self.guess_feedbacks_array,
        #                                            guess_id, possible_guesses, possible_answers)
        #                  for guess_id in top_ids])
        partitions, max_partition = np.array([heuristics.partitions_and_max(self.guess_feedbacks_array,
                                                                            guess_id, possible_answers)
                                              for guess_id in top_ids]).T

        partitions = np.array([heuristics.partitions_level2(self.guess_feedbacks_array,
                                                            guess_id, possible_guesses, possible_answers)
                               for guess_id in top_ids])

        can_be_answer = np.isin(top_ids, possible_answers, assume_unique=True)

        possible_pillars = np.intersect1d(possible_answers, self.pillars_of_doom, assume_unique=True)
        pillar_partitions = np.array([heuristics.partitions(self.guess_feedbacks_array, guess_id, possible_pillars)
                                      for guess_id in top_ids])
        exp_score = -heuristics.basic_heuristic2(self.guess_feedbacks_array, top_ids, possible_answers)[0]

        # rank = ents.argsort()[::-1].argsort()
        top_guesses = (self.words[gid] for gid in top_ids)
        word_freqs = self.word_freqs[top_ids]
        scaled_word_freqs = word_freqs / word_freqs.sum()
        # info = zip(top_guesses, keys[0][inds], ents, partitions, max_partition, pillar_partitions,
        #            scaled_word_freqs, can_be_answer)
        info = zip(top_guesses, exp_score, ents, partitions, max_partition, pillar_partitions,
                   scaled_word_freqs, can_be_answer)
        return [guess_info for guess_info in info]

    def _best_guesses(self,
                      possible_guesses: np.ndarray[np.int16],
                      possible_answers: np.ndarray[np.int16],
                      k: int = 20
                      ) -> np.ndarray:
        if possible_answers.size == 1:
            return possible_answers

        if possible_guesses.size <= k:
            return possible_guesses

        if self.hard_mode:
            keys = heuristics.pillar_aware_heuristic(self.guess_feedbacks_array, self.pillars_of_doom,
                                                     possible_guesses, possible_answers)
        else:
            keys = heuristics.basic_heuristic(self.guess_feedbacks_array, possible_guesses, possible_answers)

        if k == 1:
            i = lexmax(*keys) if len(keys) > 1 else np.argmax(keys[0])
            return possible_guesses[[i]]

        key = np.fromiter(zip(*keys), dtype='f,b') if len(keys) > 1 else keys[0]
        inds = np.argpartition(key, -k)[-k:]
        return possible_guesses[inds]

    def __best_guesses_todo_delete(self,
                     possible_guesses: np.ndarray[np.int16],
                     possible_answers: np.ndarray[np.int16],
                     k: int = 20):
        if possible_answers.size == 1:
            return possible_answers

        if possible_guesses.size <= k:
            return possible_guesses

        keys = heuristics.pillar_aware_heuristic(self.guess_feedbacks_array, self.pillars_of_doom,
                                                 possible_guesses, possible_answers)
        key = np.fromiter(zip(*keys), dtype='f,b') if len(keys) > 1 else keys[0]

        kk = min(possible_guesses.size, 60)
        inds = np.argpartition(key, -kk)[-kk:]
        top = possible_guesses[inds]

        # deep_ents = np.array([self.entropy_level2(guess_id, possible_guesses, possible_answers)
        #                        for guess_id in top])
        deep_key = np.array([heuristics.partitions_level2(self.guess_feedbacks_array, guess_id,
                                                          possible_guesses, possible_answers)
                             for guess_id in top])

        if k == 1:
            i = np.argmax(deep_key)
            return top[i]

        a = possible_guesses[np.argpartition(key, -15)[-15:]]
        b = top[np.argpartition(deep_key, -15)[-15:]]
        return np.union1d(a, b)


    def best_guesses(self,
                     possible_guesses: np.ndarray[np.int16],
                     possible_answers: np.ndarray[np.int16],
                     k: int = 30,
                     candidates_to_consider: int = 60) -> np.ndarray:
        # early exit if already < k
        if possible_answers.size == 1:
            return possible_answers
        if possible_guesses.size <= k:
            return possible_guesses

        if not self.hard_mode:
            keys = heuristics.basic_heuristic(self.guess_feedbacks_array, possible_guesses, possible_answers)
            if k == 1:
                i = lexmax(*keys)
                return possible_guesses[[i]]
            else:
                key = np.fromiter(zip(*keys), dtype='f,b')
                inds = np.argpartition(key, -k)[-k:]
                return possible_guesses[inds]

        # hard mode:
        # we use a heuristic to get a pool of top candidates, then we look two levels deep to select for best
        keys = heuristics.pillar_aware_heuristic(self.guess_feedbacks_array, self.pillars_of_doom,
                                                 possible_guesses, possible_answers)
        if k == 1:
            i = lexmax(*keys)
            return possible_guesses[[i]]
        else:
            key = np.fromiter(zip(*keys), dtype='f,b')
            num_candidates = min(possible_guesses.size, candidates_to_consider)
            inds = np.argpartition(key, -num_candidates)[-num_candidates:]
            top_ids = possible_guesses[inds]

            deep_key = np.array([heuristics.partitions_level2(self.guess_feedbacks_array, guess_id,
                                                              possible_guesses, possible_answers)
                                 for guess_id in top_ids])

            k1, k2 = round(k / 2), (k // 2)
            pool1 = np.argpartition(key, -k1)[-k1:]
            pool2 = np.argpartition(deep_key, -k2)[-k2:]
            return np.union1d(possible_guesses[pool1], top_ids[pool2])

    def map_solutions(self, starting_word: str = '', find_optimal: bool = False) -> SolutionTree:
        possible_guesses = self.game.possible_guesses
        possible_answers = self.game.possible_answers
        if starting_word:
            guess_id = self.word_index[starting_word]
            return SolutionTreeBuilder(self).build_subtree(guess_id, possible_guesses, possible_answers,
                                                           len(self.game.history))
        else:
            return SolutionTreeBuilder(self).map_solution_tree(possible_guesses, possible_answers,
                                                               len(self.game.history))
    # def map_solutions(self, starting_word: str = '', find_optimal: bool = False) -> SolutionTree:
    #     possible_guesses = self.game.possible_guesses
    #     possible_answers = self.game.possible_answers
    #
    #     if starting_word:
    #         guess_id = self.word_index[starting_word]
    #         if find_optimal:
    #             return self._map_solutions_optimal({}, possible_guesses, possible_answers,
    #                                                guess_id, level=len(self.game.history))
    #         else:
    #             return self._map_solutions_greedy(possible_guesses, possible_answers, guess_id)
    #     elif find_optimal:
    #         return self._map_solutions_optimal({}, possible_guesses, possible_answers,
    #                                            level=len(self.game.history))
    #     else:
    #         return self._map_solutions_greedy(possible_guesses, possible_answers)
    #
    # def _map_solutions_greedy(self,
    #                           possible_guesses: np.ndarray[np.int16],
    #                           possible_answers: np.ndarray[np.int16],
    #                           given_guess_id: int | None = None) -> SolutionTree:
    #
    #     if possible_answers.size == 1:
    #         answer = self.words[possible_answers[0]]
    #         return SolutionTree(answer, True)
    #
    #     guess_id = given_guess_id if (given_guess_id is not None) \
    #         else self.best_guesses(possible_guesses, possible_answers, 1)
    #     tree = SolutionTree(self.words[guess_id])
    #
    #     feedback_ids = np.unique(self.guess_feedbacks_array[guess_id, possible_answers])
    #     for feedback_id in feedback_ids:
    #         feedback = self.patterns[feedback_id]
    #         if feedback == '🟩🟩🟩🟩🟩':
    #             tree.is_answer = True
    #         else:
    #             next_possible_guesses = self.filter_words(possible_guesses, guess_id, feedback_id)
    #             next_possible_answers = self.filter_words(possible_answers, guess_id, feedback_id)
    #             tree[feedback] = self._map_solutions_greedy(next_possible_guesses, next_possible_answers)
    #
    #     return tree
    #
    # def _map_solutions_optimal(self,
    #                            memo: dict,
    #                            possible_guesses: np.ndarray[np.int16],
    #                            possible_answers: np.ndarray[np.int16],
    #                            given_guess_id: int | None = None,
    #                            level: int = 0) -> SolutionTree:
    #
    #     if possible_answers.size == 1:
    #         answer = self.words[possible_answers[0]]
    #         return SolutionTree(answer, True, level)
    #
    #     # check cache. 3,862,686 cache hits lol
    #     key = given_guess_id or (hash(possible_guesses.data.tobytes()),
    #                              hash(possible_answers.data.tobytes()))
    #     if entry := memo.get(key):
    #         # cached entry for this config might have arrived here at a different (deeper?) depth
    #         entry.level = level
    #         return entry
    #
    #     best_tree = None
    #     best_guess_ids = [given_guess_id] if given_guess_id is not None \
    #         else self.best_guesses(possible_guesses, possible_answers, k=250, candidates_to_consider=500)
    #     # else self._best_guesses(possible_guesses, possible_answers, self.pillar_aware_heuristic, 20)
    #
    #     for guess_id in best_guess_ids:
    #         tree = SolutionTree(self.words[guess_id], level=level)
    #
    #         feedback_ids = np.bincount(self.guess_feedbacks_array[guess_id, possible_answers]).nonzero()[0]
    #         for feedback_id in feedback_ids:
    #             feedback = self.patterns[feedback_id]
    #             if feedback == '🟩🟩🟩🟩🟩':
    #                 tree.is_answer = True
    #             else:
    #                 next_possible_guesses = self.filter_words(possible_guesses, guess_id, feedback_id)
    #                 next_possible_answers = self.filter_words(possible_answers, guess_id, feedback_id)
    #                 tree[feedback] = self._map_solutions_optimal(memo, next_possible_guesses, next_possible_answers,
    #                                                              level=level + 1)
    #
    #         # if tree.guess == 'CRYER' and tree.level == 2 and tree.answers_in_tree > 20:
    #         #     print()
    #         if best_tree is None or tree.cmp_key < best_tree.cmp_key:
    #             # if tree.guess in ['MICRO', 'CRUMP'] and tree.answers_in_tree == 221:
    #             #     breakpoint()
    #             best_tree = tree
    #
    #     memo[key] = best_tree
    #     return best_tree
    #
    # def _map_solutions_optimal_ez(self,
    #                               memo: dict,
    #                               possible_guesses: np.ndarray[np.int16],
    #                               possible_answers: np.ndarray[np.int16],
    #                               given_guess_id: int | None = None,
    #                               level: int = 0) -> SolutionTree:
    #
    #     if possible_answers.size == 1:
    #         answer = self.words[possible_answers[0]]
    #         return SolutionTree(answer, True, level)
    #
    #     # check cache. 3,862,686 cache hits lol
    #     key = given_guess_id or hash(possible_answers.data.tobytes())
    #     if entry := memo.get(key):
    #         # cached entry for this config might have arrived here at a different (deeper?) depth
    #         entry.level = level
    #         return entry
    #
    #     best_tree = None
    #     best_guess_ids = [given_guess_id] if given_guess_id is not None \
    #                      else self.best_guesses(possible_guesses, possible_answers, k=50, candidates_to_consider=100)
    #     # else self._best_guesses(possible_guesses, possible_answers, self.pillar_aware_heuristic, 20)
    #
    #     for guess_id in best_guess_ids:
    #         tree = SolutionTree(self.words[guess_id], level=level)
    #
    #         feedback_ids = np.bincount(self.guess_feedbacks_array[guess_id, possible_answers]).nonzero()[0]
    #         for feedback_id in feedback_ids:
    #             feedback = self.patterns[feedback_id]
    #             if feedback == '🟩🟩🟩🟩🟩':
    #                 tree.is_answer = True
    #             else:
    #                 next_possible_guesses = possible_guesses
    #                 next_possible_answers = self.filter_words(possible_answers, guess_id, feedback_id)
    #                 tree[feedback] = self._map_solutions_optimal(memo, next_possible_guesses, next_possible_answers,
    #                                                              level=level + 1)
    #
    #         # if tree.guess == 'CRYER' and tree.level == 2 and tree.answers_in_tree > 20:
    #         #     print()
    #         if best_tree is None or tree.cmp_key < best_tree.cmp_key:
    #             # if tree.guess in ['MICRO', 'CRUMP'] and tree.answers_in_tree == 221:
    #             #     breakpoint()
    #             best_tree = tree
    #
    #     memo[key] = best_tree
    #     return best_tree

@dataclasses.dataclass(slots=True)
class SolutionTreeBuilder:
    solver: WordleSolver
    memo: dict
    answer_match: int

    def __init__(self, solver: WordleSolver):
        self.solver = solver
        self.memo = {}
        self.answer_match = solver.pattern_index['🟩🟩🟩🟩🟩']

    def build_subtree(self,
                      guess_id: int,
                      possible_guesses: np.ndarray[np.int16],
                      possible_answers: np.ndarray[np.int16],
                      level: int = 0) -> SolutionTree:

        solver = self.solver
        tree = SolutionTree(guess_id, level=level)
        feedback_ids = np.bincount(solver.guess_feedbacks_array[guess_id, possible_answers]).nonzero()[0]
        answer_match_id = self.answer_match
        for feedback_id in feedback_ids:
            if feedback_id == answer_match_id:
                tree.is_answer = True
            else:
                next_possible_guesses = solver.filter_words(possible_guesses, guess_id, feedback_id)
                next_possible_answers = solver.filter_words(possible_answers, guess_id, feedback_id)
                tree[feedback_id] = self.map_solution_tree(next_possible_guesses,
                                                           next_possible_answers,
                                                           level + 1)

        return tree

    def map_solution_tree(self,
                          possible_guesses: np.ndarray[np.int16],
                          possible_answers: np.ndarray[np.int16],
                          level: int = 0) -> SolutionTree:
        """
        Node depth = node level (root to node) + node height (node to deepest leaf)
        Args:
            guess_feedbacks_array:
            memo:
            possible_guesses:
            possible_answers:
            level:

        Returns:

        """
        # assert level <= 10
        if possible_answers.size == 1:
            return SolutionTree(possible_answers[0], True, level)

        solver, memo = self.solver, self.memo
        # check cache. 3,862,686 cache hits lol
        key = (hash(possible_guesses.data.tobytes()), hash(possible_answers.data.tobytes()))
        if entry := memo.get(key):
            # cached entry for this config might have arrived here at a different (deeper?) depth
            if entry.level != level:
                entry.update_level(level)
            return entry

        best_tree = None
        best_guess_ids = solver.best_guesses(possible_guesses, possible_answers, k=30, candidates_to_consider=60)
        for guess_id in best_guess_ids:
            tree = self.build_subtree(guess_id, possible_guesses, possible_answers, level)
            if best_tree is None or tree.cmp_key < best_tree.cmp_key:
                best_tree = tree

        memo[key] = best_tree
        return best_tree


class SolutionTreeView:
    def __init__(self, solver: WordleSolver, tree: SolutionTree):
        self.solver = solver
        self.tree = tree

    @property
    def guess(self):
        word_by_id = self.solver.words
        return word_by_id[self.tree.guess_id]

    def __getitem__(self, item):
        pattern_index = self.solver.pattern_index
        # if pattern_index[item] not in self.tree:
        #     print()
        return self.__class__(self.solver, self.tree[pattern_index[item]])


def best_starts(write_file=False):
    solver = WordleSolver(hard_mode=True, use_original_answer_list=True).for_answer('')
    # for start in 'SLATE', 'TARSE', 'LEAST', 'CRANE', 'SALET', 'LEAPT', 'STEAL':
    for start in ['SALET']:
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
        if write_file:
            with open(f'tree_{start}_v5.txt', 'w') as out:
                out.write(str(tree))

def print_entropy_remaining_guesses_data(starting_word = 'SLATE'):
    solver = WordleSolver(hard_mode=True, use_original_answer_list=False).with_optimal_tree(starting_word=starting_word)
    base = set(read_words_from_file(ORIGINAL_HIDDEN_ANSWERS_PATH))
    answers = set(read_words_from_file(ALL_HIDDEN_ANSWERS_PATH)) - base
    for answer in answers:
        solver.new_game(answer)
        remaining_entropy = []
        alt = []

        feedback = solver.play(starting_word)
        rounds = 1
        while feedback != '🟩🟩🟩🟩🟩':
            rent = np.log2(solver.game.possible_answers.size)
            remaining_entropy.append(rent)
            alt_rent = (rent - np.array([heuristics.entropy(solver.guess_feedbacks_array, gid, solver.game.possible_answers)
                           for gid in solver.game.possible_guesses])).mean()
            alt.append(alt_rent)

            rounds += 1
            guess = solver.best_guess()
            feedback = solver.play(guess)

        guesses_remaining = rounds - 1
        for rent, alt_rent in zip(remaining_entropy, alt):
            print(rent, guesses_remaining, alt_rent)
            guesses_remaining -= 1

def print_entropy_remaining_guesses_data_easy(starting_word = 'SLATE'):
    solver = WordleSolver(hard_mode=False, use_original_answer_list=False)
    # base = set(read_words_from_file(ORIGINAL_HIDDEN_ANSWERS_PATH))
    # answers = set(read_words_from_file(ALL_HIDDEN_ANSWERS_PATH)) - base
    answers = read_words_from_file(ALL_HIDDEN_ANSWERS_PATH)
    for answer in answers:
        solver.new_game(answer)
        remaining_entropy = []
        # alt = []

        feedback = solver.play(starting_word)
        rounds = 1
        while feedback != '🟩🟩🟩🟩🟩':
            rent = np.log2(solver.game.possible_answers.size)
            remaining_entropy.append(rent)
            # alt_rent = (rent - np.array([heuristics.entropy(solver.guess_feedbacks_array, gid, solver.game.possible_answers)
            #                for gid in solver.game.possible_guesses])).mean()
            # alt.append(alt_rent)

            rounds += 1
            guess = solver.best_guess()
            feedback = solver.play(guess)

        guesses_remaining = rounds - 1
        for rent in remaining_entropy:
            print(rent, guesses_remaining)
            guesses_remaining -= 1

# print_entropy_remaining_guesses_data_easy()
# best_starts()
# if __name__ == '__main__':
# best_starts()
# solver = WordleSolver(hard_mode=True).for_answer('BOXER')
# tree_salet = solver.map_solutions('SALET', find_optimal=True)
# solver.play('SALET')
# with open(f'tree_salet_NEW.txt', 'w') as out:
#         out.write(tree_salet.as_str(solver.words, solver.patterns))
# for guess in 'CRUMP', 'DRONY', 'MICRO':
#     tree = solver.map_solutions(guess, find_optimal=True)
#     cnts = tree.answer_depth_distribution
#
#     s = '{}: {} total, {:.3f} avg, {} worst, {} fails'.format(
#         guess,
#         tree.total_guesses,
#         tree.total_guesses / tree.answers_in_tree,
#         tree.max_guess_depth,
#         sum(cnts[d] for d in range(6, tree.max_guess_depth + 1))
#     )
#     print(s)
#     with open(f'tree_salet_{guess}.txt', 'w') as out:
#         out.write(str(tree))
# tree_crump = solver.map_solutions('CRUMP', find_optimal=True)
# tree_drony = solver.map_solutions('DRONY', find_optimal=True)
# tree_micro = solver.map_solutions('MICRO', find_optimal=True)
# print()
