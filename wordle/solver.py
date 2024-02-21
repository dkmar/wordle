from collections.abc import Callable, Mapping, Iterable, Sequence
from typing import Optional, Any
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import wordle.feedback
import wordle.heuristics as heuristics
from wordle.feedback import get_guess_feedbacks_array
from wordle.lib import Pattern
from wordle.types import WordIndexArray, WordIndexType, FeedbackIndexType
from wordle.solutiontree import SolutionTree, PrioritizationPolicy
from wordle.utils import lexmax, filter_possible_words

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


def read_words_from_file(words_file: Path) -> tuple[str, ...]:
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


def get_index_for_words(words_file: Path, word_index: Mapping[str, int]) -> WordIndexArray:
    words = read_words_from_file(words_file)
    return np.array([word_index[word]
                     for word in words if word in word_index], dtype=WordIndexType)


@dataclass(slots=True)
class WordleContext:
    words: Sequence[str]
    word_index: Mapping[str, int]
    word_frequency: np.ndarray
    patterns: Sequence[str]
    pattern_index: Mapping[str, int]
    pillars_of_doom: WordIndexArray
    guess_feedbacks_array: np.ndarray
    using_original_answer_set: bool
    solve_policy: PrioritizationPolicy

    def __init__(self, using_original_answer_set: bool = False):
        self.words = read_words_from_file(ALLOWED_WORDS_PATH)
        self.word_index = {guess: i
                           for i, guess in enumerate(self.words)}
        self.word_frequency = get_word_frequencies(self.word_index)
        self.pillars_of_doom = get_index_for_words(PILLARS_OF_DOOM_PATH, self.word_index)
        self.patterns = Pattern.ALL_PATTERNS
        self.pattern_index = {pattern: i
                              for i, pattern in enumerate(self.patterns)}
        self.guess_feedbacks_array = get_guess_feedbacks_array(self.words, self.words,
                                                               self.pattern_index,
                                                               GUESS_FEEDBACKS_PATH)

        self.using_original_answer_set = using_original_answer_set
        self.solve_policy = PrioritizationPolicy.MINIMIZE_AVERAGE


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
    possible_guesses: WordIndexArray
    possible_answers: WordIndexArray
    history: dict[str, str]


class WordleSolver:
    def __init__(self, context: WordleContext, hard_mode: bool = False):
        self.context = context

        self.possible_guesses = np.arange(len(context.words), dtype=WordIndexType)
        self.possible_answers = get_index_for_words(
            ALL_HIDDEN_ANSWERS_PATH if not context.using_original_answer_set else ORIGINAL_HIDDEN_ANSWERS_PATH,
            # DATA_DIR / 'wordle-tools-answer-set-real.txt',
            context.word_index
        )

        self.game = Game('', self.possible_guesses, self.possible_answers, {})

        self.hard_mode = hard_mode
        self.optimal = False
        self.solution_tree: Optional[SolutionTreeView] = None
        self.grade_guess = wordle.feedback.grade_guess

    def new_game(self, answer: str):
        self.game = Game(answer, self.possible_guesses, self.possible_answers, {})

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
        self.solution_tree = SolutionTreeView(self.context, solution_tree)
        return self


    def play(self, guess: str, feedback: Optional[str] = None) -> str:
        game = self.game
        if feedback is None:
            assert len(game.answer) == 5
            feedback = self.grade_guess(guess, game.answer)

        context = self.context
        guess_id = context.word_index[guess]
        feedback_id = context.pattern_index[feedback]

        game.possible_answers = filter_possible_words(context.guess_feedbacks_array,
                                                      game.possible_answers, guess_id, feedback_id)
        if self.hard_mode:
            game.possible_guesses = filter_possible_words(context.guess_feedbacks_array,
                                                          game.possible_guesses, guess_id, feedback_id)

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
        keys = heuristics.basic_heuristic(self.context.guess_feedbacks_array,
                                          game.possible_guesses, game.possible_answers)
        i = lexmax(*keys)
        guess_id = game.possible_guesses[i]
        return self.context.words[guess_id]

    def top_guesses_info(self, k: int = 10) -> list:
        '''
        guess, "score", entropy, partitions, max partition size, pillar partitions, scaled word freq, can be answer?
        '''
        game = self.game
        ctx = self.context
        possible_guesses, possible_answers = game.possible_guesses, game.possible_answers

        # top_ids = self._best_guesses(possible_guesses, possible_answers, k=k)
        top_ids = self.best_guesses(possible_guesses, possible_answers,
                                    k=k, candidates_to_consider=2*k,
                                    force_basic_heuristic=True)

        if self.hard_mode:
            keys = heuristics.pillar_aware_heuristic(ctx.guess_feedbacks_array, ctx.pillars_of_doom,
                                                     top_ids, possible_answers)
        else:
            keys = heuristics.basic_heuristic(ctx.guess_feedbacks_array, top_ids, possible_answers)

        inds = np.lexsort(keys[::-1])[::-1]
        top_ids = top_ids[inds]

        ents = np.array(
            [heuristics.entropy(ctx.guess_feedbacks_array, guess_id, possible_answers) for guess_id in top_ids]
        )
        partitions, max_partition = np.array([
            heuristics.partitions_and_max(ctx.guess_feedbacks_array, guess_id, possible_answers)
            for guess_id in top_ids
        ]).T

        partitions = np.array([
            heuristics.partitions_level2(ctx.guess_feedbacks_array, guess_id, possible_guesses, possible_answers)
            for guess_id in top_ids
        ])

        can_be_answer = np.isin(top_ids, possible_answers, assume_unique=True)

        possible_pillars = np.intersect1d(possible_answers, ctx.pillars_of_doom, assume_unique=True)
        pillar_partitions = np.array([heuristics.partitions(ctx.guess_feedbacks_array, guess_id, possible_pillars)
                                      for guess_id in top_ids])
        exp_score = -heuristics.basic_heuristic2(ctx.guess_feedbacks_array, top_ids, possible_answers)[0]

        # rank = ents.argsort()[::-1].argsort()
        top_guesses = (ctx.words[gid] for gid in top_ids)
        word_freqs = ctx.word_frequency[top_ids]
        scaled_word_freqs = word_freqs / word_freqs.sum()
        # info = zip(top_guesses, keys[0][inds], ents, partitions, max_partition, pillar_partitions,
        #            scaled_word_freqs, can_be_answer)
        info = zip(top_guesses, exp_score, np.abs(ents), partitions, max_partition, pillar_partitions,
                   scaled_word_freqs, can_be_answer)
        return [guess_info for guess_info in info]


    def best_guesses(self,
                     possible_guesses: WordIndexArray,
                     possible_answers: WordIndexArray,
                     k: int = 30,
                     candidates_to_consider: int = 60,
                     force_basic_heuristic: bool = False) -> np.ndarray:
        # early exit if already < k
        if possible_answers.size == 1:
            return possible_answers
        if possible_guesses.size <= k:
            return possible_guesses

        ctx = self.context
        if force_basic_heuristic or not self.hard_mode:
            keys = heuristics.basic_heuristic(ctx.guess_feedbacks_array, possible_guesses, possible_answers)
            if k == 1:
                i = lexmax(*keys)
                return possible_guesses[[i]]
            else:
                key = np.fromiter(zip(*keys), dtype='f,b')
                inds = np.argpartition(key, -k)[-k:]
                return possible_guesses[inds]

        # hard mode:
        # we use a heuristic to get a pool of top candidates, then we look two levels deep to select for best
        keys = heuristics.pillar_aware_heuristic(ctx.guess_feedbacks_array, ctx.pillars_of_doom,
                                                 possible_guesses, possible_answers)
        if k == 1:
            i = lexmax(*keys)
            return possible_guesses[[i]]
        else:
            key = np.fromiter(zip(*keys), dtype='f,b')
            num_candidates = min(possible_guesses.size, candidates_to_consider)
            inds = np.argpartition(key, -num_candidates)[-num_candidates:]
            top_ids = possible_guesses[inds]

            deep_key = np.array([heuristics.partitions_level2(ctx.guess_feedbacks_array, guess_id,
                                                              possible_guesses, possible_answers)
                                 for guess_id in top_ids])

            k1, k2 = round(k / 2), (k // 2)
            pool1 = np.argpartition(key, -k1)[-k1:]
            pool2 = np.argpartition(deep_key, -k2)[-k2:]
            return np.union1d(possible_guesses[pool1], top_ids[pool2])

    def map_solutions(self, starting_word: str = '', find_optimal: bool = False) -> SolutionTree:
        possible_guesses = self.game.possible_guesses
        possible_answers = self.game.possible_answers
        ctx = self.context
        if starting_word:
            guess_id = ctx.word_index[starting_word]
            return SolutionTreeBuilder(ctx, self.best_guesses).build_subtree(
                guess_id, possible_guesses, possible_answers,
                len(self.game.history)
            )
        else:
            return SolutionTreeBuilder(ctx, self.best_guesses).map_solution_tree(
                possible_guesses, possible_answers,
                len(self.game.history)
            )


class SolutionTreeBuilder:
    def __init__(self, context: WordleContext, best_guesses_fn: Callable):
        self.context = context
        self.best_guesses = best_guesses_fn
        self.memo = {}
        self.answer_match = context.pattern_index['🟩🟩🟩🟩🟩']
        SolutionTree.set_cmp_policy(context.solve_policy)


    def build_subtree(self,
                      guess_id: int,
                      possible_guesses: WordIndexArray,
                      possible_answers: WordIndexArray,
                      level: int = 0) -> SolutionTree:

        context = self.context
        tree = SolutionTree(guess_id, level=level)
        feedback_ids = np.bincount(context.guess_feedbacks_array[guess_id, possible_answers]).nonzero()[0]
        answer_match_id = self.answer_match
        for feedback_id in feedback_ids:
            if feedback_id == answer_match_id:
                tree.is_answer = True
            else:
                next_possible_guesses = filter_possible_words(context.guess_feedbacks_array,
                                                              possible_guesses, guess_id, feedback_id)
                next_possible_answers = filter_possible_words(context.guess_feedbacks_array,
                                                              possible_answers, guess_id, feedback_id)
                tree[feedback_id] = self.map_solution_tree(next_possible_guesses,
                                                           next_possible_answers,
                                                           level + 1)

        return tree

    def map_solution_tree(self,
                          possible_guesses: WordIndexArray,
                          possible_answers: WordIndexArray,
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

        memo = self.memo
        # check cache. 3,862,686 cache hits lol
        key = (hash(possible_guesses.data.tobytes()), hash(possible_answers.data.tobytes()))
        if entry := memo.get(key):
            # cached entry for this config might have arrived here at a different (deeper?) depth
            if entry.level != level:
                entry.update_level(level)
            return entry

        best_tree = None
        best_guess_ids = self.best_guesses(possible_guesses, possible_answers, k=30, candidates_to_consider=60)
        for guess_id in best_guess_ids:
            tree = self.build_subtree(guess_id, possible_guesses, possible_answers, level)
            if best_tree is None or tree < best_tree:
                best_tree = tree

        memo[key] = best_tree
        return best_tree


class SolutionTreeView:
    def __init__(self, context: WordleContext, tree: SolutionTree):
        self.context = context
        self.tree = tree

    @property
    def guess(self):
        word_by_id = self.context.words
        return word_by_id[self.tree.guess_id]

    def __getitem__(self, item: str):
        pattern_index = self.context.pattern_index
        return self.__class__(self.context, self.tree[pattern_index[item]])


# def best_starts(write_file=False):
#     solver = WordleSolver(hard_mode=True, use_original_answer_list=True).for_answer('')
#     # for start in 'SLATE', 'TARSE', 'LEAST', 'CRANE', 'SALET', 'LEAPT', 'STEAL':
#     for start in ['SALET']:
#         tree = solver.map_solutions(start, find_optimal=True)
#         cnts = tree.answer_depth_distribution
#
#         s = '{}: {} total, {:.3f} avg, {} worst, {} fails'.format(
#             start,
#             tree.total_guesses,
#             tree.total_guesses / tree.answers_in_tree,
#             tree.max_guess_depth,
#             sum(cnts[d] for d in range(7, tree.max_guess_depth + 1))
#         )
#         print(s)
#         if write_file:
#             with open(f'tree_{start}_v5.txt', 'w') as out:
#                 out.write(str(tree))

