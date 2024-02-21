'''
For hard mode, being greedy gets punished and we need to actually look at the tree.
'''
from collections import deque, Counter
from enum import Enum
from functools import cached_property
from wordle.types import WordIndexType, FeedbackIndexType
import numpy as np


class PrioritizationPolicy(Enum):
    # Minimize average # of guesses, even if it means losing some.
    MINIMIZE_AVERAGE = 1
    # Minimize number of losses, tie-break on average # of guesses
    NON_LOSING = 2


class SolutionTree(dict[int, 'SolutionTree']):
    """
    The purpose here is to enumerate the optimal guess tree like so:
        SALET1 [8124 g, 3.51 avg]
        - ⬛🟨⬛⬛🟩: AUDIT2 [48 g, 2.40 avg]
            - 🟨⬛⬛⬛🟩: CHANT3 [11 g, 1.83 avg]
                - ⬛⬛🟩⬛🟩: GRAFT4 [1 g, 1.00 avg]

    The SolutionTree looks like this conceptually:
        Tree<SALET>[⬛🟨⬛⬛🟩] -> Tree<AUDIT>

    A SolutionTree is the tree of ~optimal~ guess paths from the current guess.
    It has a mapping from possible guess feedbacks to the next tree.

    Each tree can compute (and cache) important metadata on # of guesses and worst case.
    """
    def __init__(self, guess_id: int, is_answer: bool = False, level: int = 0):
        super().__init__()
        self.guess_id = WordIndexType(guess_id)
        self.is_answer = np.bool_(is_answer)
        self.level = np.int8(level)

    @property
    def answer_depths(self) -> dict:
        q = deque([self])
        answer_depth = {}
        depth = 0
        while q:
            depth += 1
            for _ in range(len(q)):
                tree = q.popleft()
                if tree.is_answer:
                    answer_depth[tree.guess_id] = depth
                q.extend(tree.values())

        return answer_depth

    @property
    def answer_depth_distribution(self) -> Counter[int]:
        return Counter(self.answer_depths.values())

    @cached_property
    def answers_in_tree(self):
        cnt = int(self.is_answer)
        for subtree in self.values():
            cnt += subtree.answers_in_tree

        return cnt

    @cached_property
    def total_guesses(self) -> int:
        total = int(self.is_answer)
        for subtree in self.values():
            total += subtree.answers_in_tree + subtree.total_guesses

        return total

    @cached_property
    def max_guess_depth(self) -> int:
        # max_subtree_depth = max((subtree.max_guess_depth for subtree in self.values()), default=0)
        # return max_subtree_depth or self.depth
        best = max((1 + subtree.max_guess_depth for subtree in self.values()), default=0)
        return best or int(self.is_answer)

    def update_level(self, new_level: int) -> None:
        self.level = np.int8(new_level)


    def cmp_key_min_avg(self):
        # avg = (self.total_guesses / self.answers_in_tree) if self.answers_in_tree else 4
        # return avg, self.level + self.max_guess_depth, not self.is_answer
        return self.total_guesses, self.level + self.max_guess_depth, not self.is_answer

    def cmp_key_non_losing(self):
        return not (self.level + self.max_guess_depth <= 6), self.total_guesses, not self.is_answer

    @classmethod
    def set_cmp_policy(cls, policy: PrioritizationPolicy):
        def minimize_average_policy(tree1: SolutionTree, tree2: SolutionTree):
            return tree1.cmp_key_min_avg() < tree2.cmp_key_min_avg()

        def non_losing_policy(tree1: SolutionTree, tree2: SolutionTree):
            return tree1.cmp_key_non_losing() < tree2.cmp_key_non_losing()

        match policy:
            case PrioritizationPolicy.MINIMIZE_AVERAGE:
                cls.__lt__ = minimize_average_policy
            case PrioritizationPolicy.NON_LOSING:
                cls.__lt__ = non_losing_policy

    def __str__(self) -> str:
        # deprecate?
        # guess and number of guesses
        base = f'{self.guess_id}{self.level + 1} [{self.total_guesses} g, {self.total_guesses / self.answers_in_tree:.2f} avg]'
        sb = [base]
        # if not self.is_answer:
        #     text += " (not an answer)"
        for key, val in self.items():
            lines = "\n    ".join(str(val).splitlines())
            sb.append(f"\n    - {key}: {lines}")
        return ''.join(sb)

    def as_str(self, word_by_id: tuple, pattern_by_id: tuple) -> str:
        # guess and number of guesses
        base = f'{word_by_id[self.guess_id]}{self.level + 1} [{self.total_guesses} g, {self.total_guesses / self.answers_in_tree:.2f} avg]'
        sb = [base]
        # if not self.is_answer:
        #     text += " (not an answer)"
        for key, val in self.items():
            lines = "\n    ".join(val.as_str(word_by_id, pattern_by_id).splitlines())
            sb.append(f"\n    - {pattern_by_id[key]}: {lines}")
        return ''.join(sb)
