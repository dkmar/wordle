'''
For hard mode, being greedy gets punished and we need to actually look at the tree.
'''
from collections import deque, Counter
from functools import cached_property
import numpy as np

# class SolutionTree(dict[str, 'SolutionTree']):
# class SolutionTree(UserDict):
class SolutionTree(dict[str, 'SolutionTree']):
    def __init__(self, guess_id: int, is_answer: bool = False, level: int = 0):
        super().__init__()
        self.guess_id = np.int16(guess_id)
        self.is_answer = np.bool_(is_answer)
        self.level = np.int8(level)
        self.cmp_fn = self.cmp_key_non_losing

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

    @property
    def cmp_key(self):
        return self.cmp_fn()

    def cmp_key_min_avg(self):
        # avg = (self.total_guesses / self.answers_in_tree) if self.answers_in_tree else 4
        # return avg, self.level + self.max_guess_depth, not self.is_answer
        return self.total_guesses, self.level + self.max_guess_depth, not self.is_answer

    def cmp_key_non_losing(self):
        return not (self.level + self.max_guess_depth <= 6), self.total_guesses, not self.is_answer

    def __str__(self) -> str:
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


