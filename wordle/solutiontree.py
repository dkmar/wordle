

'''
For hard mode, being greedy gets punished and we need to actually look at the tree.
'''
from collections import deque, Counter
from functools import cached_property
# class SolutionTree(dict[str, 'SolutionTree']):
# class SolutionTree(UserDict):
class SolutionTree(dict[str, 'SolutionTree']):
    def __init__(self, guess: str, is_answer: bool = False):
        super().__init__()
        self.guess = guess
        self.is_answer = is_answer

    @property
    def answer_depths(self) -> dict[str, int]:
        q = deque([self])
        answer_depth = {}
        depth = 0
        while q:
            depth += 1
            for _ in range(len(q)):
                tree = q.popleft()
                if tree.is_answer:
                    answer_depth[tree.guess] = depth
                q.extend(tree.values())

        return answer_depth

    @property
    def answer_depth_distribution(self) -> Counter[int]:
        return Counter(self.answer_depths.values())

    @cached_property
    def answers_in_tree(self):
        cnt = int(self.is_answer)
        return cnt + sum(subtree.answers_in_tree for subtree in self.values())

    @cached_property
    def total_guesses(self) -> int:
        total = int(self.is_answer)
        for subtree in self.values():
            total += subtree.answers_in_tree + subtree.total_guesses

        return total

    @cached_property
    def max_guess_depth(self) -> int:
        best = max((1 + subtree.max_guess_depth for subtree in self.values()), default=0)
        return best or int(self.is_answer)

    @cached_property
    def cmp_key(self) -> tuple[int, int, bool]:
        # avg depth, # failing? or max depth?, is_answer
        # dd = self.answer_depth_distribution
        # total_guesses = sum(depth * count
        #                     for depth, count in dd.items())
        # max_depth = max(dd.keys())
        # return (total_guesses, max_depth, not self.is_answer)
        return (self.total_guesses, self.max_guess_depth, not self.is_answer)




    def __str__(self) -> str:
        # guess and number of buckets
        base = f'{self.guess} [{len(self)}]'
        sb = [base]
        # if not self.is_answer:
        #     text += " (not an answer)"
        for key, val in self.items():
            lines = "\n    ".join(str(val).splitlines())
            sb.append(f"\n    - {key}: {lines}")
        return ''.join(sb)

