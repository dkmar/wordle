

'''
For hard mode, being greedy gets punished and we need to actually look at the tree.
'''
from collections import deque, Counter
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

