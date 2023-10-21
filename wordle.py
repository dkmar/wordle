from enum import Enum
from dataclasses import dataclass
''' Gameplay
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
- keep the top k candidates (for reporting)
repeat by playing a candidate
'''



class Status(Enum):
    Grey = 0
    Yellow = 1
    Green = 3

    def __str__(self):
        match self:
            case Status.Grey:
                return '⬛'
            case Status.Yellow:
                return '🟨'
            case Status.Green:
                return '🟩'
#
# @dataclass
# class Guess:
#     word: str
#     mask:

# '''
# a Pattern can be 5 concatenated bitsets
# where the color of each char is given by 00, 01, 11
# '''
# class Pattern:
#     def __init__(self):
#         self.value = 0


def getFeedback(guess: str, answer: str = 'SPLAT') -> list[Status]:
    """
    What should feedback look like?
    return indices of greens and yellows.

    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    # label greens
    feedback = [Status.Green if (ch == ans) else None
                for (ch, ans) in zip(guess, answer)]
    # label yellows
    for i, (ch, fb) in enumerate(zip(guess, feedback)):
        if fb == Status.Green:
            continue

        j = -1
        while (j := answer.find(ch, j+1)) != -1:
            if feedback[j] is None:
                feedback[i] = Status.Yellow
                # mark j as grey for now so that it isn't reused
                # it can later be made yellow as well.
                feedback[j] = Status.Grey
                break
    # convert remaining Nones to Grey
    return [fb if fb is not None else Status.Grey
            for fb in feedback]

def feedbackToStr(feedback: list[Status]) -> str:
    return ''.join(map(str, feedback))

# wordset = set()
# with open('data/allowed_words.txt', 'r') as f:
#     words = map(str.strip, f)
#     wordset.update(map(str.upper, words))

start = "CRANE"
answer = 'SPLAT'

print('POOCH', '\n', feedbackToStr(getFeedback('POOCH', 'TABOO')))
print('POOCH', '\n', feedbackToStr(getFeedback('POOCH', 'OTHER')))
print('SPLAT', '\n', feedbackToStr(getFeedback('SPLAT', 'SPLAT')))


