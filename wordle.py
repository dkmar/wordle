from enum import Enum
from dataclasses import dataclass
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

def indicesOf(word: str, ch: str):
    for i, c in enumerate(word):
        if c == ch:
            yield i

class Guess:
    def __init__(self, guess: str, feedback: list[Status]):
        self.word = guess
        self.feedback = feedback

    def matchesWord(self, word: str) -> bool:
        used = 0
        for i, (g, fb, w) in enumerate(zip(self.word, self.feedback, word)):
            if fb == Status.Green:
                if g != w:
                    return False
                used |= (1 << i)

        for i, (g, fb, w) in enumerate(zip(self.word, self.feedback, word)):
            if fb == Status.Green:
                continue

            if g == w:
                return False

            if fb == Status.Yellow:
                partner = next((j for j in indicesOf(word, g) if not used&(1 << j)), None)
                if partner is None:
                    return False

                used |= (1 << partner)
            else:
                partner = next((j for j in indicesOf(word, g) if not used&(1 << j)), None)
                if partner is not None:
                    return False

        return True

wordset: set[str]
with open('data/allowed_words.txt', 'r') as f:
    words = map(str.strip, f)
    wordset = set(map(str.upper, words))

start = "CRANE"
answer = 'SPLAT'

print('POOCH', '\n', feedbackToStr(getFeedback('POOCH', 'TABOO')))
print('POOCH', '\n', feedbackToStr(getFeedback('POOCH', 'OTHER')))
print('SPLAT', '\n', feedbackToStr(getFeedback('SPLAT', 'SPLAT')))

info = Guess('_OO__', getFeedback('_OO__', 'TABOO'))
res = list(filter(info.matchesWord, wordset))
print(len(res))

# TODO add pytest and some cases
# TODO decide on scoring / evaluation
