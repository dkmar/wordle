from enum import Enum
from dataclasses import dataclass
import functools, heapq

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

class Status(Enum):
    Grey = 0
    Yellow = 1
    Green = 3

    def __repr__(self):
        match self:
            case Status.Grey:
                return '⬛'
            case Status.Yellow:
                return '🟨'
            case Status.Green:
                return '🟩'


def getFeedback(guess: str, answer: str = 'SPLAT') -> list[Status]:
    """
    What should feedback look like?
    return indices of greens and yellows.

    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    feedback = [None] * 5
    used = 0
    # label greens
    for i, (ch, ans) in enumerate(zip(guess, answer)):
        if ch == ans:
            feedback[i] = Status.Green
            used |= (1 << i)

    # label yellows
    for i, (ch, fb) in enumerate(zip(guess, feedback)):
        if fb == Status.Green:
            continue

        # TODO: see if it even makes sense to usr find() instead of a generator like we do in matchesWord
        j = -1
        while (j := answer.find(ch, j+1)) != -1:
            if not used & (1 << j):
                feedback[i] = Status.Yellow
                used |= (1 << j)
                break
        else:
            feedback[i] = Status.Grey

    return feedback

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

def score(word: str) -> int:
    info = Guess(word, getFeedback(word, answer))
    res = list(filter(info.matchesWord, wordset))
    return len(res)

wordset: set[str]
with open('data/allowed_words.txt', 'r') as f:
    words = map(str.strip, f)
    wordset = set(map(str.upper, words))

start = "CRANE"
answer = 'SPLAT'
startInfo = Guess(start, getFeedback(start, answer))
# print(startInfo.feedback)
# wordset = set(filter(startInfo.matchesWord, wordset))
# print(score('BIOTA'))
# evals = []
# for word in wordset:
#     s = score(word)
#     evals.append((s, word))
#
# topk = heapq.nsmallest(10, evals)
# for s, w in topk:
#     print(f'{w}: {s}')

print('POOCH', '\n', getFeedback('POOCH', 'TABOO'))
print('POOCH', '\n', getFeedback('POOCH', 'OTHER'))
print('SPLAT', '\n', getFeedback('SPLAT', 'SPLAT'))

info = Guess('_OO__', getFeedback('_OO__', 'TABOO'))
res = list(filter(info.matchesWord, wordset))
print(len(res))

# TODO add pytest and some cases
# TODO decide on scoring / evaluation
