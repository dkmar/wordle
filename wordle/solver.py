import heapq, math
from tqdm import tqdm

from wordle.lib import Status, Pattern

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


def indicesOf(word: str, ch: str):
    for i, c in enumerate(word):
        if c == ch:
            yield i


class Feedback:
    def __init__(self, guess: str, pattern: Pattern):
        self.word = guess
        self.pattern = pattern

    def __repr__(self):
        return self.word + '\n' + str(self.pattern)

    def matchesWord(self, word: str) -> bool:
        used = 0
        for i, (g, fb, w) in enumerate(zip(self.word, self.pattern, word)):
            if fb == Status.Green:
                if g != w:
                    return False
                used |= (1 << i)

        for i, (g, fb, w) in enumerate(zip(self.word, self.pattern, word)):
            if fb == Status.Green:
                continue

            if g == w:
                return False

            if fb == Status.Yellow:
                partner = next((j for j in indicesOf(word, g) if not used & (1 << j)), None)
                if partner is None:
                    return False

                used |= (1 << partner)
            else:
                partner = next((j for j in indicesOf(word, g) if not used & (1 << j)), None)
                if partner is not None:
                    return False

        return True


class Game:
    def __init__(self, answer: str):
        self.answer = answer.upper()

    def grade_guess(self, guess: str) -> Feedback:
        """
        What should feedback look like?
        return indices of greens and yellows.

        POOCH TABOO
        _YY__

        POOCH OTHER
        _Y__Y
        """
        answer, guess = self.answer, guess.upper()
        feedback = Pattern()
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
            while (j := answer.find(ch, j + 1)) != -1:
                if not used & (1 << j):
                    feedback[i] = Status.Yellow
                    used |= (1 << j)
                    break
            else:
                feedback[i] = Status.Grey

        return Feedback(guess, feedback)


class Evaluation:
    @staticmethod
    def score(guess: str) -> float:
        total_remaining = len(relevant_words)
        expected_info = 0.0
        for pattern in map(Pattern.from_int, Pattern.all_patterns()):
            feedback = Feedback(guess, pattern)
            matches = 0
            for word in relevant_words:
                if feedback.matchesWord(word):
                    matches += 1

            if matches > 0:
                pattern_probability = matches / total_remaining
                information = -math.log2(pattern_probability)
                expected_info += pattern_probability * information

        return expected_info

    @staticmethod
    def topk(wordset: list[str], k: int = 10):
        evals = []
        for word in tqdm(relevant_words):
            s = Evaluation.score(word)
            evals.append((s, word))

        top = heapq.nlargest(k, evals)
        for s, w in top:
            print(f'{w}: {s}')

wordset: list[str]
with open('wordle/data/relevant_words.txt', 'r') as f:
    words = map(str.strip, f)
    wordset = list(map(str.upper, words))


p = Pattern([Status.Grey] * 5)
# p[0] = Status.Green
print(p, p.to_int(), Pattern.from_int(p.to_int()))
p = Pattern([Status.Green] * 5)
# p[-2] = Status.Yellow
print(p, p.to_int(), Pattern.from_int(p.to_int()))

start = "CRANE"
answer = 'SPLAT'
game = Game(answer)
startInfo = game.grade_guess(start)
print(startInfo)

# wordset = list(filter(startInfo.matchesWord, wordset))
relevant_words: list[str]
with open('wordle/data/relevant_words.txt', 'r') as f:
    words = map(str.strip, f)
    relevant_words = list(map(str.upper, words))

# print('BIOTA', Evaluation.score('BIOTA'))
# print('SPLAT', Evaluation.score('SPLAT'))
# relevant_words = list(filter(startInfo.matchesWord, relevant_words))
# nextInfo = game.grade_guess('TAILS')
# relevant_words = list(filter(nextInfo.matchesWord, relevant_words))
# evals = []
# for word in tqdm(relevant_words):
#     s = Evaluation.score(word)
#     evals.append((s, word))
#
# topk = heapq.nlargest(10, evals)
# for s, w in topk:
#     print(f'{w}: {s}')

# print(Game('TABOO').grade_guess('POOCH'))
# print(Game('OTHER').grade_guess('POOCH'))
# print(Game('SPLAT').grade_guess('SPLAT'))
#
# info = Game('TABOO').grade_guess('_OO__')
# res = list(filter(info.matchesWord, wordset))
# print(len(res))

# for code in Pattern.all_patterns():
#     pat = Pattern.from_int(code)
#     print(pat)


# TODO add pytest and some cases
# TODO decide on scoring / evaluation
# TODO should probably delineate between pattern and feedback (word, pattern)
