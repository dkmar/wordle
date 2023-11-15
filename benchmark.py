import heapq

from wordle.evaluation import *
# from concurrent.futures import ProcessPoolExecutor, as_completed
import itertools, timeit, os, math, functools
import multiprocessing
from tqdm import tqdm

# words = itertools.chain(('CRANE', 'TRACE', 'RAISE'), relevant_words)
#
#
# def work():
#     word = next(words)
#     score = Evaluation.score(relevant_words, word)
#     # print(word, score)
#
#
# t = timeit.Timer(work)
# print(t.timeit(4))



# for _ in range(4):
#     word = next(words)
#     score = Evaluation.score(relevant_words, word)
#     print(word, score)
#
# with ThreadPoolExecutor(4) as pool:
#     for _ in range(4):
#         pool.submit(work)

with open('wordle/data/wordle-nyt-answers-alphabetical.txt', 'r') as f:
    words = map(str.strip, f)
    REAL_ANSWER_SET = tuple(map(str.upper, words))

def solve(answer_id: int):
    possible_words = get_possible_words()
    rounds = 0
    while possible_words.size > 1:
        guess_id = best_guess(possible_words)
        feedback_id = guess_feedbacks_array[guess_id, answer_id]
        possible_words = refine_wordset(possible_words, guess_id, feedback_id)
        rounds += 1

    return rounds

# def solve_all():
#     answers_ids = (guess_index[answer] for answer in REAL_ANSWER_SET)
#     results = tqdm(map(solve, answers_ids))
#     return np.fromiter(results, int).mean()
#
# if __name__ == '__main__':
#     # answer_id = guess_index['FLARE']
#     # rnds = solve(answer_id)
#     # print(rnds)
#     res = solve_all()
#     print(res)