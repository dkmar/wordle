import heapq

from wordle.solver import Evaluation, relevant_words
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
def get_score(word: str):
    score = Evaluation.score(relevant_words, word)
    return (score, word)

if __name__ == '__main__':
    # relevant_words = relevant_words[:500]

    evals = []
    with tqdm(total=len(relevant_words)) as progress:
        chunksize = math.ceil(len(relevant_words) / 4)
        with multiprocessing.Pool(4) as pool:
            for item in pool.imap_unordered(get_score, relevant_words, chunksize=chunksize):
                evals.append(item)
                progress.update(1)

    top = heapq.nlargest(30, evals)
    # top = Evaluation.topk(relevant_words, k=15)
    for score, word in top:
        print(word, score)


# for _ in range(4):
#     word = next(words)
#     score = Evaluation.score(relevant_words, word)
#     print(word, score)
#
# with ThreadPoolExecutor(4) as pool:
#     for _ in range(4):
#         pool.submit(work)

# def bench():
