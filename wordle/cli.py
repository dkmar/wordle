import functools

import click
import itertools

import numpy as np

# import wordle.evaluation as evaluation
from wordle.lib import Pattern
from wordle.solver import Wordle, Game
# if __name__ == '__main__':
#     import wordle.evaluation as evaluation
#     from wordle.evaluation import GUESSES, guess_index, get_possible_words, best_guess, guess_feedbacks_array, refine_wordset
# from wordle.evaluation import GUESSES, guess_index, get_possible_words, best_guess, guess_feedbacks_array, \
#         refine_wordset

@click.group()
@click.version_option()
def cli():
    "Utility for solving and exploring wordle puzzles."


@cli.command()
@click.argument("answer", type=str, required=False)
@click.option("-h", "hard_mode", is_flag=True)
def play(answer: str | None, hard_mode: bool):
    "Play a game interactively."
    game = Game(answer, hard_mode=hard_mode)

    for round_number in itertools.count(1):
        remaining = len(game.possible_answers)
        click.echo(f'Round {round_number}')
        click.echo(f'# possible answers: {remaining}')

        for guess, entropy, parts, max_part_size, pillar_parts, freq, is_possible in game.top_guesses(20):
            possible_symbol = '✅' if is_possible else '❌'
            fb = game.grade_guess(guess) if answer else ''
            click.echo(f'\t{guess}: {entropy:.2f} {parts} {max_part_size} {pillar_parts} {freq:>10.2f} {possible_symbol} {fb}')

        click.echo()
        guess = click.prompt('Guess').upper()
        if answer is None:
            feedback = click.prompt('Feedback').upper()
            fb = Pattern.from_str(feedback)
            game.play(guess, fb)
            click.echo(guess)
            click.echo(fb)
        else:
            fb = game.play(guess)
            click.echo(guess)
            click.echo(fb)
        print()

        if fb == '🟩🟩🟩🟩🟩':
            print('----- SUCCESS ')
            for guess, feedback in game.history.items():
                print(guess, feedback)
            break
        # actual_entropy = evaluation.actual_info_from_guess(guess, fb, possible_words)
        # click.echo(f'{fb} {actual_entropy} Bits')

def solve(answer: str, starting_word: str, hard_mode: bool) -> dict[str]:
    game = Game(answer, hard_mode=hard_mode)
    feedback = game.play(starting_word)

    rounds = 1
    while feedback != '🟩🟩🟩🟩🟩' and rounds < 10:
        rounds += 1
        guess = game.best_guess()
        feedback = game.play(guess)

    return game.history


@cli.command()
@click.argument("n", type=int, required=False)
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-v", "verbose", is_flag=True)
@click.option("-h", "hard_mode", is_flag=True)
def bench(n: int | None, starting_word: str, verbose:  bool, hard_mode: bool):
    with open('wordle/data/wordlist_nyt20230701_hidden', 'r') as f:
        words = map(str.strip, f)
        REAL_ANSWER_SET = tuple(map(str.upper, words))


    answers = REAL_ANSWER_SET[:n] if n else REAL_ANSWER_SET
    N = len(answers)
    total_rounds_needed = 0
    count_failed = 0

    solve_game = functools.partial(solve, starting_word=starting_word.upper(), hard_mode=hard_mode)
    game_results = map(solve_game, answers)
    items = zip(range(1, N+1), answers, game_results)
    print_info = lambda item: f'[{item[0]}] {item[1]} {len(item[2])} {total_rounds_needed/item[0]:.2f}' if item else None

    with click.progressbar(items,
                           length=N,
                           item_show_func=print_info) as solution_info:
        for i, ans, result in solution_info:
            rnds = len(result)
            total_rounds_needed += rnds
            if rnds > 6:
                count_failed += 1
                print('\n', result[starting_word], ans, result, '\n')
            elif verbose:
                print()
                for guess, feedback in result.items():
                    print('  ', guess, feedback)


    avg = total_rounds_needed / N
    click.echo(f'Average: {avg}')
    click.echo(f'Failed: {count_failed}')

@cli.command()
# @click.argument("n", type=int, required=False)
# @click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
# @click.option("-v", "verbose", is_flag=True)
# def explore(n: int | None, starting_word: str, verbose:  bool):
@click.argument('answers', nargs=-1)
def explore(answers):
    # answer guesses...
    for answer in map(str.upper, answers):
        assert len(answer) == 5

        game_results = solve(answer, 'SLATE', hard_mode=True)

        offset = ''
        print(answer, len(game_results))
        for guess, feedback in game_results.items():
            print(offset, guess, feedback)
            # offset += ' '






@cli.command(name="command")
@click.argument(
    "example"
)
@click.option(
    "-o",
    "--option",
    help="An example option",
)
def first_command(example, option):
    "Command description goes here"
    click.echo("Here is some output")
