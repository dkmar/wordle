import functools

import click
import itertools

import numpy as np

# import wordle.evaluation as evaluation
from wordle.lib import Pattern
# from wordle.solver import Wordle, Game
from wordle.solverfinal import WordleContext, WordleSolver
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
    # if answer not in ANSWERS:
    #     click.echo('Given answer is not in answer set.')
    #     return

    solver = WordleSolver(hard_mode).for_answer(answer)
    game = solver.game

    for round_number in itertools.count(1):
        remaining = len(game.possible_answers)
        click.echo(f'Round {round_number}')
        click.echo(f'# possible answers: {remaining}')
        click.echo(f'Remaining Entropy: {np.log2(game.possible_answers.size)}')

        for i, (guess, score, entropy, parts, max_part_size, pillar_parts, freq, is_possible) in enumerate(solver.top_guesses_info(60), 1):
            possible_symbol = '✅' if is_possible else '❌'
            fb = solver.grade_guess(guess, answer) if answer else ''
            click.echo(f'\t({i}) {guess}: {score:.1f} {entropy:.2f} {parts:>.2f} {max_part_size:>3d} {pillar_parts:>2d} {freq:>10.2f} {possible_symbol} {fb}')

        click.echo()
        guess = click.prompt('Guess').upper()
        if answer is None:
            feedback = click.prompt('Feedback').upper()
            fb = Pattern.from_str(feedback)
            solver.play(guess, fb)
            click.echo(guess)
            click.echo(fb)
        else:
            fb = solver.play(guess)
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

def solve(answer: str, starting_word: str, solver: WordleSolver) -> dict[str]:
    solver.new_game(answer)
    game = solver.game
    feedback = solver.play(starting_word)

    rounds = 1
    while feedback != '🟩🟩🟩🟩🟩' and rounds < 10:
        rounds += 1
        guess = solver.best_guess()
        feedback = solver.play(guess)


    return game.history


@cli.command()
@click.argument("n", type=int, required=False, default=0)
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-v", "verbose", is_flag=True)
@click.option("-h", "hard_mode", is_flag=True)
@click.option("-o", "optimal", is_flag=True)
@click.option("--against-original-answers", "against_original_answers", is_flag=True)
def bench(n: int, starting_word: str, verbose:  bool, hard_mode: bool, optimal: bool, against_original_answers: bool):
    from wordle.solverfinal import read_words_from_file, ALL_HIDDEN_ANSWERS_PATH, ORIGINAL_HIDDEN_ANSWERS_PATH
    answers_path = ORIGINAL_HIDDEN_ANSWERS_PATH if against_original_answers else ALL_HIDDEN_ANSWERS_PATH
    answers = read_words_from_file(answers_path)
    if n:
        answers = answers[:n]

    N = len(answers)
    total_rounds_needed = 0
    count_failed = 0

    solver = WordleSolver(
        WordleContext(against_original_answers),
        hard_mode
    )

    if optimal:
        click.echo('Building optimal tree... ', nl=False)
        solver = solver.with_optimal_tree(starting_word)
        click.echo('Done.')

    solve_game = functools.partial(solve, solver=solver, starting_word=starting_word.upper())
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


@cli.command()
def leaderboard():
    solver = WordleSolver(
        WordleContext(using_original_answer_set=True),
        hard_mode=True
    ).with_optimal_tree(starting_word='SALET')
    tree = solver.solution_tree

    def find_path(answer: str) -> str:
        path = [tree.guess]
        curr = tree
        while curr.guess != answer:
            curr = curr[solver.grade_guess(curr.guess, answer)]
            path.append(curr.guess)

        return ','.join(path)

    from wordle.solverfinal import read_words_from_file, ORIGINAL_HIDDEN_ANSWERS_PATH
    original_answers = read_words_from_file(ORIGINAL_HIDDEN_ANSWERS_PATH)
    for ans in original_answers:
        click.echo(find_path(ans))


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
