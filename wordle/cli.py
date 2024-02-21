import functools
import click
import itertools
import numpy as np
from wordle.lib import Pattern
from wordle.solver import WordleContext, WordleSolver


@click.group()
@click.version_option(package_name='wordle-solver')
def cli():
    "Utility for solving and exploring wordle puzzles."


@cli.command()
@click.argument("answer", type=str, required=False)
@click.option("-h", "hard_mode", is_flag=True)
@click.option("--debug", "debug", is_flag=True)
def play(answer: str | None, hard_mode: bool, debug: bool):
    """ Play a game interactively. """
    # if answer not in ANSWERS:
    #     click.echo('Given answer is not in answer set.')
    #     return

    ctx = WordleContext()
    solver = WordleSolver(ctx, hard_mode)
    game = solver.for_answer(answer.upper()).game if answer else solver.game

    if not debug:
        click.echo('Key:')
        click.echo('   #) Guess:  Expected Score | Expected Information | Possible Answer?\n')

    for round_number in itertools.count(1):
        remaining = len(game.possible_answers)
        # if remaining == 0:
        #     click.echo('Error: 0 possible answers.')
        #     break
        click.echo(f'Round {round_number}')
        click.echo(f'# possible answers: {remaining}')
        if debug:
            click.echo(f'Remaining Entropy: {np.log2(game.possible_answers.size):.2f}')
        click.echo('-'*24)

        for i, info in enumerate(solver.top_guesses_info(20), 1):
            (guess, score, entropy, parts, max_part_size, pillar_parts, freq, is_possible) = info
            possible_symbol = '✅' if is_possible else '❌'
            fb = solver.grade_guess(guess, answer) if answer else ''

            if debug:
                click.echo(
                    '  {0:>2d}) {1}: {2:.1f} {3:.2f} {4:>.2f} {5:>3d} {6:>2d} {7:>8.2f} {8} {9}'.format(
                        i, guess,
                        score,
                        entropy,
                        parts, max_part_size,
                        pillar_parts,
                        freq,
                        possible_symbol,
                        fb
                    )
                )
            else:
                click.echo(
                    '  {0:>2d}) {1}:  {2:.1f}  {3:.2f}  {4}'.format(
                        i, guess,
                        score,
                        entropy,
                        possible_symbol
                    )
                )

        click.echo()
        while (guess := click.prompt('Guess').upper()) not in ctx.word_index:
            click.echo('Invalid guess. Please try again.')

        if answer is None:
            while True:
                feedback = click.prompt('Feedback').upper()
                try:
                    fb = Pattern.from_str(feedback)
                    break
                except ValueError:
                    click.echo('Invalid feedback. Example: GGY_Y for 🟩🟩🟨⬛🟨')

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
    from wordle.solver import read_words_from_file, ALL_HIDDEN_ANSWERS_PATH, ORIGINAL_HIDDEN_ANSWERS_PATH, DATA_DIR
    answers_path = ORIGINAL_HIDDEN_ANSWERS_PATH if against_original_answers else ALL_HIDDEN_ANSWERS_PATH
    # answers_path = DATA_DIR / 'wordle-tools-answer-set-real.txt'
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
    # returns None when item is None
    print_info = lambda item: item and f'[{item[0]}] {item[1]} {len(item[2])} {(total_rounds_needed + min(len(item[2]), 7))/item[0]:.2f}'

    with click.progressbar(items,
                           length=N,
                           item_show_func=print_info) as solution_info:
        for i, ans, result in solution_info:
            rnds = len(result)
            # TrackVol said for accurate averages counting, we should count all failures as just 7.
            total_rounds_needed += min(rnds, 7)
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
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-o", "optimal", is_flag=True)
# @click.option("-v", "verbose", is_flag=True)
# def explore(n: int | None, starting_word: str, verbose:  bool):
@click.argument('answers', nargs=-1)
def explore(starting_word: str, optimal: bool, answers):
    solver = WordleSolver(
        WordleContext(),
        hard_mode=True
    )
    if optimal:
        click.echo('Building optimal tree... ', nl=False)
        solver = solver.with_optimal_tree(starting_word, read_cached=True)
        click.echo('Done.')

    # answer guesses...
    for answer in map(str.upper, answers):
        assert len(answer) == 5

        game_results = solve(answer, starting_word, solver)

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

    from wordle.solver import read_words_from_file, ORIGINAL_HIDDEN_ANSWERS_PATH
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
