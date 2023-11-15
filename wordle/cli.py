import click
import itertools
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

        for guess, entropy, parts, max_part_size, freq, is_possible in game.top_guesses():
            possible_symbol = '✅' if is_possible else '❌'
            click.echo(f'\t{guess}: {entropy:.2f} {parts} {max_part_size} {freq:>10.2f} {possible_symbol}')

        click.echo()
        guess = click.prompt('Guess').upper()
        if answer is None:
            feedback = click.prompt('Feedback').upper()

            # fb = solver.play(guess, feedback.upper())
            fb = Pattern.from_str(feedback)
            click.echo(guess)
            click.echo(fb)
            game.play(guess, fb)
        else:
            fb = game.play(guess)
            click.echo(guess)
            click.echo(fb)

        if fb == '🟩🟩🟩🟩🟩':
            break
        # actual_entropy = evaluation.actual_info_from_guess(guess, fb, possible_words)
        # click.echo(f'{fb} {actual_entropy} Bits')



@cli.command()
@click.argument("n", type=int, required=False)
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-v", "verbose", is_flag=True)
@click.option("-h", "hard_mode", is_flag=True)
def bench(n: int | None, starting_word: str, verbose:  bool, hard_mode: bool):
    starting_word = starting_word.upper()
    with open('wordle/data/wordlist_nyt20230701_hidden', 'r') as f:
        words = map(str.strip, f)
        REAL_ANSWER_SET = tuple(map(str.upper, words))

    def solve(answer: str) -> list[str]:
        game = Game(answer, hard_mode=hard_mode)
        feedback = game.play(starting_word)

        rounds = 1
        while feedback != '🟩🟩🟩🟩🟩' and rounds < 10:
            rounds += 1
            guess = game.best_guess()
            feedback = game.play(guess)

        return game.history

    answers = REAL_ANSWER_SET[:n] if n else REAL_ANSWER_SET
    N = len(answers)
    total_rounds_needed = 0
    count_failed = 0

    game_results = map(solve, answers)
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
                print('\n', result, '\n')
            elif verbose:
                print()
                for guess in result:
                    print('  ', guess)


    avg = total_rounds_needed / N
    click.echo(f'Average: {avg}')
    click.echo(f'Failed: {count_failed}')

@cli.command()
@click.argument("n", type=int, required=False)
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-v", "verbose", is_flag=True)
def explore(n: int | None, starting_word: str, verbose:  bool):
    # answer guesses...
    pass

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
