import click
import itertools
# import wordle.evaluation as evaluation
from wordle.lib import Pattern
from wordle.solver import Wordle
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
def play():
    "Play a game interactively."
    import wordle.evaluation as evaluation
    # solver = Solver(relevant_words)
    possible_words = evaluation.get_possible_words()

    for round_number in itertools.count(1):
        click.echo(f'Round {round_number}')
        click.echo(f'# possible answers: {len(possible_words)}')
        for guess, score in evaluation.best_guesses(possible_words):
            click.echo(f'\t{guess}: {score}')

        click.echo()
        guess = click.prompt('Guess').upper()
        feedback = click.prompt('Feedback').upper()

        # fb = solver.play(guess, feedback.upper())
        fb = Pattern.from_str(feedback)
        actual_entropy = evaluation.actual_info_from_guess(guess, fb, possible_words)
        click.echo(guess)
        click.echo(f'{fb} {actual_entropy} Bits')

        possible_words = evaluation.refine_possible_words(possible_words, guess, fb)

@cli.command()
@click.argument("n", type=int, required=False)
@click.option("-w", "--w", "starting_word", default='SLATE', help='Set a starting word.')
@click.option("-v", "verbose", is_flag=True)
def bench(n: int | None, starting_word: str, verbose:  bool):
    w = Wordle()
    with open('wordle/data/wordle-nyt-answers-alphabetical.txt', 'r') as f:
        words = map(str.strip, f)
        REAL_ANSWER_SET = tuple(map(str.upper, words))

    def solve(answer_id: int):
        # if verbose:
        #     print('\n', GUESSES[answer_id])
        possible_words = w.get_possible_words()
        guess_id = w.word_index[starting_word]
        if verbose:
            print('\n  ', w.guesses[guess_id])
        feedback_id = w.guess_feedbacks_array[guess_id, answer_id]

        rounds = 1
        while Pattern.ALL_PATTERNS[feedback_id] != '🟩🟩🟩🟩🟩':
            possible_words = w.refine_wordset(possible_words, guess_id, feedback_id)

            rounds += 1
            guess_id = w.best_guess(possible_words)
            if verbose:
                print('  ', w.guesses[guess_id])
            feedback_id = w.guess_feedbacks_array[guess_id, answer_id]

        return rounds

    answers = REAL_ANSWER_SET[:n] if n else REAL_ANSWER_SET
    N = len(answers)
    total_rounds_needed = 0

    answers_ids = (w.word_index[answer] for answer in answers)
    rounds_needed = map(solve, answers_ids)
    items = zip(range(1, N+1), answers, rounds_needed)
    print_info = lambda item: f'[{item[0]}] {item[1]} {item[2]} {total_rounds_needed/item[0]}' if item else None

    with click.progressbar(items,
                           length=N,
                           item_show_func=print_info) as solution_info:
        for i, ans, rnds in solution_info:
            total_rounds_needed += rnds

    avg = total_rounds_needed / N
    click.echo(f'Average: {avg}')

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
