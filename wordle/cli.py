import click
import itertools
# from .solver import *
import wordle.evaluation as evaluation
from .solver import Pattern, Feedback


@click.group()
@click.version_option()
def cli():
    "Utility for solving and exploring wordle puzzles."


@cli.command()
def play():
    "Play a game interactively."
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
        fb = str(Pattern.from_str(feedback))
        actual_entropy = evaluation.actual_info_from_guess(guess, fb, possible_words)
        click.echo(guess)
        click.echo(f'{fb} {actual_entropy} Bits')

        possible_words = evaluation.refine_possible_words(possible_words, guess, fb)

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
