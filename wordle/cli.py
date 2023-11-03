import click
from .solver import *


@click.group()
@click.version_option()
def cli():
    "Utility for solving and exploring wordle puzzles."


@cli.command()
def play():
    "Play a game interactively."
    solver = Solver(relevant_words)

    while not solver.solved():
        guess = click.prompt('Guess')
        feedback = click.prompt('Feedback')
        fb = solver.play(guess, feedback.upper())
        click.echo(fb)

        click.echo('\n'.join(
            f'\t{word}: {score}'
            for (word, score) in solver.suggestions()))

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
