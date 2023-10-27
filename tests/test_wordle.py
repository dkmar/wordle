from click.testing import CliRunner
from wordle.cli import cli
from wordle.solver import *

def test_version():
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["--version"])
        assert result.exit_code == 0
        assert result.output.startswith("cli, version ")

def test_patterns():
    """
    POOCH TABOO
    _YY__

    POOCH OTHER
    _Y__Y
    """
    fb = Game('TABOO').grade_guess('POOCH')
    assert fb.pattern == Pattern.from_str('_YY__')
    fb = Game('OTHER').grade_guess('POOCH')
    assert fb.pattern == Pattern.from_str('_Y__Y')