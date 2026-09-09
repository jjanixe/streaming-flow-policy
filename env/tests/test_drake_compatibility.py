import subprocess
import sys


def test_drake_first_order_hold_imports_in_fresh_process():
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.metadata as m; "
                "from pydrake.trajectories import PiecewisePolynomial; "
                "print(m.version('drake')); "
                "print(m.version('numpy')); "
                "print(PiecewisePolynomial.__name__)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    drake_version, numpy_version, class_name = completed.stdout.splitlines()
    assert drake_version.startswith("1.26.")
    assert int(numpy_version.split(".", maxsplit=1)[0]) < 2
    assert class_name == "PiecewisePolynomial"
