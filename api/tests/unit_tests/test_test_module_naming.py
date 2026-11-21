import ast
from pathlib import Path

_TESTS_ROOT = Path(__file__).resolve().parents[1]


def _defines_test_functions(path: Path) -> bool:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:
        return False
    return any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test_")
        for node in ast.walk(tree)
    )


def test_every_module_with_tests_is_collected_by_pytest():
    """Pytest only collects test_*.py, so tests in other modules silently never run."""
    uncollected = sorted(
        str(path.relative_to(_TESTS_ROOT))
        for path in _TESTS_ROOT.rglob("*.py")
        if not path.name.startswith("test_")
        and path.name != "conftest.py"
        and "__mock" not in path.parts
        and _defines_test_functions(path)
    )

    assert uncollected == [], f"Rename these to test_*.py so pytest runs them: {uncollected}"
