"""Remove saídas e contadores de execução dos notebooks antes da versionagem."""
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def clear_outputs() -> None:
    for path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        notebook = nbformat.read(path, as_version=4)
        for cell in notebook.cells:
            if cell.cell_type == "code":
                cell.outputs = []
                cell.execution_count = None
        nbformat.write(notebook, path)
        print(path.name)


if __name__ == "__main__":
    clear_outputs()
