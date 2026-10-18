"""Check notebook validity, local links and historical evidence consistency."""
import ast
import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    for folder in ['src', 'scripts', 'tests']:
        for path in (ROOT / folder).glob('*.py'):
            ast.parse(path.read_text(), filename=str(path))
    path = ROOT / 'notebooks/plant_seedling_classification.ipynb'
    notebook = json.loads(path.read_text())
    try:
        import nbformat
    except ImportError:
        print('nbformat unavailable: JSON/cell checks only; run again after dependency install.')
    else:
        nbformat.validate(nbformat.read(path, as_version=4))
    for cell in notebook['cells']:
        assert cell['cell_type'] in {'markdown', 'code', 'raw'}
        if cell['cell_type'] == 'code':
            ast.parse(''.join(cell['source']))
    text = json.dumps(notebook)
    assert '/content/drive' not in text and 'colab.research.google.com/drive/' not in text
    assert not any(o.get('output_type') == 'error' for c in notebook['cells']
                   for o in c.get('outputs', []))
    with (ROOT / 'reports/model_comparison.csv').open() as stream:
        rows = list(csv.DictReader(stream))
    assert [float(r['test_accuracy']) for r in rows] == [0.6358, 0.7095]
    assert all(int(r['test_n']) == 475 for r in rows)
    with (ROOT / 'reports/classification_reports.csv').open() as stream:
        classes = list(csv.DictReader(stream))
    assert len(classes) == 24
    for model in ['baseline', 'augmented']:
        part = [r for r in classes if r['model'] == model]
        assert sum(int(r['support']) for r in part) == 475
        assert part[0]['species'] == 'Black-grass'
        assert part[10]['species'] == 'Small-flowered Cranesbill'
    for path in ROOT.rglob('*.md'):
        for link in re.findall(r'\]\(([^)]+)\)', path.read_text()):
            if '://' not in link and not link.startswith('#'):
                assert (path.parent / link.split('#')[0]).exists(), (path, link)
    print('Repository structure, notebook code, links and historical metrics: OK')


if __name__ == '__main__':
    main()
