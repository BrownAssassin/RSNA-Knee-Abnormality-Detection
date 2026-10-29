"""Inspect public label schemas on Kaggle, never export report text."""
import json
from pathlib import Path
import pandas as pd
from .contracts import sha256, write_json


def main():
    records = []
    for folder in sorted(Path('/kaggle/input').glob('*')) + sorted(Path('/kaggle/input').glob('datasets/*/*')):
        if not folder.is_dir() or folder.name == 'rsna-knee-abnormality-detection':
            continue
        for path in sorted(folder.glob('*.csv')):
            table = pd.read_csv(path, nrows=3)
            # Schema only: no cell text, reports or individual study labels printed.
            records.append({'file': str(path), 'sha256': sha256(path), 'columns': table.columns.tolist(),
                            'types': {c: str(table[c].dtype) for c in table}})
        for path in folder.glob('COLUMN_GUIDE.json'):
            records.append({'guide': str(path), 'content': json.loads(path.read_text())})
    write_json('/kaggle/working/label_schema_audit.json', records)
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
