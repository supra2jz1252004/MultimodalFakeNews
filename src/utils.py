"""Reproducibility and file-integrity utilities."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def file_hash(path: Path) -> str:
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.part')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def log_stage(root: Path, title: str, text: str) -> None:
    stamp = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')
    with (root / 'docs/project_worklog.md').open('a', encoding='utf-8') as stream:
        stream.write(f'\n\n### {stamp} - {title}\n\n{text}\n')
