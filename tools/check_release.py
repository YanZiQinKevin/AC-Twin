"""Check the code-only release boundary before committing or pushing."""
from __future__ import annotations
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
LIMIT = 1024 * 1024
ATTRIBUTION = re.compile(r'(?i)(?:generated|written|authored|created|produced)\s+(?:with|by|using)\s+(?:an?\s+)?(?:AI\b|LLM\b|GPT\b|ChatGPT\b|Claude\b|Codex\b)|(?:AI|大模型|人工智能|ChatGPT|Claude|Codex).{0,12}(?:生成|编写|创作)|(?:由|使用).{0,8}(?:AI|大模型|人工智能).{0,8}(?:生成|编写)|co-authored-by:.*(?:Claude|Codex|GPT|AI\b)')
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|AKIA[A-Z0-9]{16}|-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----)')

def allowed(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if rel.as_posix() in {'.gitignore', 'README.md', 'pytest.ini', 'conftest.py'}:
        return True
    if path.suffix in {'.py', '.ps1', '.sh', '.js'}:
        return rel.parts[0] in {'code', 'tests', 'tools', 'legacy', 'paper', 'experiments'} or len(rel.parts) == 1
    if path.suffix == '.json':
        return rel.parts[0] == 'config' or (rel.parts[0] == 'experiments' and (path.name == 'config.json' or path.name.endswith('_config.json')))
    return path.suffix == '.txt' and path.name.startswith('requirements')

def main() -> int:
    names = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard'], cwd=ROOT).decode('utf-8').split('\0')
    files = sorted({ROOT / name for name in names if name})
    problems = []
    total = 0
    largest = (0, '')
    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        if not path.is_file():
            problems.append(f'{rel}: tracked path missing')
            continue
        size = path.stat().st_size
        total += size
        largest = max(largest, (size, rel))
        if not allowed(path) or size >= LIMIT:
            problems.append(f'{rel}: outside code-only boundary or >= 1 MiB')
            continue
        try:
            content = path.read_text(encoding='utf-8-sig')
            # This scanner's own patterns are matching expressions, not author credits.
            if path != Path(__file__).resolve() and ATTRIBUTION.search(content):
                problems.append(f'{rel}: generation attribution found')
            if SECRET.search(content):
                problems.append(f'{rel}: credential pattern found')
            if path.suffix == '.py':
                ast.parse(content, filename=rel)
            if path.suffix == '.json':
                json.loads(content)
        except (UnicodeError, SyntaxError, ValueError) as exc:
            problems.append(f'{rel}: parse check failed ({type(exc).__name__})')
    print(json.dumps({'status': 'FAIL' if problems else 'PASS', 'files': len(files), 'bytes': total, 'largest_bytes': largest[0], 'largest_path': largest[1], 'problems': problems}, indent=2))
    return 1 if problems else 0

if __name__ == '__main__':
    raise SystemExit(main())
