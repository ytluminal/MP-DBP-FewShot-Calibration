#!/usr/bin/env python3
"""Verify curated release byte integrity and .py syntax (not numerical reproducibility)."""
from pathlib import Path
import ast, hashlib, json, sys
ROOT=Path(__file__).resolve().parents[1]
entries=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))['files']
bad=[]
for record in entries:
 p=ROOT/record['path']
 if not p.is_file():bad.append('MISSING '+record['path']);continue
 if hashlib.sha256(p.read_bytes()).hexdigest()!=record['sha256']:bad.append('SHA256 '+record['path'])
 if p.suffix=='.py':
  try:ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
  except (SyntaxError, UnicodeDecodeError) as exc:bad.append('PYTHON '+str(exc))
print(f'Checked {len(entries)} packaged files.')
if bad:
 print('FAILED:',*bad,sep='\n  ');sys.exit(1)
print('OK: saved files and Python source syntax. This is NOT a full scientific refit audit.')
