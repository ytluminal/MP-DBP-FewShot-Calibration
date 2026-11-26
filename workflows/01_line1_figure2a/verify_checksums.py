"""Verify integrity of the archived replay files. Run: python verify_checksums.py"""
from pathlib import Path
import json,hashlib,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
bad=[]
for relative, expected in manifest['files'].items():
    p=root/relative
    if not p.is_file(): bad.append(('MISSING', relative)); continue
    actual=hashlib.sha256(p.read_bytes()).hexdigest()
    if actual != expected['sha256']: bad.append(('CHANGED',relative))
print('Checked:',len(manifest['files']),'files')
if bad:
    for kind,name in bad: print(kind,name)
    sys.exit(1)
print('OK: all saved payload files match SHA-256 checksums.')
