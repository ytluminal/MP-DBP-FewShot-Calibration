"""Verify integrity of this package (SHA-256 manifest) and archived 80/80 predictions."""
import hashlib,json,subprocess,sys
from pathlib import Path
root=Path(__file__).resolve().parent
manifest=json.loads((root/'PACKAGE_SHA256.json').read_text(encoding='utf-8'))
for name,expected in manifest['files'].items():
 p=root/name
 if not p.is_file():raise SystemExit('MISSING '+name)
 actual=hashlib.sha256(p.read_bytes()).hexdigest()
 if actual!=expected:raise SystemExit('CHECKSUM MISMATCH '+name)
print('PASS: SHA256 checks:',len(manifest['files']),flush=True)
subprocess.run([sys.executable,'verify_all.py'],cwd=root,check=True)
print('PASS: archived Figure 2c all 80 model configurations are consistent.',flush=True)
