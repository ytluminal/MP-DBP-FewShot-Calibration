"""Build the exact source JSON used by the frozen Line-2 rerun from hash-verified XLSX.
Uses artifact_tool to reproduce XLSX values without CSV conversion/rounding.
"""
from pathlib import Path
import json,hashlib
from artifact_tool import Blob,SpreadsheetFile
P=Path(__file__).resolve().parent
sources=[('Training(9).xlsx',995,'train55.json','de1e28d1525c97d802e5b3dd261dd7409bea3bbeade02875d3ac36fe76947fb9'),('Test(9).xlsx',249,'test55.json','75e5164aba4c974e54f4974dee86a65d37228e9194ada504f6133552c77fdfc8')]
for filename,n,out,sha in sources:
 path=P/filename
 assert hashlib.sha256(path.read_bytes()).hexdigest()==sha,(filename,'SHA-256 mismatch')
 ss=SpreadsheetFile.import_xlsx(Blob.load(str(path)))
 vals=ss.worksheets.get_item('Data').get_range(f'A1:BE{n+1}').values
 assert len(vals)==n+1 and len(vals[0])==57
 assert vals[0][22].startswith('DBP concentration') and vals[0][23]=='ref ID'
 (P/out).write_text(json.dumps({'headers':vals[0],'rows':vals[1:]},ensure_ascii=False))
 print(filename,'->',out,'rows',n)
