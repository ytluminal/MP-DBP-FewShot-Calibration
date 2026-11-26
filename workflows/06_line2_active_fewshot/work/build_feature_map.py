import json
from pathlib import Path
from docx import Document
p=Path(__file__).resolve().parent
raw=json.loads((p/'inputs/line2/train55.json').read_text())
heads=raw['headers']
base=heads[:22]
full=heads[:22]+heads[24:]
t=Document(p/'SI_Verified.docx').tables[21]
names=['RF','XGBoost','LightGBM','CatBoost']
s4={name:[x.strip().rstrip('.') for x in t.rows[i].cells[1].text.split(';')] for i,name in enumerate(names,1)}
assert [len(s4[a]) for a in names]==[37,15,18,40]
assert all(set(s4[a])<=set(full) for a in names)
union=set().union(*[set(s4[a]) for a in names]); s4['Equal4']=[f for f in full if f in union]
assert len(s4['Equal4'])==47
features={str(stage):{m:(base if stage==1 else full if stage in [2,3] else s4[m]) for m in names+['Equal4']} for stage in range(1,5)}
(p/'stage_matched_feature_lists.json').write_text(json.dumps({'source':'Line2 SI_Verified.docx Table S22? original run.py SUBSETS uses tables[21]','stage_features':features,'S4_equal4':'Union of four component RFE feature lists'},indent=2,ensure_ascii=False))
print('Feature lengths:',{st:{m:len(cols) for m,cols in models.items()} for st,models in features.items()})
