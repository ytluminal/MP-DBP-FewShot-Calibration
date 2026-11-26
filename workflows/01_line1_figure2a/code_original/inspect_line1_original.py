from artifact_tool import Blob,SpreadsheetFile
import os,time
files=['Microplastics_DBPs_Combined_Original22 features(5).xlsx','Conventional_RandomSplit_Stage1_Training_22features(3).xlsx','Conventional_RandomSplit_Stage1_Test_22features(3).xlsx','Conventional_RandomSplit_Stage2_Training_55features(3).xlsx','Conventional_RandomSplit_Stage2_Test_55features(3).xlsx']
for f in files:
 t=time.time(); w=SpreadsheetFile.import_xlsx(Blob.load('/mnt/data/'+f)); s=w.worksheets.get_item_at(0)
 print(f,'sheet',s.name, 'import time',round(time.time()-t,1), flush=True)
 print(s.get_range('A1:D3').values,flush=True)
 print('last header',s.get_range('T1:BI1').values,flush=True)
