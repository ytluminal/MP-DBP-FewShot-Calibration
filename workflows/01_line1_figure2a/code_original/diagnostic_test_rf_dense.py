import reproduce_line1 as q
old=q.preprocess
def dense_rf(*a,**kw):
 x,y,c=old(*a,**kw)
 if a[3]=='Random Forest': return x.toarray(),y.toarray(),c
 return x,y,c
q.preprocess=dense_rf
for s in ['S1','S2']:
 o=q.run(s,'Random Forest')
 print('DENSE',s,o['cv']['r2'],o['test']['r2'])
