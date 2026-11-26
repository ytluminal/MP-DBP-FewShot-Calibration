"""Persistent LightGBM preprocessing, exact clone of frozen Line 2 run.py prepare().

Place this module alongside the saved .joblib files (or on PYTHONPATH) to load
preprocessing objects. No external test labels are ever used for fit().
"""
import numpy as np
from scipy.sparse import csr_matrix,hstack
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import KNNImputer,SimpleImputer
CATCOLS = ['polymer type','additives','pretreatment type','sample phase','buffer','disinfectant','quenching agent','DBP name']

class FrozenLightGBMPreprocess:
    """Implements source run.py prepare() for LightGBM; each component fit only on active outer fold."""
    def __init__(self,features):
        self.features=list(features)
        self.cat_local=[i for i,c in enumerate(features) if c in CATCOLS]
        self.num_local=[i for i in range(len(features)) if i not in self.cat_local]
        self.numerator=KNNImputer(n_neighbors=5,weights='distance',metric='nan_euclidean')
        self.scaler=StandardScaler()
        self.mode=SimpleImputer(strategy='most_frequent') if self.cat_local else None
        self.encoder=OneHotEncoder(handle_unknown='ignore',dtype=np.float64,sparse_output=True) if self.cat_local else None
    def _num(self,x):
        return np.asarray([[np.nan if x[i,j] is None or x[i,j]=='' else float(x[i,j]) for j in self.num_local] for i in range(len(x))],dtype=float)
    def _cat(self,x):
        z=np.asarray([['__NAN__' if x[i,j] is None or x[i,j]=='' else str(x[i,j]) for j in self.cat_local] for i in range(len(x))],dtype=object).reshape(len(x),len(self.cat_local))
        z[z=='__NAN__']=np.nan
        return z
    def fit(self,raw):
        n=self.numerator.fit_transform(self._num(raw));self.scaler.fit(n)
        if self.cat_local:self.encoder.fit(self.mode.fit_transform(self._cat(raw)))
        return self
    def transform(self,raw):
        n=self.scaler.transform(self.numerator.transform(self._num(raw)))
        if not self.cat_local:return csr_matrix(n)
        c=self.encoder.transform(self.mode.transform(self._cat(raw)))
        return hstack([csr_matrix(n),c],format='csr')
    def transform_dataframe(self,df):
        """For inference, df columns must contain exact original feature labels."""
        x=df.loc[:,self.features].astype(object).to_numpy(dtype=object)
        # pandas missing sentinel -> None matching JSON missing representation
        x[np.asarray(__import__('pandas').isna(x))]=None
        return self.transform(x)

