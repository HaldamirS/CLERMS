import pickle
from tqdm import tqdm
import numpy as np
from scipy.sparse import csr_matrix
import pandas as pd

def get_vec(x, dim=1000):
    intensity_list = np.sqrt(x['intensities'])
    intensity_list = intensity_list/np.max(intensity_list)
    vector = [0] * dim
    mzlist = x['mz']
    indexes = mzlist
    indexes = np.around(indexes).astype('int32')

    for i, index in enumerate(indexes):
        if index < dim:
            vector[index] = max(intensity_list[i], vector[index])
    return csr_matrix(vector, shape=(1, dim))

def get_trans_vec(x, dim=100):
    intensity_list = np.sqrt(x['intensities'])
    intensity_list = intensity_list/np.max(intensity_list)
    mzlist = x['mz']
    n_peaks = len(intensity_list)
    descend_order = intensity_list.argsort()[::-1]
    mzlist = mzlist[descend_order]
    intensity_list = intensity_list[descend_order]
    if n_peaks>dim:
        return np.stack([mzlist[:dim],intensity_list[:dim]])
    else:
        intensity_list = np.pad(intensity_list, (0, dim-n_peaks), 'constant', constant_values=0)
        mzlist = np.pad(mzlist, (0, dim-n_peaks), 'constant', constant_values=0)
        return np.stack([mzlist,intensity_list])

def mgf2df(mgf_path,bin=True):    
    with open(mgf_path,'rb') as f:
        mgf = pickle.load(f) 
    data_collect = []
    # 只存一部分信息    
    print(f"从{mgf_path}加载数据")
    for i in tqdm(mgf):
        if i:
            temp = {}
            temp['mz'] = i.mz
            temp['intensities']=i.intensities
            temp['inchikey'] = i.metadata.get("inchikey","-1")
            temp['precursor_mz'] = i.metadata.get("precursor_mz",0)
            temp['inchikey14'] = i.get("inchikey14",'0')
            temp['source_instrument'] = i.get('source_instrument','0')
            temp['adduct'] = i.get('adduct','0')
            temp['charge'] = i.get('charge')
            data_collect.append(temp)

    data = pd.DataFrame(data_collect)
    if bin:
        data["vec"] = data.apply(get_vec,axis=1)
    else:
        data["vec"] = data.apply(get_trans_vec, axis=1)
    data['m_id'] = data['inchikey14']
    
    print(f"总数据为{data.shape[0]},其中唯一的化合物为{len(data['m_id'].unique())}")
    
    return data
