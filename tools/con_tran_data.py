from torch.utils.data import Dataset
import numpy as np
import random

class ConTranDataGen(Dataset):
    def __init__(self, dataset,keys,score_dict, nround = 1, nviews=2,augment=False):
        self.data = dataset
        self.keys = keys
        self.nround = nround
        self.score_dict = score_dict
        self.nviews = nviews
        self.augment = augment

    def __len__(self):
        return len(self.keys) * self.nround

    def aug(self, vec, prob=0.5):
        if np.random.random()<prob:
            return vec
        augment_removal_max=0.5
        augment_removal_intensity=0.3
        augment_intensity=0.4
        self.augment_noise_max=10
        self.augment_noise_intensity=0.01
        idx = vec[0]
        values = vec[1]
        indices_select = np.where(values<augment_removal_intensity)[0]
        removal_part = np.random.random(1) * augment_removal_max
        indices_select = np.random.choice(
            indices_select, int(np.ceil((1-removal_part)*len(indices_select))), replace=False
        )
        indices = np.concatenate(
            [
                indices_select,
                np.where(values>=augment_removal_intensity)[0]
            ]
        )
        if len(indices) > 0:
            idx = idx[indices]
            values = values[indices]
        values = (
            1 - augment_intensity * 2 * (np.random.random(values.shape)-0.5)
        ) * values

        idx, values = self._peak_addition(idx, values)
        return np.concatenate([idx.reshape(-1,1),values.reshape(-1,1)/values.max()], axis=1)

    def _peak_addition(self, idx, values):
        n_noise_peaks = np.random.randint(0, self.augment_noise_max)
        idx_no_peaks = np.setdiff1d(np.arange(0,1000,0.01),idx)
        idx_noise_peaks = np.random.choice(idx_no_peaks, n_noise_peaks, replace=False)
        idx = np.concatenate([idx, idx_noise_peaks])
        new_values = self.augment_noise_intensity * np.random.random(len(idx_noise_peaks))
        values = np.concatenate([values, new_values])
        return idx, values

    def __getitem__(self, index):
        inchikey = self.keys[index%len(self.keys)]
        specs = []
        if self.augment: # 如果有数据增强
            total_specs = len(self.data[inchikey])
            if total_specs>self.nviews:
                specs = [self.data[inchikey][i] for i in np.random.choice(np.arange(total_specs),self.nviews,replace=False)]
                res = [[self.aug(spec1[0]),spec1[1],spec1[2],np.array(spec1[3]).reshape(1)] for spec1 in specs]
            else:
                res = [[self.aug(spec1[0]),spec1[1],spec1[2],np.array(spec1[3]).reshape(1)] for spec1 in self.data[inchikey]]
                for i in range(total_specs,self.nviews):
                    spec1 = random.choice(self.data[inchikey])
                    res.append([self.aug(spec1[0],0),spec1[1],spec1[2],np.array(spec1[3]).reshape(1)])
            return res
        else:
            for i in range(self.nviews):
                spec1 = random.choice(self.data[inchikey])
                specs.append([spec1[0],spec1[1],spec1[2],np.array(spec1[3]).reshape(1),spec1[4]])
            specs.append(self.score_dict[inchikey])
            return specs
