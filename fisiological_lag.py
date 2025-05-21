# https://zenodo.org/records/4670277
#
#

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import nibabel as nib
from sklearn.decomposition import PCA
from scipy.stats import zscore
import os
from sklearn.mixture import GaussianMixture
from scipy.ndimage import gaussian_filter,gaussian_filter1d
from nilearn import plotting, image
from scipy.ndimage import zoom

import numpy as np

def estimate_best_lag(pupil, fmri, max_lag=20):
    """
    Stima il lag (in campioni) che massimizza la correlazione lineare
    tra 'pupil' e 'fmri'.
    
    Parameters
    ----------
    pupil : np.ndarray
        Serie temporale del diametro pupillare, shape (T,).
    fmri : np.ndarray
        Serie temporale fMRI (ad es. media ROI o primo PC), shape (T,).
    max_lag : int
        Il lag massimo (in campioni) da esplorare in avanti o indietro. 
    
    Returns
    -------
    best_lag : int
        Il lag (positivo o negativo) che massimizza la correlazione.
        Se best_lag > 0, vuol dire che la fMRI è in ritardo (lag) 
        rispetto al segnale pupillare.
    best_corr : float
        Valore di correlazione (Pearson) massimo riscontrato.
    all_corr : dict
        Dizionario con lag -> correlazione, utile per debug o plotting.
    """
    # Assicuriamoci che i due segnali abbiano la stessa lunghezza
    T = min(len(pupil), len(fmri))
    pupil = pupil[:T]
    fmri = fmri[:T]
    
    # Normalizziamo (z-score) i dati per evitare offset
    p_mean, p_std = pupil.mean(), pupil.std()
    pupil_z = (pupil - p_mean) / (p_std if p_std>0 else 1e-8)
    
    f_mean, f_std = fmri.mean(), fmri.std()
    fmri_z = (fmri - f_mean) / (f_std if f_std>0 else 1e-8)

    # Converti fmri_z in un array 1D se è 2D
    fmri_z = np.squeeze(fmri_z)
    
    # Per memorizzare i risultati
    lags = range(-max_lag, max_lag + 1)
    all_corr = {}
    
    best_corr = -999
    best_lag = 0
    
    for lag in lags:
        if lag < 0:
            # Se lag è negativo, shiftiamo in avanti pupil
            # Esempio: pupil(t + lag) ~ pupil spostato in avanti
            # Quindi fmri(t) corrisponde a pupil(t + lag)
            # => pupil_z[:lag] vs fmri_z[-lag:]
            shifted_p = pupil_z[:lag]   # da 0 a T+lag
            shifted_f = fmri_z[-lag:]  # da -lag a T
        else:
            # lag >= 0
            # fmri(t) corrisponde a pupil(t - lag)
            # => pupil_z[lag:] vs fmri_z[:T-lag]
            shifted_p = pupil_z[lag:]   
            shifted_f = fmri_z[:len(pupil_z[lag:])]  # che è T-lag
        
        # Calcola correlazione di Pearson
        if len(shifted_p) > 1 and len(shifted_p) == len(shifted_f):
            corr = np.corrcoef(shifted_p, shifted_f)[0, 1]
        else:
            corr = np.nan
        
        all_corr[lag] = corr
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    
    return best_lag, best_corr, all_corr


'''
# Esempio di utilizzo:
if __name__ == "__main__":
    # Supponiamo di avere le serie pupil e fmri come array np
    # (ad es. caricati da un file .npy)
    pupil = np.random.randn(1000)  # dummy data
    fmri  = np.random.randn(1000)  # dummy data

    lag, corr, corr_dict = estimate_best_lag(pupil, fmri, max_lag=50)
    print(f"Miglior lag: {lag} campioni con correlazione={corr:.3f}")
'''


#Loading the functional data from file
frmi_path = os.path.join("",'rs-fmri-5.nii') 
Im=nib.load(frmi_path)
functional_data=Im.get_fdata()

#Loading pupil data:
file_name = 'pupil_diameter_74_trials.npy'

index = 5
pupil = np.load(file_name)
pupil = pupil[index,:]
fmin = 0.01
fmax = 0.2
b, a = butter(4, [fmin,fmax], 'bandpass')
pupil_filtered = filtfilt(b, a, pupil,padlen=10) 

# Function to evaluate std of the functional data along the time axis
func_std = np.std(functional_data,axis=-1)
brain_mask = func_std>30

#print("Functional data shape: ", np.shape(functional_data))
#print("Brain mask shape: ", np.shape(brain_mask))

# Reshape of the array to have the time series as rows
functional_data_res = np.reshape(functional_data,[56*48*32,925]) 
func_std_res = np.reshape(func_std,[56*48*32,1])

# Remove the voxels with std=0 or less
functional_data_res=functional_data_res[np.where(func_std_res>0)[0],:]
active_voxel_indices = np.where(func_std_res > 0)[0]

print('Functional data res shape:',np.shape(functional_data_res))
pca = PCA(n_components=1)
y=pca.fit_transform(functional_data_res.T)
print(np.shape(y))
b, a = butter(4, [fmin,fmax], 'bandpass')
y_filtered = filtfilt(b, a, y,padlen=10,axis=0)

print('PC1 fmri shape:', np.shape(y_filtered))
print('Pupil shape:', np.shape(pupil_filtered))

lag, corr, corr_dict = estimate_best_lag(pupil_filtered, y_filtered, max_lag=100)
print(f"Miglior lag: {lag} campioni con correlazione={corr:.3f}")




