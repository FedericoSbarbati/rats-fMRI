import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import zscore
import os
import re
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from Notebooks.Tools.fmri_tools import extract_number_from_filename, filter_pupil_signal


#NOTE: La logica nel codice principale prevede che questo venga eseguito sull'intero dataset zenodo compresso

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
    
    best_corr = 999                             #POSITIVE CORR: -999 for best positive correlation
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
        if corr < best_corr:        #POSITIVE CORR: > best_corr
            best_corr = corr
            best_lag = lag
    
    return best_lag, best_corr, all_corr


# Cartelle di input e output
uncompressed_folder = "Compressed Data"
print(uncompressed_folder)

# Pattern per individuare file rs-fmri-#.nii
pattern = re.compile(r"rs-fmri-(\d+)\.npz")
# Lista dei file nella cartella Uncompressed Data che soddisfano il pattern
files = sorted([f for f in os.listdir(uncompressed_folder) if pattern.match(f)], key=lambda x: int(pattern.search(x).group(1)))

pupil_name = 'pupil_diameter_74_trials.npy'
pupil = np.load(pupil_name)
fmin = 0.01
fmax = 0.05
b, a = butter(4, [fmin,fmax], 'bandpass')
print(np.shape(pupil))
pupil_filtered = filtfilt(b, a, pupil,padlen=10) 


best_lag = []
best_corr = []
corr_dict_list = []  # lista per contenere i dizionari di correlazione

for file_name in files:
    print(f"Processing {file_name} ...")
    # Percorso completo del file
    input_path = os.path.join(uncompressed_folder, file_name)
    index = extract_number_from_filename(file_name)

    # Rimuovo l'estensione per usarla nel nome della cartella di output
    base_name = file_name.replace('.npz', '')

    # Caricamento dei dati funzionali
    data = np.load(input_path)
    functional_data_res = data['functional_data_res']
    print('Functional data res shape:', np.shape(functional_data_res))
    pca = PCA(n_components=1)
    y = pca.fit_transform(functional_data_res.T)
    print(np.shape(y))
    b, a = butter(4, [fmin, fmax], 'bandpass')
    y_filtered = filtfilt(b, a, y, padlen=10, axis=0)
    
    # Ora, assicurati che il segnale pupil passato sia 1D:
    lag, corr, all_corr = estimate_best_lag(pupil_filtered[(index-1), :], y_filtered, max_lag=20)
    print(f"Miglior lag: {lag} campioni con correlazione={corr:.3f}")

    best_lag.append(lag)
    best_corr.append(corr)
    corr_dict_list.append(all_corr)

np.save('best_lag.npy', best_lag)
np.save('best_corr.npy', best_corr)
np.save('corr_dict.npy', corr_dict_list)






