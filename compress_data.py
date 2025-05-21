import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.signal import butter, filtfilt, detrend
from scipy.stats import zscore
from scipy.ndimage import gaussian_filter,gaussian_filter1d
import os
import re


# Cartelle di input e output
uncompressed_folder = "/content/drive/MyDrive/Colab Notebooks/fmri/Uncompressed Data"
compressed_folder = "/content/drive/MyDrive/Colab Notebooks/fmri/Compressed Data"


uncompressed_folder = "Uncompressed Data"
compressed_folder = "Compressed Data"

# Crea la cartella di output se non esiste
if not os.path.exists(compressed_folder):
    os.makedirs(compressed_folder)

# Pattern per individuare file rs-fmri-#.nii
pattern = re.compile(r"rs-fmri-\d+\.nii")

# Lista dei file nella cartella Uncompressed Data che soddisfano il pattern
files = [f for f in os.listdir(uncompressed_folder) if pattern.match(f)]

for file_name in files:
    print(f"Processing {file_name} ...")
    # Percorso completo del file
    input_path = os.path.join(uncompressed_folder, file_name)

    # Rimuovo l'estensione per usarla nel nome della cartella di output
    base_name = file_name.replace('.nii', '')

    # Caricamento dei dati funzionali
    Im = nib.load(input_path)
    functional_data = Im.get_fdata()
    func_std = np.std(functional_data, axis=-1)

    # Detrending temporale
    detrended_data = detrend(functional_data, axis=-1)

    # Reshape of the array to have the time series as rows
    functional_data_res = np.reshape(detrended_data,[56*48*32,925]) 
    func_std_res = np.reshape(func_std,[56*48*32,1])

    # Seleziona tutti i voxel attivi (con deviazione standard maggiore di zero)
    active_voxel_indices = np.where(func_std_res > 0)[0]
    functional_data_res = functional_data_res[active_voxel_indices, :]


    # Normalizzazione Z-score e salvataggio di media e deviazione standard per la denormalizzazione
    mean_voxel = np.mean(functional_data_res, axis=-1, keepdims=True)
    std_voxel = np.std(functional_data_res, axis=-1, keepdims=True)
    zscore_normalized_data = (functional_data_res - mean_voxel) / std_voxel

    # Se qualche riga (voxel) contiene inf o nan, sostituiscila con zero
    for i in range(zscore_normalized_data.shape[0]):
        if not np.all(np.isfinite(zscore_normalized_data[i])):
            zscore_normalized_data[i] = 0

    # Cast zscore_normalized_data to float32
    zscore_normalized_data = zscore_normalized_data.astype(np.float32)

    print("Shape: ", zscore_normalized_data.shape)

    save_path = os.path.join(compressed_folder, base_name + ".npz")
    print(f"  Saving to {save_path}")

    # Salvataggio dei dati compressi in formato NPZ
    np.savez(save_path,
             functional_data_res=zscore_normalized_data,
             active_voxel_indices=active_voxel_indices,
             mean_voxel=mean_voxel,
             std_voxel=std_voxel)