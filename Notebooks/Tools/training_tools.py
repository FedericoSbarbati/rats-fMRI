import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torch.optim.lr_scheduler as lr_scheduler
from torch.optim.lr_scheduler import LambdaLR
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from torch.utils.data import random_split
from sklearn.metrics import mean_absolute_error
import os
import math


# AUXILIARY FUNCTIONS FOR TRAINING AND EVALUATION
class EarlyStopping:
    def __init__(self, patience=10, delta=0.001, verbose=False):
        """
        Implementa Early Stopping per interrompere il training.
        
        Parametri:
        - patience: numero di epoche senza miglioramenti dopo cui fermare il training.
        - delta: minimo miglioramento richiesto per considerare una variazione significativa.
        - verbose: se True, stampa messaggi sullo stato dell'early stopping.
        """
        self.patience = patience
        self.delta = delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.stopped_epoch = 0

    def __call__(self, val_loss, epoch):
        """
        Controlla se fermare il training.
        
        Parametri:
        - val_loss: perdita di validazione dell'epoca corrente.
        - epoch: epoca corrente.
        """
        if self.best_loss is None or val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement.")
            if self.counter >= self.patience:
                self.early_stop = True
                self.stopped_epoch = epoch

def get_scheduler(optimizer, scheduler_config):
    """
    Crea uno scheduler in base alla configurazione fornita.

    Parametri:
    - optimizer: l'ottimizzatore associato.
    - scheduler_config: configurazione completa dello scheduler (con "type" e "params").

    Ritorna:
    - Istanza dello scheduler selezionato.
    """
    if scheduler_config is None or "type" not in scheduler_config:
        raise ValueError("Scheduler configuration or 'type' is missing.")

    scheduler_type = scheduler_config["type"]
    if scheduler_type == "None":
        return LambdaLR(optimizer, lr_lambda=lambda epoch: 1.0)


    scheduler_params = scheduler_config.get("params", {})

    if scheduler_type == "StepLR":
        return lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_type == "ExponentialLR":
        return lr_scheduler.ExponentialLR(optimizer, **scheduler_params)
    elif scheduler_type == "ReduceLROnPlateau":
        relevant_params = {k: scheduler_params[k] for k in ["mode", "factor", "patience", "min_lr"] if k in scheduler_params}
        return lr_scheduler.ReduceLROnPlateau(optimizer, **relevant_params)
    elif scheduler_type == "CyclicLR":
        # Filtra i parametri per CyclicLR
        relevant_params = {k: scheduler_params[k] for k in ["base_lr", "max_lr", "step_size_up"] if k in scheduler_params}
        return lr_scheduler.CyclicLR(optimizer, **relevant_params)
    elif scheduler_type == "CosineAnnealingLR":
        # Filtra i parametri per CosineAnnealingLR
        relevant_params = {k: scheduler_params[k] for k in ["T_max", "eta_min"] if k in scheduler_params}
        return lr_scheduler.CosineAnnealingLR(optimizer, **relevant_params)
    elif scheduler_type == "CosineAnnealingWarmRestarts":
        # Filtra i parametri per CosineAnnealingWarmRestarts
        relevant_params = {k: scheduler_params[k] for k in ["T_0", "T_mult", "eta_min"] if k in scheduler_params}
        return lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **relevant_params)
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

def get_beta(epoch, warmup_epochs, method="constant", beta_value=1.0, decay_start=100, decay_epochs=50):
    """
    Calcola il valore di beta in base al metodo specificato.

    Parametri:
    - epoch: epoca corrente.
    - warmup_epochs: numero di epoche di warm-up.
    - method: metodo per calcolare beta (constant, sigmoid, linear, linear_decay, exponential_decay, annealing).
    - beta_value: valore costante di beta (usato per il metodo "constant").
    - decay_start: epoca in cui iniziare la diminuzione di beta.
    - decay_epochs: numero di epoche su cui effettuare la diminuzione (lineare/esponenziale).

    Ritorna:
    - Valore di beta.
    """
    if method == "constant":
        return beta_value

    elif method == "sigmoid":
        return beta_value / (1 + math.exp(-0.1 * (epoch - warmup_epochs // 2)))

    elif method == "linear":
        return min(1.0, epoch / warmup_epochs)

    elif method == "linear_decay":
        # Warm-up fino a warmup_epochs, poi diminuzione lineare
        if epoch <= warmup_epochs:
            return min(1.0, epoch / warmup_epochs)
        elif epoch > decay_start:
            return max(0.0, 1.0 - (epoch - decay_start) / decay_epochs)
        else:
            return 1.0

    elif method == "exponential_decay":
        # Warm-up fino a warmup_epochs, poi diminuzione esponenziale
        if epoch <= warmup_epochs:
            return min(1.0, epoch / warmup_epochs)
        elif epoch > decay_start:
            decay_rate = decay_epochs / 5  # Fattore di scala per il decadimento
            return max(0.0, 1.0 * math.exp(-float(epoch - decay_start) / decay_rate))
        else:
            return 1.0

    elif method == "annealing":
        # Annealing da 1.0 fino a beta_value
        if epoch < warmup_epochs:
            return 1.0 - (epoch / warmup_epochs) * (1.0 - beta_value)
        else:
            return beta_value

    else:
        raise ValueError(f"Metodo di beta sconosciuto: {method}")
    
def loss_function_vae(recon_x, x, mu, log_var, beta):
    # Ricostruzione loss (normalizzata per batch size)
    batch_size = x.size(0)
    recon_loss = torch.nn.functional.mse_loss(recon_x, x, reduction='sum') / batch_size
    
    # KL divergence normalizzata per il numero di dimensioni latenti
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size
    
    return recon_loss + beta * kld_loss, recon_loss, kld_loss

def save_network(encoder, decoder, optimizer, epoch, stopped_epoch, encoder_layers, decoder_layers, 
               train_losses, val_losses, recon_losses, kld_losses, effective_kld_losses,separate_recon_losses, beta_values, gradient_history, lr_evolution, 
               output_folder="Models", model_name="vae_model.pth"):
    """
    Salva encoder, decoder, lo stato dell'ottimizzatore e i dati di training/validazione in un unico file.

    Parametri:
    - encoder: rete encoder da salvare.
    - decoder: rete decoder da salvare.
    - optimizer: ottimizzatore associato.
    - train_data: dati di training.
    - val_data: dati di validazione.
    - epoch: epoca corrente al momento del salvataggio.
    - stopped_epoch: epoca in cui l'early stopping ha fermato il training.
    - encoder_layers, decoder_layers: dimensioni dei layer di encoder e decoder.
    - train_losses, val_losses, recon_losses, kld_losses, beta_values: metriche del training.
    - gradient_history: storico dei gradienti.
    - output_folder: cartella di destinazione.
    - model_name: nome del file di salvataggio.
    - denormalizing_params: parametri per la denormalizzazione dei dati.
    - legendreBasis: base di Legendre per la ricostruzione.
    """
    os.makedirs(output_folder, exist_ok=True)
    file_path = os.path.join(output_folder, model_name)

    torch.save({
        'epoch': epoch,
        'stopped_epoch': stopped_epoch,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'encoder_layers': encoder_layers,
        'decoder_layers': decoder_layers,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'recon_losses': recon_losses,
        'kld_losses': kld_losses,
        'effective_kld_losses': effective_kld_losses,
        'separate_recon_losses': separate_recon_losses,
        'beta_values': beta_values,
        'gradient_history': gradient_history,
        'lr_evolution': lr_evolution
    }, file_path)

    print(f"Model saved to {file_path}")   

def load_vae(file_path,decoder_layers, encoder_layers,decoder_class, encoder_class, VAE_class):
    """
    Carica il decoder e le informazioni del training da un file .pth.
    
    Parametri:
    - file_path: percorso del file .pth salvato.
    - decoder_class: classe da usare per ricostruire il decoder.
    
    Ritorna:
    - decoder: istanza del decoder con i parametri caricati.
    - training_info: dizionario contenente le variabili del training e dei dati.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Carica il checkpoint dal file
    checkpoint = torch.load(file_path,map_location=device)

    # Ricostruisci il decoder
    decoder_layers_reversed = decoder_layers[::-1]
    decoder = decoder_class(decoder_layers_reversed)
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    # Encoder reconstruction
    encoder = encoder_class(encoder_layers)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])

    # VAE model instance:
    vae_model = VAE_class(encoder,decoder)

    # Raccogli le informazioni del training
    training_info = {
        'train_losses': checkpoint['train_losses'],
        'val_losses': checkpoint['val_losses'],
        'recon_losses': checkpoint['recon_losses'],
        'kld_losses': checkpoint['kld_losses'],
        'effective_kld_losses': checkpoint['effective_kld_losses'],
        'separate_recon_losses': checkpoint['separate_recon_losses'],
        'beta_values': checkpoint['beta_values'],
        'epoch': checkpoint['epoch'],
        'stopped_epoch': checkpoint['stopped_epoch'],
        'gradient_history': checkpoint['gradient_history'],
        'lr_evolution': checkpoint['lr_evolution'],
    }

    print(f"Decoder and training info loaded from {file_path}")
    return vae_model, training_info





