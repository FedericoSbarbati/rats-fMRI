import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
import re

from Tools.training_tools import*

class one_layer_net(torch.nn.Module):    
    # Constructor
    def __init__(self, input_size, output_size):
        super(one_layer_net, self).__init__()
        # hidden layer 
        self.linear_one = torch.nn.Linear(input_size, output_size) 
        # defining layers as attributes
    # prediction function
    def forward(self, x):
        y_pred = self.linear_one(x)
        return y_pred

def reconstruct_image_from_compressed_data(data, indices):
    original_shape = (56,48)
    time_points = 925

    recon = np.zeros((np.prod(original_shape), time_points))

    recon[indices, :] = data
    # Reshape per tornare alla forma originale
    recon = recon.reshape(original_shape + (time_points,))

    return recon

def extract_number_from_filename(filename):
    """
    Estrae il numero dal nome del file del tipo 'rs-fmri-#'.
    
    Args:
    filename (str): Il nome del file da cui estrarre il numero.
    
    Returns:
    int: Il numero estratto dal nome del file.
    """
    match = re.search(r'rs-fmri-(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError("Il nome del file non corrisponde al formato 'rs-fmri-#'")
    
def filter_pupil_signal(pupil,fmin,fmax):

    b, a = butter(4, [fmin,fmax], 'bandpass')

    pupil_filtered = filtfilt(b, a, pupil,padlen=10) #, axis=1)
    return pupil_filtered

def reconstruct_vector_signal_from_embedding(embedding, delay):
    """
    Ricostruisce il segnale temporale vettoriale da un embedding ritardato.

    Parametri:
      - embedding: Array numpy con l'embedding di forma (embedding_dim, d, n_samples),
                   dove d è la dimensione del segnale (numero di canali).
      - embedding_dim: Dimensione dell'embedding (int)
      - delay: Ritardo tra campioni nell'embedding (int)

    Ritorna:
      - signal_reconstructed: Array numpy 2D con il segnale ricostruito di forma (n_total_samples, d),
                              dove n_total_samples = n_samples + (embedding_dim - 1) * delay.
    """
    embedding_dim, d ,n_samples = embedding.shape
    n_total_samples = n_samples + (embedding_dim - 1) * delay

    signal_reconstructed = np.zeros((n_total_samples, d))
    weight = np.zeros(n_total_samples)

    for i in range(n_samples):
        for j in range(embedding_dim):
            idx = i + j * delay
            signal_reconstructed[idx, :] += embedding[j, :, i]
            weight[idx] += 1

    # Evita divisione per zero e media i contributi sovrapposti
    signal_reconstructed /= np.maximum(weight[:, None], 1)

    return signal_reconstructed
    
def invert_functional_data(functional_data_active, active_voxel_indices, spatial_shape, n_timepoints):
    """
    Inverte il processo di flattening dei dati funzionali.
    
    Parametri:
      - functional_data_active: array numpy di forma (n_active, n_timepoints) contenente 
          le time series dei voxel attivi.
      - active_voxel_indices: array numpy contenente gli indici dei voxel attivi 
          ottenuti dalla versione flattenata degli originali dati (dim = np.prod(spatial_shape)).
      - spatial_shape: tupla contenente la forma spaziale originale (es. (56, 48, 32)).
      - n_timepoints: numero di time points (ad es. 925).
    
    Ritorna:
      - full_data: array numpy 4D di forma (spatial_shape[0], spatial_shape[1], spatial_shape[2], n_timepoints)
        in cui alle posizioni corrispondenti ai voxel attivi viene inserito il segnale e
        i voxel inattivi sono impostati a np.nan.
    """
    # Inizializza l'array flatten con np.nan
    full_flat = np.full((np.prod(spatial_shape), n_timepoints), np.nan, dtype=np.float32)
    # Inserisce il segnale nei voxel attivi
    full_flat[active_voxel_indices, :] = functional_data_active.astype(np.float32)
    # Rimodella in 4D secondo la shape spaziale originale e il numero di timepoints
    full_data = full_flat.reshape((*spatial_shape, n_timepoints))
    return full_data

def recon_fmri_pc(device, epochs, train_loader, val_loader, 
                model, optimizer, scheduler, scheduler_config,
                kl_annealing_epochs, decay_start, decay_epoch,
                r, n_components,
                beta_method="constant", beta_value=1.0, early_stopping_params=None):
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    recon_losses = []         # loss di ricostruzione totale
    kld_losses = []           # loss KLD
    effective_kld_losses = [] # beta * KLD
    separate_recon_losses = []  # lista per le loss per ciascun componente (lista di liste)
    beta_values = []
    lr_evolution = []
    
    # Per salvare i gradienti medi per epoch
    gradient_history = {
        "encoder": [],
        "decoder": [],
        "latent": []
    }

    for epoch in range(epochs):
        model.train()
        model.to(device)

        epoch_loss = 0
        kld_loss_epoch = 0
        recon_loss_epoch = 0
        real_kld_loss = 0
        # Inizializzo la lista che accumulerà la loss per ciascun componente per questa epoca
        separate_recon_loss = [0] * n_components

        # Calcola il valore di beta
        beta = get_beta(epoch, kl_annealing_epochs, method=beta_method, beta_value=beta_value, decay_start=decay_start, decay_epochs=decay_epoch)
        beta_values.append(beta)

        for batch in train_loader:
            input, target = batch  # input: (batch_size, r), target: (batch_size, n_components, r)
            input = input.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad()

            # Determino il batch size
            batch_size = input.size(0)

            # Calcola l'output della rete
            recon, mu, log_var = model(input)
            # Effettua il reshape: da (batch_size, n_components * r) a (batch_size, n_components, r)
            recon_reshaped = recon.view(batch_size, n_components, r)
            
            # Calcolo della loss di ricostruzione per ciascun componente (sommo sulle righe e normalizzo per il batch)
            total_recon_loss = 0.0
            for i in range(n_components):
                recon_loss_component = torch.nn.functional.mse_loss(
                    recon_reshaped[:, i, :],
                    target[:, i, :],
                    reduction='sum'
                ) / batch_size
                separate_recon_loss[i] += recon_loss_component.item()
                total_recon_loss += recon_loss_component

            # KL divergence normalizzata per il numero di dimensioni latenti
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            batch_loss = total_recon_loss + beta * kld_loss
            batch_loss.backward()

            # Variabili temporanee per sommare i gradienti per batch
            encoder_grad_total = 0
            decoder_grad_total = 0
            latent_grad_total = 0
            num_batches = 0
            # Clip dei gradienti
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Calcolo dei gradienti per parte
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    if name in ["encoder.encoder.0.weight", "encoder.encoder.0.bias", "encoder.encoder.2.weight", "encoder.encoder.2.bias"]:
                        encoder_grad_total += grad_norm
                    elif name in ["decoder.decoder.0.weight", "decoder.decoder.0.bias", "decoder.decoder.2.weight", "decoder.decoder.2.bias", "decoder.decoder.4.weight", "decoder.decoder.4.bias"]:
                        decoder_grad_total += grad_norm
                    elif name in ["encoder.fc_mu.weight", "encoder.fc_mu.bias", "encoder.fc_log_var.weight", "encoder.fc_log_var.bias"]:
                        latent_grad_total += grad_norm
            num_batches += 1

            # Accesso al learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            lr_evolution.append(current_lr)

            # Salva i gradienti medi per questo batch
            gradient_history["encoder"].append(encoder_grad_total / num_batches)
            gradient_history["decoder"].append(decoder_grad_total / num_batches)
            gradient_history["latent"].append(latent_grad_total / num_batches)

            optimizer.step()

            epoch_loss += batch_loss.item()
            kld_loss_epoch += kld_loss.item()
            recon_loss_epoch += total_recon_loss.item()
            real_kld_loss += (beta * kld_loss.item())

        # Calcolo delle medie per epoca, usando il numero totale di campioni
        n_total = len(train_loader.dataset)
        epoch_loss /= n_total
        kld_loss_epoch /= n_total
        recon_loss_epoch /= n_total
        real_kld_loss /= n_total
        # Per le loss separate, dividiamo ciascuna per n_total
        separate_recon_loss_avg = [loss_val / n_total for loss_val in separate_recon_loss]

        train_losses.append(epoch_loss)
        recon_losses.append(recon_loss_epoch)
        kld_losses.append(kld_loss_epoch)
        effective_kld_losses.append(real_kld_loss)
        separate_recon_losses.append(separate_recon_loss_avg)
               
        # VALIDATION
        model.eval()
        model.to(device)
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch  # (batch_size, r) e (batch_size, n_components, r)
                input = input.float().to(device)
                target = target.float().to(device)
                recon, mu, log_var = model(input)
                batch_size_val = input.size(0)
                recon_reshaped = recon.view(batch_size_val, n_components, r)
                total_recon_loss = 0.0
                for i in range(n_components):
                    val_loss_component = torch.nn.functional.mse_loss(
                        recon_reshaped[:, i, :],
                        target[:, i, :],
                        reduction='sum'
                    ) / batch_size_val
                    total_recon_loss += val_loss_component
                kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size_val  
                val_loss += (total_recon_loss + beta * kld_loss).item()

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            if scheduler_config["type"] == "None":
                pass
            else:
                scheduler.step()

        # Early Stopping
        if early_stopping:
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Stampa i valori medi per l'epoca
        print(f'Epoch [{epoch+1}/{epochs}], Rec Loss: {recon_loss_epoch:.4f}, KLD Loss: {kld_loss_epoch:.4f}, '
              f'Beta: {beta:.4f}, KLD*B = {real_kld_loss:.4f}, Total Loss: {epoch_loss:.4f}, lr = {current_lr:.6f}')
        print(f'Separate Recon Loss per component: {separate_recon_loss_avg}')


    return (train_losses, val_losses, recon_losses, kld_losses, effective_kld_losses, 
            beta_values, early_stopping.stopped_epoch, gradient_history, lr_evolution, separate_recon_losses)

def run_linear_model_and_collect_outputs(model, dataloader, device, n_components):
    """
    Esegue il modello su un DataLoader e restituisce gli output, mean e logvar, 
    e i dati originali in forma di array numpy.

    Parametri:
    - model: Modello PyTorch da eseguire
    - dataloader: DataLoader con i dati di input
    - device: Dispositivo (CPU o GPU)

    Ritorna:
    - true_data: Array numpy con i dati originali
    - predicted_data: Array numpy con gli output del modello
    - mean_vectors: Array numpy con i vettori mean
    - logvar_vectors: Array numpy con i vettori logvar
    """
    model.eval()  # Modalità eval
    true_data = []
    predicted_data = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(device)
            
            # Inferenza del modello
            outputs = model(inputs)  # Supponendo che il modello restituisca anche mean e logvar
            
            # Aggiungi i risultati a liste
            true_data.append(inputs.cpu().numpy())

            # Usa view dinamico: invece di fissare la dimensione con 'r', usiamo -1 per calcolarla automaticamente.
            # Se sai per certo che outputs.numel() == batch_size * n_components * r, potresti usare:
            # outputs = outputs.view(outputs.size(0), n_components, r)
            # Altrimenti, per essere flessibile:
            outputs = outputs.view(outputs.size(0), n_components, -1)
            predicted_data.append(outputs.cpu().numpy())

    # Concatena tutti i batch in array numpy
    true_data = np.concatenate(true_data, axis=0)
    predicted_data = np.concatenate(predicted_data, axis=0)

    return true_data, predicted_data,

def run_model_and_collect_outputs(model, dataloader, device, n_components, r):
    """
    Esegue il modello sui dati nel dataloader e raccoglie gli output,
    assicurando che i predicted data abbiano la stessa forma dei target definiti in EmbeddedDatasetforDecoder:
       (n_samples, n_components, r)
    Inoltre, restituisce anche i mean_vectors e log_var.
    
    Parameters:
       - model: il modello PyTorch.
       - dataloader: DataLoader contenente (input, target).
       - device: device su cui eseguire il modello.
       - n_components: numero di componenti principali.
       - r: dimensione ridotta dell'embedding (attesa).
    
    Returns:
       - predicted_data: tensore di forma (n_samples, n_components, r_inferito)
       - mean_vectors: tensore di forma (n_samples, n_components, r_inferito) (o la forma prevista dal modello)
       - log_var: tensore di forma (n_samples, n_components, r_inferito)
    """
    true_data = []
    target_data = []
    predicted_data = []
    mean_vectors = []
    logvar_vectors = []
    
    model.eval()
    model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            # I batch sono (input, target); usiamo solo gli input
            inputs, targets = batch
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            outputs, mean, logvar = model(inputs)
            
            # Se outputs è una tupla, prendi il primo elemento
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            # Sposta su CPU e detach
            # Aggiungi i risultati a liste
            true_data.append(inputs.cpu().detach().numpy())

            # Usa view dinamico: invece di fissare la dimensione con 'r', usiamo -1 per calcolarla automaticamente.
            # Se sai per certo che outputs.numel() == batch_size * n_components * r, potresti usare:
            # outputs = outputs.view(outputs.size(0), n_components, r)
            # Altrimenti, per essere flessibile:
            outputs = outputs.view(outputs.size(0), n_components, -1)
            targets = targets.view(targets.size(0), n_components, -1)

            target_data.append(targets.cpu().numpy())
            predicted_data.append(outputs.cpu().numpy())
            mean_vectors.append(mean.cpu().numpy())
            logvar_vectors.append(logvar.cpu().numpy())
                        
    # Concatena tutti i batch in array numpy
    true_data = np.concatenate(true_data, axis=0)
    target_data = np.concatenate(target_data, axis=0)
    predicted_data = np.concatenate(predicted_data, axis=0)
    mean_vectors = np.concatenate(mean_vectors, axis=0)
    logvar_vectors = np.concatenate(logvar_vectors, axis=0)
    

    return true_data, target_data, predicted_data, mean_vectors, logvar_vectors

def recon_fmri_pclinear(device, epochs, train_loader, val_loader, 
                model, optimizer, scheduler, scheduler_config,
                r, n_components,
                early_stopping_params=None):
    # Early stopping
    early_stopping = EarlyStopping(**early_stopping_params) if early_stopping_params else None

    # Liste per memorizzare le perdite
    train_losses = []
    val_losses = []
    recon_losses = []         # loss di ricostruzione totale
    separate_recon_losses = []  # lista per le loss per ciascun componente (lista di liste)
    lr_evolution = []
    

    for epoch in range(epochs):
        model.train()
        model.to(device)

        epoch_loss = 0
        recon_loss_epoch = 0
        # Inizializzo la lista che accumulerà la loss per ciascun componente per questa epoca
        separate_recon_loss = [0] * n_components

        for batch in train_loader:
            input, target = batch  # input: (batch_size, r), target: (batch_size, n_components, r)
            input = input.float().to(device)
            target = target.float().to(device)
            optimizer.zero_grad()

            # Determino il batch size
            batch_size = input.size(0)

            # Calcola l'output della rete
            recon = model(input)
            # Effettua il reshape: da (batch_size, n_components * r) a (batch_size, n_components, r)
            recon_reshaped = recon.view(batch_size, n_components, r)
            
            # Calcolo della loss di ricostruzione per ciascun componente (sommo sulle righe e normalizzo per il batch)
            total_recon_loss = 0.0
            for i in range(n_components):
                recon_loss_component = torch.nn.functional.mse_loss(
                    recon_reshaped[:, i, :],
                    target[:, i, :],
                    reduction='sum'
                ) / batch_size
                separate_recon_loss[i] += recon_loss_component.item()
                total_recon_loss += recon_loss_component

            batch_loss = total_recon_loss 
            batch_loss.backward()

            # Clip dei gradienti
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            

            # Accesso al learning rate
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            lr_evolution.append(current_lr)


            optimizer.step()

            epoch_loss += batch_loss.item()
            recon_loss_epoch += total_recon_loss.item()

        # Calcolo delle medie per epoca, usando il numero totale di campioni
        n_total = len(train_loader.dataset)
        epoch_loss /= n_total
        recon_loss_epoch /= n_total
        # Per le loss separate, dividiamo ciascuna per n_total
        separate_recon_loss_avg = [loss_val / n_total for loss_val in separate_recon_loss]

        train_losses.append(epoch_loss)
        recon_losses.append(recon_loss_epoch)
        separate_recon_losses.append(separate_recon_loss_avg)
               
        # VALIDATION
        model.eval()
        model.to(device)
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                input, target = batch  # (batch_size, r) e (batch_size, n_components, r)
                input = input.float().to(device)
                target = target.float().to(device)
                recon = model(input)
                batch_size_val = input.size(0)
                recon_reshaped = recon.view(batch_size_val, n_components, r)
                total_recon_loss = 0.0
                for i in range(n_components):
                    val_loss_component = torch.nn.functional.mse_loss(
                        recon_reshaped[:, i, :],
                        target[:, i, :],
                        reduction='sum'
                    ) / batch_size_val
                    total_recon_loss += val_loss_component

        val_loss /= len(val_loader.dataset)
        val_losses.append(val_loss)

        if scheduler_config["type"] == "ReduceLROnPlateau":
            scheduler.step(val_loss)
        else:
            if scheduler_config["type"] == "None":
                pass
            else:
                scheduler.step()

        # Early Stopping
        if early_stopping:
            early_stopping(val_loss, epoch)
            if early_stopping.early_stop:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        # Stampa i valori medi per l'epoca
        print(f'Epoch [{epoch+1}/{epochs}], Rec Loss: {recon_loss_epoch:.4f}, , lr = {current_lr:.6f}')
        print(f'Separate Recon Loss per component: {separate_recon_loss_avg}')


    return (train_losses, val_losses, recon_losses,  early_stopping.stopped_epoch, lr_evolution, separate_recon_losses)


















