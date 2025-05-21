import json

def load_configs(file_path):
    """
    Carica le configurazioni dei modelli da un file JSON.
    
    Parametri:
    - file_path: percorso del file JSON.
    
    Ritorna:
    - Dizionario con tutte le configurazioni dei modelli.
    """
    with open(file_path, "r") as f:
        return json.load(f)

def save_configs(configs, file_path):
    """
    Salva le configurazioni dei modelli in un file JSON.
    
    Parametri:
    - configs: dizionario con le configurazioni dei modelli.
    - file_path: percorso del file JSON.
    """
    with open(file_path, "w") as f:
        json.dump(configs, f, indent=4)

def add_config(config, model_name, file_path):
    """
    Aggiunge una nuova configurazione al file.
    
    Parametri:
    - config: configurazione del modello da aggiungere (deve includere input_variable).
    - model_name: nome univoco del modello.
    - file_path: percorso del file JSON.
    """
    configs = load_configs(file_path)
    
    # Controllo per evitare duplicati
    if model_name in configs:
        print(f"Model {model_name} already exists in the database. Overwriting...")
        
    
    configs[model_name] = config
    save_configs(configs, file_path)

def find_model(file_path, **criteria):
    """
    Cerca un modello nel file basandosi sui criteri forniti.
    
    Parametri:
    - file_path: percorso del file JSON.
    - criteria: parametri per filtrare i modelli (incluso input_variable, se necessario).
    
    Ritorna:
    - Lista di modelli che soddisfano i criteri.
    """
    configs = load_configs(file_path)
    results = []
    
    for model_name, config in configs.items():
        # Verifica che tutti i criteri siano soddisfatti
        if all(config.get(key) == value for key, value in criteria.items()):
            results.append({model_name: config})
    
    return results

def get_model_config_by_name(file_path, model_name):
    """
    Carica i parametri di un modello a partire dal nome.

    Parametri:
    - file_path: percorso del file JSON.
    - model_name: nome del modello da cercare.

    Ritorna:
    - Configurazione del modello se trovato, altrimenti None.
    """
    configs = load_configs(file_path)
    
    return configs.get(model_name, None)


