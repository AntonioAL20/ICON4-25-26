import os
import random
import pandas as pd

def generate_large_dataset(num_samples=1000):
    """
    Genera un dataset CSV sintetico avanzato.
    Oltre ai sintomi NLP, genera feature numeriche e categoriche di "sensori" 
    per simulare una reale stampante 3D e testare reti neurali e SVM.
    """
    guasti_rules = {
        "GuastoTermico": ["TempTroppoAlta", "TempInstabile", "PiattoFreddo", "VentolaRumorosa"],
        "GuastoMeccanico": ["LayerSpostati", "RumoreCinghia", "HomeFallito", "TicchettioEstrusore", "SottoEstrusione"],
        "ProblemaElettrico": ["OdoreBruciato", "SchermoNero", "HomeFallito", "TempInstabile"],
        "ProblemaAdesione": ["Warping", "PiattoFreddo", "FilamentoNonEsce"]
    }
    materiali = ["PLA", "ABS", "PETG", "TPU"]
    
    data = []
    
    for _ in range(num_samples):
        guasto_target = random.choice(list(guasti_rules.keys()))
        sintomi_possibili = guasti_rules[guasto_target]
        
        num_sintomi = random.randint(1, min(3, len(sintomi_possibili)))
        sintomi_scelti = random.sample(sintomi_possibili, num_sintomi)
        
        if random.random() < 0.10:
            sintomo_rumore = random.choice(["VentolaRumorosa", "SchermoNero", "TicchettioEstrusore"])
            if sintomo_rumore not in sintomi_scelti:
                sintomi_scelti.append(sintomo_rumore)
                
        sintomi_str = ";".join(sintomi_scelti)
        materiale = random.choice(materiali)
        
        # Generazione parametrica di sensori basata sul target per addestrare i modelli ML
        if guasto_target == "GuastoTermico":
            temp_estrusore = random.randint(240, 270) if random.random() > 0.5 else random.randint(160, 180)
            temp_piatto = random.randint(20, 110)
        elif guasto_target == "ProblemaAdesione":
            temp_piatto = random.randint(20, 50) # Piatto troppo freddo
            temp_estrusore = random.randint(190, 230)
        else:
            temp_estrusore = random.randint(195, 215) if materiale in ["PLA", "TPU"] else random.randint(235, 255)
            temp_piatto = random.randint(50, 65) if materiale in ["PLA", "TPU"] else random.randint(80, 105)
            
        tempo_stampa_ore = round(random.uniform(0.5, 72.0), 1)
        
        data.append({
            "materiale": materiale,
            "temperatura_estrusore": temp_estrusore,
            "temperatura_piatto": temp_piatto,
            "tempo_stampa_ore": tempo_stampa_ore,
            "sintomi": sintomi_str, 
            "guasto": guasto_target
        })
        
    df = pd.DataFrame(data)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "dataset_guasti.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generato dataset multi-variato di {num_samples} istanze in {csv_path}")

if __name__ == "__main__":
    generate_large_dataset()