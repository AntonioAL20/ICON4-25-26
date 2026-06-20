import os
import random
import pandas as pd

def generate_large_dataset(num_samples=1500):
    """
    Genera un dataset CSV sintetico ad altissima complessità.
    Aggiunte feature ambientali, cinematiche e di usura per rendere 
    le Reti Neurali e le SVM essenziali rispetto ai modelli lineari.
    """
    guasti_rules = {
        "GuastoTermico": ["TempTroppoAlta", "TempInstabile", "PiattoFreddo", "VentolaRumorosa"],
        "GuastoMeccanico": ["LayerSpostati", "RumoreCinghia", "HomeFallito", "TicchettioEstrusore", "SottoEstrusione"],
        "ProblemaElettrico": ["OdoreBruciato", "SchermoNero", "HomeFallito", "TempInstabile"],
        "ProblemaAdesione": ["Warping", "PiattoFreddo", "FilamentoNonEsce"]
    }
    
    # Parco materiali notevolmente espanso
    materiali = ["PLA", "ABS", "PETG", "TPU", "NYLON", "ASA", "PC"]
    
    data = []
    
    for _ in range(num_samples):
        guasto_target = random.choice(list(guasti_rules.keys()))
        sintomi_possibili = guasti_rules[guasto_target]
        
        num_sintomi = random.randint(1, min(3, len(sintomi_possibili)))
        sintomi_scelti = random.sample(sintomi_possibili, num_sintomi)
        
        if random.random() < 0.15: # Aumentato il rumore stocastico al 15%
            sintomo_rumore = random.choice(["VentolaRumorosa", "SchermoNero", "TicchettioEstrusore"])
            if sintomo_rumore not in sintomi_scelti:
                sintomi_scelti.append(sintomo_rumore)
                
        sintomi_str = ";".join(sintomi_scelti)
        materiale = random.choice(materiali)
        
        # Generazione parametrica complessa
        # Parametri Base
        velocita = random.randint(40, 70)
        umidita = random.randint(30, 50)
        usura = random.randint(100, 1500)
        
        if guasto_target == "GuastoTermico":
            temp_estrusore = random.randint(240, 290) if random.random() > 0.5 else random.randint(150, 180)
            temp_piatto = random.randint(20, 120)
            usura = random.randint(500, 4000) # Spesso causato da termistori vecchi
        elif guasto_target == "GuastoMeccanico":
            velocita = random.randint(80, 180) # Alte velocità causano perdita di passi (Layer Spostati)
            usura = random.randint(2000, 6000) # Altissima usura meccanica
            temp_estrusore = random.randint(200, 250)
            temp_piatto = random.randint(50, 90)
        elif guasto_target == "ProblemaAdesione":
            temp_piatto = random.randint(20, 50) # Piatto sempre troppo freddo
            temp_estrusore = random.randint(190, 240)
            velocita = random.randint(80, 120) # Primo layer stampato troppo veloce
        else: # ProblemaElettrico
            temp_estrusore = random.randint(200, 260)
            temp_piatto = random.randint(60, 110)
            umidita = random.randint(60, 90) # Alta umidità facilita i cortocircuiti
            
        # Condizioni specifiche per materiali igroscopici
        if materiale in ["NYLON", "PETG", "TPU"] and guasto_target == "ProblemaAdesione":
            umidita = random.randint(60, 95) # Filamento umido
            
        tempo_stampa_ore = round(random.uniform(0.5, 96.0), 1)
        
        data.append({
            "materiale": materiale,
            "temperatura_estrusore": temp_estrusore,
            "temperatura_piatto": temp_piatto,
            "velocita_stampa": velocita,
            "umidita_ambientale": umidita,
            "usura_motore": usura,
            "tempo_stampa_ore": tempo_stampa_ore,
            "sintomi": sintomi_str, 
            "guasto": guasto_target
        })
        
    df = pd.DataFrame(data)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "dataset_guasti.csv")
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"Generato dataset multi-variato (Sensori Avanzati) di {num_samples} istanze in {csv_path}")

if __name__ == "__main__":
    generate_large_dataset()