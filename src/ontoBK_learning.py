import numpy as np

class SemanticFeatureExtractor:
    def __init__(self, reasoner, df_raw):
        self.reasoner = reasoner
        self.raw_symptoms_vocab = set()
        self.semantic_classes_vocab = set()
        self.materiali_vocab = set()
        
        for index, row in df_raw.iterrows():
            symptoms = row['sintomi'].split(';')
            for s in symptoms:
                self.raw_symptoms_vocab.add(s)
                
                inferred = self.reasoner.get_inferred_classes_for_symptom(s)
                for inf_cls in inferred:
                    self.semantic_classes_vocab.add(inf_cls)
            self.materiali_vocab.add(row['materiale'])
                    
        self.raw_features = sorted(list(self.raw_symptoms_vocab))
        self.semantic_features = sorted(list(self.semantic_classes_vocab))
        self.materiali_features = sorted(list(self.materiali_vocab))
        
        # Spazio Vettoriale Esteso comprensivo dei sensori fisici e delle nuove astrazioni causali
        self.feature_names = (
            ["temp_estrusore", "temp_piatto", "tempo_stampa", "velocita_stampa", "umidita_ambientale", "usura_motore"] + 
            [f"mat_{m}" for m in self.materiali_features] +
            [f"raw_{s}" for s in self.raw_features] + 
            [f"onto_{c}" for c in self.semantic_features]
        )
        print(f"[OntoBK] Spazio Vettoriale Multimodale: {len(self.feature_names)} features totali")

    def extract_features(self, temp_estr, temp_piatto, tempo_stampa, vel_stampa, umidita, usura, materiale, raw_symptoms_list):
        vec = {name: 0 for name in self.feature_names}
        
        # 1. Feature Numeriche Continue (Sensori e Ambiente)
        vec["temp_estrusore"] = float(temp_estr)
        vec["temp_piatto"] = float(temp_piatto)
        vec["tempo_stampa"] = float(tempo_stampa)
        vec["velocita_stampa"] = float(vel_stampa)
        vec["umidita_ambientale"] = float(umidita)
        vec["usura_motore"] = float(usura)
        
        # 2. Feature Categoriche Nominali (One-Hot Encoding del materiale)
        mat_key = f"mat_{materiale}"
        if mat_key in vec:
            vec[mat_key] = 1
            
        # 3. NLP Grezzo e Semantic Lifting Ontologico (Sfrutta le nuove classi dedotte dal ragionamento)
        for symptom in raw_symptoms_list:
            raw_key = f"raw_{symptom}"
            if raw_key in vec:
                vec[raw_key] = 1
            
            inferred_classes = self.reasoner.get_inferred_classes_for_symptom(symptom)
            for cls in inferred_classes:
                onto_key = f"onto_{cls}"
                if onto_key in vec:
                    vec[onto_key] = 1
                
        return [vec[f] for f in self.feature_names]

    def process_dataset(self, dataframe):
        X, y = [], []
        for index, row in dataframe.iterrows():
            symptoms_list = row['sintomi'].split(';')
            features = self.extract_features(
                row['temperatura_estrusore'],
                row['temperatura_piatto'],
                row['tempo_stampa_ore'],
                row['velocita_stampa'],
                row['umidita_ambientale'],
                row['usura_motore'],
                row['materiale'],
                symptoms_list
            )
            X.append(features)
            y.append(row['guasto'])
        return np.array(X), np.array(y)