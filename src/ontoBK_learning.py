import numpy as np

class SemanticFeatureExtractor:
    def __init__(self, reasoner, df_raw):
        self.reasoner = reasoner
        self.raw_symptoms_vocab = set()
        self.semantic_classes_vocab = set()
        self.materiali_vocab = set()
        
        # 1. Analizziamo il dataset per i vocabolari
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
        
        # Costruzione dinamica dell'ordine delle colonne per il ML
        self.feature_names = (
            ["temp_estrusore", "temp_piatto", "tempo_stampa"] + 
            [f"mat_{m}" for m in self.materiali_features] +
            [f"raw_{s}" for s in self.raw_features] + 
            [f"onto_{c}" for c in self.semantic_features]
        )
        print(f"[OntoBK] Spazio Vettoriale Esteso: {len(self.feature_names)} features totali")

    def extract_features(self, temp_estr, temp_piatto, tempo_stampa, materiale, raw_symptoms_list):
        vec = {name: 0 for name in self.feature_names}
        
        # 1. Feature Numeriche (Sensori)
        vec["temp_estrusore"] = float(temp_estr)
        vec["temp_piatto"] = float(temp_piatto)
        vec["tempo_stampa"] = float(tempo_stampa)
        
        # 2. Categoriche (One-Hot Encoding del materiale)
        mat_key = f"mat_{materiale}"
        if mat_key in vec:
            vec[mat_key] = 1
            
        # 3. NLP Grezzo e Semantic Lifting Ontologico
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
                row['materiale'],
                symptoms_list
            )
            X.append(features)
            y.append(row['guasto'])
        return np.array(X), np.array(y)