import numpy as np

class SemanticFeatureExtractor:
    def __init__(self, reasoner, df_raw):
        self.reasoner = reasoner
        self.raw_symptoms_vocab = set()
        self.semantic_classes_vocab = set()
        
        # 1. fase di analisi del dataset per trovare tutti i sintomi unici
        for index, row in df_raw.iterrows():
            symptoms = row['sintomi'].split(';')
            for s in symptoms:
                self.raw_symptoms_vocab.add(s)
                
                # 2. il reasoner viene interrogato per trovare le superclassi
                inferred = self.reasoner.get_inferred_classes_for_symptom(s)
                for inf_cls in inferred:
                    self.semantic_classes_vocab.add(inf_cls)
                    
    
        self.raw_features = sorted(list(self.raw_symptoms_vocab))
        self.semantic_features = sorted(list(self.semantic_classes_vocab))
        self.feature_names = [f"raw_{s}" for s in self.raw_features] + [f"onto_{c}" for c in self.semantic_features]
        print(f"[OntoBK] Vocabolario dinamico costruito: {len(self.raw_features)} RAW + {len(self.semantic_features)} SEMANTICHE")

    def extract_features(self, raw_symptoms_list):
        # Vettore inizializzato a zero
        vec = {name: 0 for name in self.feature_names}
        
        for symptom in raw_symptoms_list:
            raw_key = f"raw_{symptom}"
            if raw_key in vec:
                vec[raw_key] = 1
            
            # Inferenza semantica
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
            X.append(self.extract_features(symptoms_list))
            y.append(row['guasto'])
        return np.array(X), np.array(y)
