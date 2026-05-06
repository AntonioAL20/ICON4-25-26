import numpy as np

class SemanticFeatureExtractor:
    """
    Soddisfa il requisito di Apprendimento "ML + OntoBK" (Ontology Background Knowledge).
    Trasforma un dataset grezzo arricchendolo con le deduzioni del reasoner semantico.
    """
    def __init__(self, reasoner):
        self.reasoner = reasoner
        
        # Mappatura delle feature per il vettore ML
        self.feature_names = [
            # Feature Raw (Bag of Words)
            "raw_TempAlta", "raw_Ticchettio", "raw_NoFilamento",
            # Feature Semantiche (Background Knowledge dal Reasoner)
            "onto_AllarmeTermico", "onto_AllarmeMeccanico"
        ]

    def extract_features(self, raw_symptoms_list):
        """
        Converte una lista di sintomi in un vettore di feature numerico.
        Combina la presenza del sintomo grezzo con l'appartenenza alle classi
        inferite dall'ontologia.
        """
        feature_vector = {name: 0 for name in self.feature_names}
        
        for symptom in raw_symptoms_list:
            # 1. Feature Raw
            if symptom == "TempTroppoAlta": feature_vector["raw_TempAlta"] = 1
            if symptom == "TicchettioEstrusore": feature_vector["raw_Ticchettio"] = 1
            if symptom == "FilamentoNonEsce": feature_vector["raw_NoFilamento"] = 1
            
            # 2. Semantic Lifting (Estrazione classi inferite dal Reasoner)
            inferred_classes = self.reasoner.get_inferred_classes_for_symptom(symptom)
            
            if "AllarmeTermico" in inferred_classes:
                feature_vector["onto_AllarmeTermico"] = 1
            if "AllarmeMeccanico" in inferred_classes:
                feature_vector["onto_AllarmeMeccanico"] = 1
                
        return [feature_vector[f] for f in self.feature_names]

    def prepare_dataset(self, raw_dataset):
        """
        Converte un intero dataset di stringhe in matrici X (feature) e y (target) per scikit-learn.
        """
        print(f"[OntoBK] Arricchimento di {len(raw_dataset)} istanze con Background Knowledge...")
        X = []
        y = []
        for symptoms, target_fault in raw_dataset:
            X.append(self.extract_features(symptoms))
            y.append(target_fault)
            
        return np.array(X), np.array(y)
