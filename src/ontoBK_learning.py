import numpy as np

class SemanticFeatureExtractor:
    """
    Implementa l'approccio OntoBK (Ontology Background Knowledge).
    Arricchisce le features di base usando le deduzioni dell'Ontologia.
    """
    def __init__(self, reasoner):
        self.reasoner = reasoner
        
        # Le features del nostro modello Machine Learning
        self.feature_names = [
            # 1. Feature Raw (Bag of Words)
            "raw_TempAlta", "raw_Ticchettio", "raw_NoFilamento", 
            "raw_PiattoFreddo", "raw_Vibrazione", "raw_Bruciato",
            # 2. Feature Semantiche Inferite (La "Background Knowledge")
            "onto_AllarmeTermico", "onto_AllarmeMeccanico", "onto_AllarmeElettrico"
        ]

    def extract_features(self, raw_symptoms_list):
        """Converte una lista di sintomi stringa in un feature vector."""
        vec = {name: 0 for name in self.feature_names}
        
        for symptom in raw_symptoms_list:
            # Popolamento feature RAW
            if symptom == "TempTroppoAlta": vec["raw_TempAlta"] = 1
            if symptom == "TicchettioEstrusore": vec["raw_Ticchettio"] = 1
            if symptom == "FilamentoNonEsce": vec["raw_NoFilamento"] = 1
            if symptom == "PiattoFreddo": vec["raw_PiattoFreddo"] = 1
            if symptom == "VibrazioneAnomala": vec["raw_Vibrazione"] = 1
            if symptom == "OdoreBruciato": vec["raw_Bruciato"] = 1
            
            # Popolamento feature SEMANTICHE (Query al Reasoner)
            inferred_classes = self.reasoner.get_inferred_classes_for_symptom(symptom)
            
            if "AllarmeTermico" in inferred_classes: vec["onto_AllarmeTermico"] = 1
            if "AllarmeMeccanico" in inferred_classes: vec["onto_AllarmeMeccanico"] = 1
            if "AllarmeElettrico" in inferred_classes: vec["onto_AllarmeElettrico"] = 1
                
        return [vec[f] for f in self.feature_names]

    def process_dataset(self, dataframe):
        """Processa un intero dataframe Pandas in X (features) e y (target)."""
        X = []
        y = []
        
        for index, row in dataframe.iterrows():
            # I sintomi nel CSV sono separati dal punto e virgola
            symptoms_list = row['sintomi'].split(';')
            feature_vector = self.extract_features(symptoms_list)
            
            X.append(feature_vector)
            y.append(row['guasto'])
            
        return np.array(X), np.array(y)