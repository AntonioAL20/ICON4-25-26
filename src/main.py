import os
import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.preprocessing import StandardScaler

# IMPORT DEI 3 ALGORITMI
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import make_scorer, accuracy_score, f1_score

# Import moduli locali
from ontology_core import build_advanced_ontology
from semantic_reasoner import DiagnosticReasoner
from ontoBK_learning import SemanticFeatureExtractor

warnings.filterwarnings('ignore')

def load_and_augment_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "dataset_guasti.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}. Esegui prima generate_dataset.py")
        
    df = pd.read_csv(csv_path)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def evaluate_ml_model(X, y):
    """Valutazione rigorosa con K-Fold comparando 3 architetture ML."""
    print("\n[Valutazione] Avvio Repeated K-Fold Cross Validation (5 Fold, 5 Ripetizioni)...")
    
    # Standardizzazione obbligatoria per SVM e Reti Neurali
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    models = {
        "Random Forest (Ensemble)": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42),
        "Support Vector Machine (RBF)": SVC(kernel='rbf', probability=True, random_state=42),
        "Rete Neurale (MLP)": MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42)
    }
    
    rkf = RepeatedKFold(n_splits=5, n_repeats=5, random_state=42)
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    print("\n" + "="*75)
    print(" COMPARATIVA PRESTAZIONALE MODELLI (ML + OntoBK)")
    print("="*75)
    
    for name, clf in models.items():
        scores = cross_validate(clf, X_scaled, y, scoring=scoring, cv=rkf, n_jobs=-1)
        print(f" Modello: {name}")
        print(f"  - Accuracy: {np.mean(scores['test_accuracy']):.4f} ± {np.std(scores['test_accuracy']):.4f}")
        print(f"  - F1-Score: {np.mean(scores['test_f1']):.4f} ± {np.std(scores['test_f1']):.4f}\n")
    print("="*75)

def interactive_diagnostic_shell(model, scaler, extractor, X_dataset, y_dataset):
    print("\n" + "*"*75)
    print(" MODALITA' DIAGNOSTICA AVANZATA E APPRENDIMENTO ONLINE")
    print(" (Digita 'esci' alla richiesta del materiale per terminare)")
    print("*"*75)
    
    print(f"Materiali supportati: {', '.join(extractor.materiali_features)}")
    print(f"Sintomi ontologici supportati: {', '.join(extractor.raw_features)}\n")
    
    X_list = X_dataset.tolist()
    y_list = y_dataset.tolist()
    
    while True:
        try:
            mat = input("Inserisci Materiale (PLA, ABS, PETG...): ").strip()
            if mat.lower() in ['esci', 'q']: break
            
            t_est = float(input("Temperatura Estrusore (°C): ").strip())
            t_bed = float(input("Temperatura Piatto (°C): ").strip())
            ore = float(input("Ore di stampa al momento del guasto: ").strip())
            
            sint = input("Inserisci i sintomi testuali (separati da virgola): ").strip()
            symptoms = [s.strip() for s in sint.split(',')] if sint else []
            
            # Creazione vettore
            raw_features = extractor.extract_features(t_est, t_bed, ore, mat.upper(), symptoms)
            
            # Lo scaler necessita di un array 2D
            scaled_features = scaler.transform([raw_features])
            
            prediction = model.predict(scaled_features)[0]
            probabilities = model.predict_proba(scaled_features)[0]
            
            print("\n ---> DIAGNOSI DELLA RETE NEURALE: ", prediction)
            prob_dict = {cls: prob for cls, prob in zip(model.classes_, probabilities) if prob > 0}
            for cls, prob in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True):
                print(f"       - {cls}: {prob*100:.1f}%")

            print("\n--- Feedback & Online Learning ---")
            feedback = input("Qual era il guasto effettivo? (es. GuastoTermico, premi Invio per saltare): ").strip()
            if feedback:
                X_list.append(raw_features)
                y_list.append(feedback)
                # Ri-addestriamo aggiornando prima lo scaler con i nuovi dati
                X_new_scaled = scaler.fit_transform(np.array(X_list))
                model.fit(X_new_scaled, np.array(y_list))
                print(" Rete Neurale aggiornata con successo!")
                
        except ValueError:
            print("[!] Errore di inserimento. Assicurati di inserire numeri per le temperature e le ore.\n")

def main():
    print("=== AVVIO SISTEMA DIAGNOSTICO MULTIMODALE ICon ===\n")
    
    print("[1/5] Costruzione dell'Ontologia OWL 2.0...")
    onto, save_path = build_advanced_ontology()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    onto.save(file=save_path, format="rdfxml")
    
    print("\n[2/5] Materializzazione assiomi (HermiT Reasoner)...")
    reasoner = DiagnosticReasoner(onto)
    reasoner.run_inference()
    
    print("\n[3/5] Lettura CSV e Generazione Vettoriale (Sensori + OntoBK)...")
    df_raw = load_and_augment_data()
    extractor = SemanticFeatureExtractor(reasoner, df_raw)
    X, y = extractor.process_dataset(df_raw)
    
    print("\n[4/5] Esecuzione Comparativa Architetture ML...")
    evaluate_ml_model(X, y)
    
    print("\n[5/5] Addestramento modello di punta (Rete Neurale) per la Shell...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Scegliamo la rete neurale come modello finale per l'interfaccia interattiva
    final_model = MLPClassifier(hidden_layer_sizes=(50, 25), max_iter=300, random_state=42)
    final_model.fit(X_scaled, y) 
    
    interactive_diagnostic_shell(final_model, scaler, extractor, X, y)

if __name__ == "__main__":
    main()