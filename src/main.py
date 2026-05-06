import os
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold, cross_validate
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
import warnings

# Import moduli locali
from ontology_core import build_advanced_ontology
from semantic_reasoner import DiagnosticReasoner
from ontoBK_learning import SemanticFeatureExtractor

warnings.filterwarnings('ignore')

def load_and_augment_data():
    """Legge il CSV base e lo espande per avere varianza statistica per il test."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, "..", "data", "dataset_guasti.csv")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"File non trovato: {csv_path}")
        
    df = pd.read_csv(csv_path)
    # Moltiplichiamo il dataset aggiungendo leggero rumore/varianza per simulare un dataset corposo
    df_ampliato = pd.concat([df]*20, ignore_index=True)
    df_ampliato = df_ampliato.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_ampliato

def evaluate_ml_model(X, y):
    """Valutazione rigorosa con K-Fold come richiesto dalle specifiche."""
    print("\n[Valutazione] Avvio Repeated K-Fold Cross Validation (5 Fold, 10 Ripetizioni)...")
    
    clf = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=42)
    
    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='macro', zero_division=0),
        'recall': make_scorer(recall_score, average='macro', zero_division=0),
        'f1': make_scorer(f1_score, average='macro', zero_division=0)
    }
    
    scores = cross_validate(clf, X, y, scoring=scoring, cv=rkf, n_jobs=-1)
    
    print("\n" + "="*60)
    print(" RISULTATI STATISTICI DEL SISTEMA KBS (ML + OntoBK)")
    print("="*60)
    print(f"Accuracy:  {np.mean(scores['test_accuracy']):.4f} ± {np.std(scores['test_accuracy']):.4f}")
    print(f"Precision: {np.mean(scores['test_precision']):.4f} ± {np.std(scores['test_precision']):.4f}")
    print(f"Recall:    {np.mean(scores['test_recall']):.4f} ± {np.std(scores['test_recall']):.4f}")
    print(f"F1-Score:  {np.mean(scores['test_f1']):.4f} ± {np.std(scores['test_f1']):.4f}")
    print("="*60)

def interactive_diagnostic_shell(model, extractor, X_dataset, y_dataset):
    """
    Shell interattiva che implementa l'Apprendimento Online (Feedback loop).
    Simula l'interfaccia utente del KBS con addestramento in tempo reale.
    """
    print("\n" + "*"*65)
    print(" MODALITA' DIAGNOSTICA INTERATTIVA E APPRENDIMENTO ATTIVATI")
    print(" (Digita 'esci' per terminare il programma)")
    print("*"*65)
    
    # Elenchiamo in modo chiaro i sintomi testuali esatti attesi
    sintomi_validi = ["TempTroppoAlta", "TicchettioEstrusore", "FilamentoNonEsce", 
                      "PiattoFreddo", "VibrazioneAnomala", "OdoreBruciato"]
    print(f"Sintomi supportati (usa ESATTAMENTE questi nomi):\n{', '.join(sintomi_validi)}\n")
    
    # Convertiamo i dataset numpy in liste per poter fare append dinamicamente
    X_list = X_dataset.tolist()
    y_list = y_dataset.tolist()
    
    while True:
        user_input = input("\nInserisci i sintomi osservati (separati da virgola): ").strip()
        
        if user_input.lower() in ['esci', 'exit', 'quit', 'q']:
            print("Chiusura del sistema diagnostico. Arrivederci!")
            break
            
        if not user_input:
            continue
            
        # Pulisci input
        symptoms = [s.strip() for s in user_input.split(',')]
        
        # 1. ESTENSIONI SEMANTICHE (Il reasoner deduce nuove classi in background)
        features = extractor.extract_features(symptoms)
        
        # Controllo anti-errore di battitura
        if sum(features) == 0:
            print(" [!] Nessun sintomo riconosciuto. Assicurati di usare i nomi esatti dalla lista sopra.")
            continue
        
        # 2. PREDIZIONE ML (La RandomForest valuta i sintomi raw + le deduzioni semantiche)
        prediction = model.predict([features])[0]
        probabilities = model.predict_proba([features])[0]
        classes = model.classes_
        
        print("\n ---> DIAGNOSI DEL SISTEMA: ", prediction)
        print("      Dettaglio probabilità (basate sull'esperienza appresa):")
        
        # Stampa le probabilità ordinate
        prob_dict = {cls: prob for cls, prob in zip(classes, probabilities) if prob > 0}
        for cls, prob in sorted(prob_dict.items(), key=lambda item: item[1], reverse=True):
            print(f"       - {cls}: {prob*100:.1f}%")

        # 3. APPRENDIMENTO ONLINE (Il Feedback dell'utente)
        print("\n--- Fase di Apprendimento ---")
        feedback = input("Qual era il guasto effettivo? (es. GuastoElettrico, o premi Invio per saltare): ").strip()
        
        if feedback:
            print(" [Apprendimento] Aggiornamento del modello con la nuova esperienza...")
            
            # Aggiungiamo la nuova esperienza (feature + guasto reale) al nostro dataset in memoria
            X_list.append(features)
            y_list.append(feedback)
            
            # Riaddestriamo il modello al volo (il dataset è piccolo, è un'operazione istantanea)
            model.fit(np.array(X_list), np.array(y_list))
            print(" Modello aggiornato con successo! Le probabilità si adatteranno alla prossima diagnosi.")

def main():
    print("=== AVVIO SISTEMA DIAGNOSTICO ICon (OntoBK) ===\n")
    
    # 1. Costruzione Ontologia in Memoria
    print("[1/5] Costruzione dell'Ontologia OWL 2.0...")
    onto, save_path = build_advanced_ontology()
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    onto.save(file=save_path, format="rdfxml")
    
    # 2. Avvio Ragionamento Semantico
    print("\n[2/5] Materializzazione assiomi (HermiT Reasoner)...")
    reasoner = DiagnosticReasoner(onto)
    reasoner.run_inference()
    
    # 3. Lettura e Arricchimento Dataset
    print("\n[3/5] Lettura CSV e Semantic Feature Extraction...")
    df_raw = load_and_augment_data()
    extractor = SemanticFeatureExtractor(reasoner)
    X, y = extractor.process_dataset(df_raw)
    
    # 4. Machine Learning & Valutazione (K-Fold statitistico)
    print("\n[4/5] Valutazione Statistica Modello...")
    evaluate_ml_model(X, y)
    
    # 5. ADDESTRAMENTO FINALE E AVVIO SHELL INTERATTIVA
    print("\n[5/5] Addestramento modello finale e avvio interfaccia...")
    final_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    final_model.fit(X, y) # Addestriamo il modello su tutto il dataset iniziale
    
    # Lancia il loop interattivo PASSANDO X e y per poterli espandere!
    interactive_diagnostic_shell(final_model, extractor, X, y)

if __name__ == "__main__":
    main()