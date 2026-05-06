from owlready2 import sync_reasoner

class DiagnosticReasoner:
    """
    Gestisce l'inferenza usando il reasoner HermiT.
    Sostituisce il vecchio backward chaining manuale con una vera
    deduzione logica basata sulla Description Logic (DL).
    """
    def __init__(self, ontology):
        self.onto = ontology
        self.reasoner_run = False

    def run_inference(self):
        print("[Reasoner] Avvio di HermiT DL Reasoner...")
        with self.onto:
            sync_reasoner(infer_property_values=True)
        self.reasoner_run = True
        print("[Reasoner] Inferenza completata. Nuove classi materializzate.")

    def get_inferred_classes_for_symptom(self, symptom_name):
        """Restituisce le superclassi logiche inferite per un dato sintomo grezzo."""
        if not self.reasoner_run:
            raise RuntimeError("Eseguire run_inference() prima di interrogare il reasoner.")

        istanza_sintomo = self.onto.search_one(iri=f"*{symptom_name}")
        if not istanza_sintomo:
            return []

        # Estrae tutte le classi inferite escludendo le classi base generiche
        inferred = [cls.name for cls in istanza_sintomo.INDIRECT_is_a 
                    if hasattr(cls, 'name') and cls.name not in ["Thing", "Sintomo", "EntitaDiagnostica"]]
        
        return inferred