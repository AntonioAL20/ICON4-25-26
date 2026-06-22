from owlready2 import sync_reasoner

class DiagnosticReasoner:
    def __init__(self, ontology):
        self.onto = ontology
        self.reasoner_run = False
        self.cache_inferenze = {} 

    def run_inference(self):
        print("[Reasoner] Avvio di HermiT DL Reasoner...")
        with self.onto:
            sync_reasoner(infer_property_values=True)
        self.reasoner_run = True
        print("[Reasoner] Inferenza completata. Nuove classi materializzate.")

    def get_inferred_classes_for_symptom(self, symptom_name):
        if not self.reasoner_run:
            raise RuntimeError("Eseguire run_inference() prima di interrogare il reasoner.")

        if symptom_name in self.cache_inferenze:
            return self.cache_inferenze[symptom_name]

        istanza_sintomo = self.onto.search_one(iri=f"*{symptom_name}")
        if not istanza_sintomo:
            return []

        inferred = [cls.name for cls in istanza_sintomo.INDIRECT_is_a 
                    if hasattr(cls, 'name') and cls.name not in ["Thing", "Sintomo", "EntitaDiagnostica", "StatoOperativo", "CategoriaDegrado"]]
        
        self.cache_inferenze[symptom_name] = inferred
        return inferred