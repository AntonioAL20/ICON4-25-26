from owlready2 import *
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def build_advanced_ontology():
    """
    Costruisce un'ontologia OWL 2.0 complessa per stampanti 3D.
    Usa ObjectProperties inverse e restrizioni logiche per abilitare
    l'inferenza automatica (Description Logic).
    """
    # Percorso per il salvataggio opzionale
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onto_path = os.path.join(base_dir, "..", "data", "stampante3d_inferred.owl")
    
    onto = get_ontology("http://www.di.uniba.it/icon/stampante3d.owl")

    with onto:
        # --- CLASSI BASE ---
        class EntitaDiagnostica(Thing): pass
        class Componente(EntitaDiagnostica): pass
        class Sintomo(EntitaDiagnostica): pass

        # --- OBJECT PROPERTIES (Transitive e Inverse) ---
        class ha_sottocomponente(ObjectProperty, TransitiveProperty):
            domain = [Componente]
            range = [Componente]
        
        class parte_di(ObjectProperty):
            domain = [Componente]
            range = [Componente]
            inverse_property = ha_sottocomponente

        class coinvolge_componente(ObjectProperty):
            domain = [Sintomo]
            range = [Componente]

        # --- TASSONOMIA COMPONENTI (Ereditarietà Multipla) ---
        class ComponenteElettrico(Componente): pass
        class ComponenteMeccanico(Componente): pass
        class ComponenteTermico(Componente): pass

        class MotoreStepper(ComponenteMeccanico, ComponenteElettrico): pass
        class Termistore(ComponenteTermico, ComponenteElettrico): pass
        class Ugello(ComponenteMeccanico, ComponenteTermico): pass
        class PiattoRiscaldato(ComponenteTermico, ComponenteElettrico): pass
        class SchedaMadre(ComponenteElettrico): pass

        # Individui (Topologia)
        extruder_asm = ComponenteMeccanico("GruppoEstrusore")
        motore_e = MotoreStepper("MotoreEstrusore", parte_di=[extruder_asm])
        ugello_1 = Ugello("UgelloPrincipale", parte_di=[extruder_asm])
        termistore_hotend = Termistore("TermistoreHotend", parte_di=[extruder_asm])
        piatto = PiattoRiscaldato("PiattoDiStampa")
        mainboard = SchedaMadre("SchedaMadrePrincipale")

        # --- RESTRIZIONI LOGICHE (Il vero potere della DL) ---
        # Definiamo dinamicamente le classi di allarme. Il Reasoner classificherà 
        # i sintomi automaticamente se coinvolgono i componenti corretti.
        class AllarmeTermico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteTermico)]
        
        class AllarmeMeccanico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteMeccanico)]
            
        class AllarmeElettrico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteElettrico)]

        # --- INDIVIDUI SINTOMI (RAW) ---
        # Vengono solo legati al componente, senza dire se sono termici o meccanici.
        Sintomo("TempTroppoAlta", coinvolge_componente=[termistore_hotend])
        Sintomo("TicchettioEstrusore", coinvolge_componente=[motore_e])
        Sintomo("FilamentoNonEsce", coinvolge_componente=[ugello_1])
        Sintomo("PiattoFreddo", coinvolge_componente=[piatto])
        Sintomo("VibrazioneAnomala", coinvolge_componente=[extruder_asm])
        Sintomo("OdoreBruciato", coinvolge_componente=[mainboard])

    return onto, onto_path