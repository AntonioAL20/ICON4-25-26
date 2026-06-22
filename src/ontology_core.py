from owlready2 import *
import warnings
import os

warnings.filterwarnings("ignore", category=UserWarning)

def build_advanced_ontology():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    onto_path = os.path.join(base_dir, "..", "data", "stampante3d_inferred.owl")
    
    onto = get_ontology("http://www.di.uniba.it/icon/stampante3d.owl")

    with onto:
        # --- CLASSI BASE ---
        class EntitaDiagnostica(Thing): pass
        class Componente(EntitaDiagnostica): pass
        class Sintomo(EntitaDiagnostica): pass
        class StatoOperativo(Thing): pass
        class CategoriaDegrado(Thing): pass

        # --- TASSONOMIA STATO OPERATIVO E PROFILI DI DEGRADO ---
        class AmbienteUmido(StatoOperativo): pass
        class AltaVelocita(StatoOperativo): pass
        class UsuraAvanzata(StatoOperativo): pass

        class DegradoMeccanico(CategoriaDegrado): pass
        class DegradoTermico(CategoriaDegrado): pass

        # --- OBJECT PROPERTIES (LE 5 RELAZIONI DI PROGETTO) ---
        class ha_sottocomponente(ObjectProperty, TransitiveProperty):
            domain = [Componente]; range = [Componente]
        
        class parte_di(ObjectProperty):
            domain = [Componente]; range = [Componente]
            inverse_property = ha_sottocomponente

        class coinvolge_componente(ObjectProperty):
            domain = [Sintomo]; range = [Componente]

        # 1. sensibile_a: Relazione tra entità e stati ambientali critici
        class sensibile_a(ObjectProperty):
            domain = [EntitaDiagnostica]; range = [StatoOperativo]

        # 2. aggravato_da: Relazione tra manifestazioni di guasto e dinamiche operative
        class aggravato_da(ObjectProperty):
            domain = [Sintomo]; range = [StatoOperativo]

        # 3. alimentato_da: Relazione di dipendenza energetico-funzionale meccatronica
        class alimentato_da(ObjectProperty):
            domain = [Componente]; range = [Componente]

        # 4. conseguenza_di: Catena causale tra sintomi (Proprietà Transitiva)
        class conseguenza_di(ObjectProperty, TransitiveProperty):
            domain = [Sintomo]; range = [Sintomo]

        # 5. soggetto_a_degrado: Mappatura della vulnerabilità fisica dell'hardware
        class soggetto_a_degrado(ObjectProperty):
            domain = [Componente]; range = [CategoriaDegrado]

        # --- TASSONOMIA HARDWARE COMPONENTI ---
        class ComponenteElettrico(Componente): pass
        class ComponenteMeccanico(Componente): pass
        class ComponenteTermico(Componente): pass
        class StrutturaMeccanica(ComponenteMeccanico): pass

        class MotoreStepper(ComponenteMeccanico, ComponenteElettrico): pass
        class Termistore(ComponenteTermico, ComponenteElettrico): pass
        class Ugello(ComponenteMeccanico, ComponenteTermico): pass
        class PiattoRiscaldato(ComponenteTermico, ComponenteElettrico): pass
        class SchedaMadre(ComponenteElettrico): pass
        class Finecorsa(ComponenteElettrico, ComponenteMeccanico): pass
        class Ventola(ComponenteElettrico, ComponenteMeccanico): pass
        class Cinghia(ComponenteMeccanico): pass

        # --- ISTANZE DI CONTESTO (ABox - Ambienti e Profili) ---
        config_umido = AmbienteUmido("ContestoAmbienteUmido")
        config_veloce = AltaVelocita("ContestoAltaVelocita")
        config_usura = UsuraAvanzata("ContestoUsuraAvanzata")

        degrado_mecc = DegradoMeccanico("ProfiloDegradoMeccanico")
        degrado_term = DegradoTermico("ProfiloDegradoTermico")

        # --- ISTANZE HARDWARE (ABox - Topologia) ---
        extruder_asm = ComponenteMeccanico("GruppoEstrusore")
        asse_x = StrutturaMeccanica("AsseX")
        asse_y = StrutturaMeccanica("AsseY")
        
        mainboard = SchedaMadre("SchedaMadrePrincipale")
        piatto = PiattoRiscaldato("PiattoDiStampa")
        
        motore_e = MotoreStepper("MotoreEstrusore", parte_di=[extruder_asm])
        motore_x = MotoreStepper("MotoreX", parte_di=[asse_x])
        cinghia_x = Cinghia("CinghiaX", parte_di=[asse_x])
        endstop_x = Finecorsa("FinecorsaX", parte_di=[asse_x])
        
        ugello_1 = Ugello("UgelloPrincipale", parte_di=[extruder_asm])
        termistore_hotend = Termistore("TermistoreHotend", parte_di=[extruder_asm])
        ventola_hotend = Ventola("VentolaHotend", parte_di=[extruder_asm])

        # --- VALORIZZAZIONE ASSERZIONI SULLE NUOVE PROPRIETA' ---
        # 3. Modellazione albero elettrico (Alimentazione Hardware)
        motore_e.alimentato_da = [mainboard]
        motore_x.alimentato_da = [mainboard]
        piatto.alimentato_da = [mainboard]
        termistore_hotend.alimentato_da = [mainboard]
        ventola_hotend.alimentato_da = [mainboard]
        endstop_x.alimentato_da = [mainboard]

        # 5. Mappatura del ciclo di vita fisico (Degrado)
        cinghia_x.soggetto_a_degrado = [degrado_mecc]
        motore_x.soggetto_a_degrado = [degrado_mecc]
        motore_e.soggetto_a_degrado = [degrado_mecc]
        termistore_hotend.soggetto_a_degrado = [degrado_term]
        ugello_1.soggetto_a_degrado = [degrado_term]

        # --- RESTRIZIONI LOGICHE AVANZATE (Description Logic) ---
        class AllarmeTermico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteTermico)]
        
        class AllarmeMeccanico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteMeccanico)]
            
        class AllarmeElettrico(Sintomo):
            equivalent_to = [Sintomo & coinvolge_componente.some(ComponenteElettrico)]

        # Nuove restrizioni deduttive abilitate dalle nuove Object Properties
        class AllarmeAdesioneIgroscopica(Sintomo):
            equivalent_to = [Sintomo & aggravato_da.some(AmbienteUmido)]

        class AllarmeCriticoUsura(Sintomo):
            equivalent_to = [Sintomo & aggravato_da.some(UsuraAvanzata)]

        # --- POPOLAMENTO SINTOMI ED ASSERZIONI CONTESTUALI CAUSALI ---
        s_alta_temp = Sintomo("TempTroppoAlta", coinvolge_componente=[termistore_hotend])
        s_inst_temp = Sintomo("TempInstabile", coinvolge_componente=[termistore_hotend], aggravato_da=[config_usura])
        s_piatto_fr = Sintomo("PiattoFreddo", coinvolge_componente=[piatto])
        
        s_ticchettio = Sintomo("TicchettioEstrusore", coinvolge_componente=[motore_e])
        s_no_filam = Sintomo("FilamentoNonEsce", coinvolge_componente=[ugello_1])
        s_sotto_est = Sintomo("SottoEstrusione", coinvolge_componente=[extruder_asm])
        
        s_layer_sp = Sintomo("LayerSpostati", coinvolge_componente=[motore_x, cinghia_x], aggravato_da=[config_veloce, config_usura])
        s_rum_cingh = Sintomo("RumoreCinghia", coinvolge_componente=[cinghia_x], aggravato_da=[config_usura])
        s_home_fall = Sintomo("HomeFallito", coinvolge_componente=[endstop_x, motore_x])
        
        s_bruciato = Sintomo("OdoreBruciato", coinvolge_componente=[mainboard])
        s_scr_nero = Sintomo("SchermoNero", coinvolge_componente=[mainboard])
        s_vent_rum = Sintomo("VentolaRumorosa", coinvolge_componente=[ventola_hotend])
        
        s_warping = Sintomo("Warping", coinvolge_componente=[piatto], aggravato_da=[config_umido])

        # 4. Modellazione Catena di Conseguenze (Root Cause Analysis - Transitività)
        s_ticchettio.conseguenza_di = [s_no_filam]  # Il ticchettio è conseguenza dell'ugello ostruito
        s_sotto_est.conseguenza_di = [s_ticchettio]  # La sotto-estrusione è conseguenza del ticchettio del motore
        # Grazie alla transitività di conseguenza_di, HermiT dedurrà autonomamente che 
        # SottoEstrusione è conseguenza_di FilamentoNonEsce.

    return onto, onto_path