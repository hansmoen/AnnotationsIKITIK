# -*- coding: iso-8859-1 -*-

#### description

# classes and mappings for pain



#### constants

# the order of classes in output table

CLASSES_ORIGINAL = [
    "Kipu",
    "Implisiittinen_kipu",
    "Potentiaalinen_kipu",
    "Kipumaare",
    "Voimakkuus",
    "Aika",
    "Kesto",
    "Ajanhetki",
    "Jaksollisuus",
    "Laatu",
    "Sijainti",
    "Tilanne",
    "Toistuva_tilanne",
    "Toimenpide",
    "Hoidon_onnistuminen",
    "Kivunhoito",
    "Laakitys",
    "Emotionaalinen_hoito",
    "Kognitiivinen_hoito",
    "Fysikaalinen_hoito",
    "Ohjeistus",
    "Suunnitelma",
    "Kipuun_liittyva_asia",
    "Lisaaja",
    "Seuraus",
    "Syy",
    ]

CLASSES_PRETTY_EN = [
    "Pain",
    "Implicit pain",
    "Potential pain",
    "Attribute",
    "Intensity",
    "Time",
    "Time sublevel 1",
    "Time sublevel 2",
    "Time sublevel 3",
    "Character",
    "Location",
    "Situation",
    "Cause of recurrence",
    "Operation",
    "Success of treatment",
    "Pain management",
    "Pain management sublevel 1",
    "Pain management sublevel 2",
    "Pain management sublevel 3",
    "Pain management sublevel 4",
    "Patient education",
    "Plan",
    "Pain-related",
    "Pain-related sublevel 1",
    "Pain-related sublevel 2",
    "Pain-related sublevel 3",
    ]

#### helper functions

# the mapping of classes to simplify the analysis

def simplify_class(cls):
    if cls in ["Kesto",
               "Ajanhetki",
               "Jaksollisuus"]:
        return "Aika"
    elif cls in ["Laakitys",
                 "Emotionaalinen_hoito",
                 "Kognitiivinen_hoito",
                 "Fysikaalinen_hoito"]:
        return "Kivunhoito"
    elif cls in ["Lisaaja",
                "Vahentaja",
                "Seuraus",
                "Syy"]:
        return "Kipuun_liittyva_asia"
    else:
        return cls

# generate a mapping for renaming purposes

def get_mapping(f, t):
    return dict(zip(f,t))
