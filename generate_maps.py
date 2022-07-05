from oblige import *


oblige_config_easy = {
    "length": "game",
    "theme": "bit_mixed",
    "size": "micro",
    "outdoors": "mixed",
    "caves": "mixed",
    "liquids": "none",
    "hallways": "mixed",
    "teleporters": "none",
    "steepness": "mixed",

    "mons": "nuts",
    "strength": "easier",
    "ramp_up": "medium",
    "bosses": "none",
    "traps": "none",
    "cages": "none",

    "health": "heaps",
    "ammo": "heaps",
    "weapons": "sooner",
    "items": "more",
    "secrets": "mixed",

    "misc": 1,
    # misc module options:
    "pistol_starts": "yes",
    "alt_starts": "no",
    "big_rooms": "mixed",
    "parks": "mixed",
    "windows": "mixed",
    "symmetry": "mixed",
    "darkness": "mixed",
    "mon_variety": "mixed",
    "barrels": "some",
    "doors": "none",
    "keys": "none",
    "switches": "none",

    "doom_mon_control": 1,
    "spectre": "none",

    "stealth_mons": 0,
    "stealth_mon_control": 1,
    # stealth_mon_control module options
    "stealth_demon": "none",
    "stealth_baron": "none",
    "stealth_zombie": "none",
    "stealth_caco": "none",
    "stealth_imp": "none",
    "stealth_mancubus": "none",
    "stealth_arach": "none",
    "stealth_revenant": "none",
    "stealth_shooter": "none",
    "stealth_vile": "none",
    "stealth_knight": "none",
    "stealth_gunner": "none",
}

oblige_config_smoketest = {
    "length": "game",
    "theme": "original",
    "size": "micro",
    "outdoors": "none",
    "caves": "none",
    "liquids": "none",
    "hallways": "none",
    "teleporters": "none",
    "steepness": "none",

    "mons": "none",
    "strength": "easier",
    "ramp_up": "medium",
    "bosses": "none",
    "traps": "none",
    "cages": "none",

    "health": "none",
    "ammo": "none",
    "weapons": "none",
    "items": "none",
    "secrets": "none",

    "misc": 1,
    "pistol_starts": "yes",
    "alt_starts": "no",
    "big_rooms": "none",
    "parks": "none",
    "windows": "none",
    "symmetry": "none",
    "darkness": "none",
    "mon_variety": "none",
    "barrels": "none",
    "doors": "none",
    "keys": "none",
    "switches": "none"
}


def generate_maps(filename="wads/temp/oblige.wad", seed=1507715517, is_smoketest=False):
    # return
    generator = oblige.DoomLevelGenerator()
    generator.set_seed(seed)

    if is_smoketest:
        generator.set_config(oblige_config_smoketest)
    else:
        generator.set_config(oblige_config_easy)
    
    print("Generating {} ...".format(filename))
    num_maps = generator.generate(filename, verbose=False)
    print("Generated {} maps.".format(num_maps))
