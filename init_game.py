import vizdoom as vzd


def init_game(episode_length):
    # Create DoomGame instance. It will run the game and communicate with you.
    game = vzd.DoomGame()

    # Now it's time for configuration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path("wads/doom2.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Easy difficulty
    game.set_doom_skill(4)

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_320X240)

    # Sets the screen buffer format. Not used here but now you can change it. Default is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    #game.set_screen_format(vzd.ScreenFormat.GRAY8)
    # Enables depth buffer.
    game.set_depth_buffer_enabled(False)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(False)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(False)

    # Enables information about all objects present in the current episode/level.
    game.set_objects_info_enabled(False)

    # Enables information about all sectors (map layout).
    game.set_sectors_info_enabled(False)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(True)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(True)  # Bullet holes and blood on the walls
    game.set_render_particles(True)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(True)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.ATTACK)
    game.add_available_button(vzd.Button.USE)
    game.add_available_button(vzd.Button.SPEED)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.SELECT_WEAPON1)
    game.add_available_button(vzd.Button.SELECT_WEAPON2)
    game.add_available_button(vzd.Button.SELECT_WEAPON3)
    game.add_available_button(vzd.Button.SELECT_WEAPON4)
    game.add_available_button(vzd.Button.SELECT_WEAPON5)
    game.add_available_button(vzd.Button.SELECT_WEAPON6)
    game.add_available_button(vzd.Button.SELECT_WEAPON7)
    game.add_available_button(vzd.Button.TURN_LEFT_RIGHT_DELTA, 10)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.WEAPON0)
    game.add_available_game_variable(vzd.GameVariable.WEAPON1)
    game.add_available_game_variable(vzd.GameVariable.WEAPON2)
    game.add_available_game_variable(vzd.GameVariable.WEAPON3)
    game.add_available_game_variable(vzd.GameVariable.WEAPON4)
    game.add_available_game_variable(vzd.GameVariable.WEAPON5)
    game.add_available_game_variable(vzd.GameVariable.WEAPON6)
    game.add_available_game_variable(vzd.GameVariable.AMMO2) # bullets
    game.add_available_game_variable(vzd.GameVariable.AMMO3) # shells
    game.add_available_game_variable(vzd.GameVariable.AMMO5) # rockets
    game.add_available_game_variable(vzd.GameVariable.AMMO6) # plasma

    game.add_available_game_variable(vzd.GameVariable.HEALTH)
    game.add_available_game_variable(vzd.GameVariable.ARMOR)
    game.add_available_game_variable(vzd.GameVariable.DAMAGECOUNT)

    game.add_available_game_variable(vzd.GameVariable.VELOCITY_X)
    game.add_available_game_variable(vzd.GameVariable.VELOCITY_Y)
    game.add_available_game_variable(vzd.GameVariable.ATTACK_READY)

    # How many ticks the episode is at maximum
    game.set_episode_timeout(episode_length)

    # Makes episodes start after 14 tics (~after raising the weapon)
    game.set_episode_start_time(14)

    # Makes the window appear (turned on by default)
    game.set_window_visible(True)
    
    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console.
    #game.set_console_enabled(True)

    game.set_death_penalty(1000)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

    return game