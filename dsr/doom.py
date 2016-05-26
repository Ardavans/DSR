#!/usr/bin/python
#####################################################################
# This script presents how to use the most basic features of the environment.
# It configures the engine, and makes the agent perform random actions.
# It also gets current state and reward earned with the action.
# <episodes> number of episodes are played. 
# Random combination of buttons is chosen for every action.
# Game variables from state and last reward are printed.
# To see the scenario description go to "../../scenarios/README.md"
# 
#####################################################################
from __future__ import print_function
from vizdoom import DoomGame
from vizdoom import Mode
from vizdoom import Button
from vizdoom import GameVariable
from vizdoom import ScreenFormat
from vizdoom import ScreenResolution
# Or just use from vizdoom import *

from random import choice
from time import sleep
from time import time
import zmq
import json
import sys,pdb,scipy.misc
import numpy as np

# Create DoomGame instance. It will run the game and communicate with you.
game = DoomGame()

# Now it's time for configuration!
# load_config could be used to load configuration instead of doing it here with code.
# If load_config is used in-code configuration will work. Note that the most recent changes will add to previous ones.
#game.load_config("../../examples/config/basic.cfg")

# Sets path to vizdoom engine executive which will be spawned as a separate process. Default is "./vizdoom".
game.set_vizdoom_path("bin/vizdoom")

# Sets path to doom2 iwad resource file which contains the actual doom game. Default is "./doom2.wad".
# game.set_doom_game_path("scenarios/freedoom2.wad")
game.set_doom_game_path("scenarios/DOOM2.WAD")

#game.set_doom_game_path("../../scenarios/doom2.wad")  # Not provided with environment due to licences.

# Sets path to additional resources iwad file which is basically your scenario iwad.
# If not specified default doom2 maps will be used and it's pretty much useles... unless you want to play doom.
# game.set_doom_scenario_path("scenarios/basic.wad")
game.set_doom_scenario_path("scenarios/2roomsbest.wad")

# Sets map to start (scenario .wad files can contain many maps).
game.set_doom_map("map01")

# Sets resolution. Default is 320X240
# game.set_screen_resolution(ScreenResolution.RES_640X480)

# Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.

# Sets other rendering options
game.set_render_hud(False)
game.set_render_crosshair(False)
game.set_render_weapon(True)
game.set_render_decals(False)
game.set_render_particles(False)

# Adds buttons that will be allowed. 
# game.add_available_button(Button.MOVE_LEFT)
# game.add_available_button(Button.MOVE_RIGHT)
game.add_available_button(Button.MOVE_FORWARD)
game.add_available_button(Button.TURN_LEFT)
game.add_available_button(Button.TURN_RIGHT)
# game.add_available_button(Button.TURN180)
# game.add_available_button(Button.ATTACK)
game.add_available_button(Button.USE)

# game.add_available_button(Button.ATTACK)

# Adds game variables that will be included in state.
game.add_available_game_variable(GameVariable.AMMO2)

# Causes episodes to finish after 200 tics (actions)
game.set_episode_timeout(5000)

# Makes episodes start after 10 tics (~after raising the weapon)
game.set_episode_start_time(10)

# Makes the window appear (turned on by default)
game.set_window_visible(False)

# Turns on the sound. (turned off by default)
game.set_sound_enabled(False)

# Sets the livin reward (for each move) to -1
game.set_living_reward(-0.01)

# Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
game.set_mode(Mode.PLAYER)

# Initialize the game. Further configuration won't take any effect from now on.
game.init()

# available_buttons = { 
#         TURN_LEFT,
#         TURN_RIGHT,
#         MOVE_FORWARD 
#         MOVE_LEFT
#         MOVE_RIGHT
#     }

# Define some actions. Each list entry corresponds to declared buttons:
# MOVE_LEFT, MOVE_RIGHT, ATTACK
# 5 more combinations are naturally possible but only 3 are included for transparency when watching.	
# actions = [[True,False,False,False,False],[False,True,False,False,False],

# actions=[[True,False,False, False, False, False], [False,True , False,False, False, False],[False,False, True, False, False,False], [False,False, False,  True,False,False],
# [False,False,False,False, True, False], [False,False,False,False, False, True]]

# actions=[[True,False,False, False, False], [False,True , False,False, False],[False,False, True, False, False], [False,False, False,  True,False],
# [False,False,False,False, True]]


actions=[[True,False,False, False], [False,True , False, False],[False,False, True, False], [False,False, False,  True]]

# Sets time that will pause the engine after each action.
# Without this everything would go too fast for you to keep track of what's happening.
# 0.05 is quite arbitrary, nice to watch with my hardware setup. 
sleep_time = 0#0.028


def get_state(state, r, done):
    send_msg = 'state = '+json.dumps(state).replace('[','{').replace(']','}')
    send_msg += ';reward = '+json.dumps(r).replace('[','{').replace(']','}')
    send_msg += ';terminal = '+json.dumps(done).replace('[','{').replace(']','}')
    return send_msg


port = "1728"
if len(sys.argv) > 1:
    port =  int(sys.argv[1])

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:%s" % port)

game.new_episode()
r = -0.01 #base
s=None
while True:
    msg=socket.recv()

    if msg == "state":
        s = game.get_state(); img = s.image_buffer; 
        img = scipy.misc.imresize(img,(84,84,3)); scipy.misc.imsave('../games/current' +str(port)+ '.png', img)

        socket.send(get_state('../games/current' +str(port)+ '.png', r, game.is_episode_finished()))

    elif msg == "step": 
        socket.send("action")
        a_indx = socket.recv()
        a_indx = int(a_indx)-1
        if a_indx == 1 or a_indx == 2:
            for ii in range(5):
                r = game.make_action(actions[a_indx])
        else:
            r = game.make_action(actions[a_indx])
        # for our toy world, ammo is reward

        r = s.game_variables[0]-50 #50 is baseline
        if r == 0:
            r = -0.01
        terminal = game.is_episode_finished()
        if r >= 0:
            terminal = True 
        if terminal == False:
            s = game.get_state(); img = s.image_buffer; 

        img = s.image_buffer; img = scipy.misc.imresize(img,(84,84,3))
        scipy.misc.imsave('../games/current' +str(port)+ '.png', img)
        socket.send(get_state('../games/current'+str(port)+'.png', r, terminal))
        if terminal:
            game.new_episode()    

    elif msg == "actions":
        socket.send('actions = '+json.dumps(range(1,len(actions)+1)).replace('[','{').replace(']','}'))

    elif msg == "reset":
        game.new_episode()
        socket.send("ack")

    if game.is_episode_finished():
        game.new_episode()