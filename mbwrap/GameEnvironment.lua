local gameEnv = torch.class('mbwrap.GameEnvironment')

function gameEnv:__init(game_name)
    g_opts = {}
    g_opts.games_config_path = '../mbwrap/games/config/game_config.lua'

    -- Specify game environment
    g_opts.game = game_name

    g_init_vocab()
    g_init_game()
    self._state = {}

    self:newGame()

    return self
end

function gameEnv:_updateState(frame, reward, terminal)
    self._state.observation = frame
    self._state.reward = reward
    self._state.terminal = terminal
    return self
end

function gameEnv:getState()
    --returns screen, reward, terminal
    self._state.observation = self._state.observation or self.g.map:to_image():clone()
    self._state.observation:copy(self.g.map:to_image())
    return self._state.observation, self._state.reward, self._state.terminal
end

function gameEnv:_randomStep()
    --function plays one random action in the game and returns game state
    return self:_step(self._actions[torch.random(#self._actions)])
end

function gameEnv:step(action, training)
    -- training is boolean, returns self:getState()
    -- print('this is action', action)
    assert(action)
    action_name_to_index = {['up'] = 1, ['down'] = 2,
                            ['left'] = 3, ['right'] = 4,
                            ['stop'] = 5}
    action = action_name_to_index[action]
    self.g:act(action)
    self.g:update()
    frame = self.g.map:to_image()
    reward = self.g:get_reward()
    terminal = self.g:is_terminal()
    self:_updateState(frame, reward, terminal)
    return self:getState()
end

function gameEnv:newGame()
    -- creates new game
    self.g = new_game()
    -- take one null action in the new game
    frame = self.g.map:to_image()
    reward = 0
    terminal = false
    return self:_updateState(frame, reward, terminal):getState()
end

function gameEnv:nextRandomGame(k)
    -- currently, this wrapper does not support random_starts
    print("mbwrapper does not support random_starts > 0")
    os.exit()
end

function gameEnv:nObsFeature()
    -- returns feature dimension
    frame_size = self.g.map:to_image():size()
    return frame_size[1]*frame_size[2]*frame_size[3]

end

function gameEnv:getActions()
    t = self.g.agent.action_names
    table.remove(t,6)
    print(t)
    return t
end

