local gameEnv = torch.class('games.GameEnvironment')
zmq = require "lzmq"
require 'image'

function gameEnv:__init(port)
    self.port = port

    self.ZMQ_PORT = ''..port  -- '1728'
    self.ctx = zmq.context()
    self.skt = self.ctx:socket{zmq.REQ,
        linger = 0, rcvtimeo = 1000;
        connect = "tcp://127.0.0.1:" .. self.ZMQ_PORT;
    }
    self:newGame()
    return self
end

function gameEnv:getState()
    self.skt:send("state")
    msg = self.skt:recv(); while msg == nil do msg = self.skt:recv() end
    loadstring(msg)()
    state = image.load('../games/current' .. self.port .. '.png')
    return state, reward, terminal
    -- return self._state.observation, self._state.reward, self._state.terminal
end


function gameEnv:step(action)
    action = ''..action
    self.skt:send("step")
    -- wait for an ack
    msg = self.skt:recv(); while msg == nil do msg = self.skt:recv() end
    self.skt:send(action)
    msg = self.skt:recv(); while msg == nil do msg = self.skt:recv() end
    loadstring(msg)()
    state = image.load('../games/current' .. self.port .. '.png')
    return state, reward, terminal
end

function gameEnv:newGame()
    self.skt:send("reset")
    msg = self.skt:recv(); while msg == nil do msg = self.skt:recv() end
    return self:getState() 
end

function gameEnv:nextRandomGame(k)
    -- currently, this wrapper does not support random_starts
    print("doom wrapper does not support random_starts > 0")
    os.exit()
end

function gameEnv:getActions(player)
    self.skt:send("actions")
    msg = self.skt:recv(); while msg == nil do msg = self.skt:recv() end
    loadstring(msg)()
    return torch.Tensor(actions); 
end

