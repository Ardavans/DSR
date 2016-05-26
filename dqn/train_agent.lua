--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

if not dqn then
    require "initenv"
end

require 'optim'
require 'xlua'

local cmd = torch.CmdLine()
cmd:text()
cmd:text('Train Agent in Environment:')
cmd:text()
cmd:text('Options:')

cmd:option('-framework', '', 'name of training framework')
cmd:option('-game_name', 'MovingGoals', 'name of the mazebase environment')
cmd:option('-env', '', 'name of environment to use')
cmd:option('-game_path', '', 'path to environment file (ROM)')
cmd:option('-env_params', '', 'string of environment parameters')
cmd:option('-pool_frms', '',
           'string of frame pooling parameters (e.g.: size=2,type="max")')
cmd:option('-actrep', 1, 'how many times to repeat action')
cmd:option('-random_starts', 0, 'play action 0 between 1 and random_starts ' ..
           'number of times at the start of each training episode')

cmd:option('-name', '', 'filename used for saving network and training history')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-agent', '', 'name of agent file to use')
cmd:option('-agent_params', '', 'string of agent parameters')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')
cmd:option('-saveNetworkParams', false,
           'saves the agent network in a separate file')
cmd:option('-prog_freq', 5*10^3, 'frequency of progress output')
cmd:option('-save_freq', 5*10^4, 'the model is saved every save_freq steps')
cmd:option('-eval_freq', 10^4, 'frequency of greedy evaluation')
cmd:option('-save_versions', 0, '')

cmd:option('-steps', 10^5, 'number of training steps to perform')
cmd:option('-eval_steps', 10^5, 'number of evaluation steps')

cmd:option('-verbose', 2,
           'the higher the level, the more information is printed to screen')
cmd:option('-threads', 1, 'number of BLAS threads')
cmd:option('-gpu', -1, 'gpu flag')
cmd:option('max_steps_episode', 150, 'maximum number of steps per episode')
cmd:text()

local opt = cmd:parse(arg)
disp = require 'display'
disp.configure({port=opt.seed})
--- General setup.
local game_env, game_actions, agent, opt = setup(opt)

-- override print to always flush the output
local old_print = print
local print = function(...)
    old_print(...)
    io.flush()
end

local learn_start = agent.learn_start
local start_time = sys.clock()
local reward_counts = {}
local episode_counts = {}
local time_history = {}
local v_history = {}
local qmax_history = {}
local td_history = {}
local reward_history = {}
local step = 0
time_history[1] = 0

local total_reward
local nrewards
local nepisodes
local episode_reward

local screen, reward, terminal = game_env:getState()

local episode_step_counter = 0

print("Iteration ..", step)
foldername = '../results/' .. opt.name .. '/'
os.execute("mkdir -p " .. foldername)

local win = nil
while step < opt.steps do
    step = step + 1
    local action_index = agent:perceive(reward, screen, terminal)

    -- game over? get next game!
    if not terminal and episode_step_counter < opt.max_steps_episode then
        -- for k,v in pairs(game_actions) do print(k,v) end
        screen, reward, terminal = game_env:step(game_actions[action_index], true)
        episode_step_counter = episode_step_counter + 1
    else
        if opt.random_starts > 0 then
            screen, reward, terminal = game_env:nextRandomGame()
        else
            screen, reward, terminal = game_env:newGame()
        end
        episode_step_counter = 0
    end

    -- display screen
    --win = image.display({image=screen, win=win})
    disp.image(screen, {win=3, title=foldername})
    if step % opt.prog_freq == 0 then
        assert(step==agent.numSteps, 'trainer step: ' .. step ..
                ' & agent.numSteps: ' .. agent.numSteps)
        print("Steps: ", step)
        agent:report()
        collectgarbage()
    end


    if step%1000 == 0 then collectgarbage() end

    xlua.progress(step, opt.steps)

    if step % opt.eval_freq == 0 and step > learn_start then
        print('Eval on-going ... ')
        screen, reward, terminal = game_env:newGame()

        total_reward = 0
        nrewards = 0
        nepisodes = 0
        episode_reward = 0

        reward_plot_data = reward_plot_data or {}
        v_plot_data = v_plot_data or {}
        td_plot_data = td_plot_data or {}

        
        test_avg_R = test_avg_R or optim.Logger(paths.concat(foldername .. 'test_avgR.log'))
        tderr_avg = tderr_avg or optim.Logger(paths.concat(foldername .. 'tderr_avg.log'))
        vavg = vavg or optim.Logger(paths.concat(foldername .. 'vavg.log'))

        local eval_time = sys.clock()
        for estep=1,opt.eval_steps do
            local action_index = agent:perceive(reward, screen, terminal, true, 0.05)

            -- Play game in test mode (episodes don't end when losing a life)
            screen, reward, terminal = game_env:step(game_actions[action_index])

            -- display screen
            --win = image.display({image=screen, win=win})
            disp.image(screen, {win=3, title='evaluation'})

            if estep%1000 == 0 then collectgarbage() end

            -- record every reward

            episode_reward = episode_reward + reward
            if reward ~= 0 then
               nrewards = nrewards + 1
            end

            if terminal or estep == opt.eval_steps then
                total_reward = total_reward + episode_reward
                episode_reward = 0
                nepisodes = nepisodes + 1
                -- screen, reward, terminal = game_env:nextRandomGame()
                screen, reward, terminal = game_env:newGame()
            end
        end

        eval_time = sys.clock() - eval_time
        start_time = start_time + eval_time
        agent:compute_validation_statistics()
        local ind = #reward_history+1
        total_reward = total_reward/math.max(1, nepisodes)

        if #reward_history == 0 or total_reward > torch.Tensor(reward_history):max() then
            agent.best_network = agent.network:clone()
        end

        if agent.v_avg then
            v_history[ind] = agent.v_avg
            td_history[ind] = agent.tderr_avg
            qmax_history[ind] = agent.q_max
        end
        print("V", v_history[ind], "TD error", td_history[ind], "Qmax", qmax_history[ind])

        reward_history[ind] = total_reward
        reward_counts[ind] = nrewards
        episode_counts[ind] = nepisodes

        time_history[ind+1] = sys.clock() - start_time

        local time_dif = time_history[ind+1] - time_history[ind]

        local training_rate = opt.actrep*opt.eval_freq/time_dif

        -- test_avg_R:add{['% Average Extrinsic Reward'] = total_reward}
        -- test_avg_R:style{['% Average Extrinsic Reward'] = '-'}; test_avg_R:plot()

        -- vavg:add{['% V avg '] = agent.v_avg}  
        -- vavg:style{['% V avg '] = '-'}; vavg:plot()

        -- tderr_avg:add{['% TD avg '] = agent.tderr_avg} --total_reward
        -- tderr_avg:style{['% TD avg '] = '-'}; tderr_avg:plot()


        test_avg_R:add{['% Average Extrinsic Reward'] = total_reward} 
        
        --test_avg_R:style{['% Average Extrinsic Reward'] = '-'}; --test_avg_R:plot()
        
        vavg:add{['% V avg '] = agent.v_avg}  
        --vavg:style{['% V avg '] = '-'}; vavg:plot()

        tderr_avg:add{['% TD avg '] = agent.m_tderr_avg} --total_reward
        --m_tderr_avg:style{['% TD avg '] = '-'}; m_tderr_avg:plot()
        table.insert(reward_plot_data, {step, total_reward})
        table.insert(v_plot_data, {step, agent.v_avg})
        table.insert(td_plot_data, {step,  agent.m_tderr_avg})
        disp.plot(reward_plot_data, {win=4, title='Average Reward', xlabel = 'Steps'})
        disp.plot(v_plot_data, {win=5, title='Average Q', xlabel = 'Steps'})
        disp.plot(td_plot_data, {win=6, title='Average TD Error', xlabel = 'Steps'})


        print(string.format(
            '\nSteps: %d (frames: %d), reward: %.2f, epsilon: %.2f, lr: %G, ' ..
            'training time: %ds, training rate: %dfps, testing time: %ds, ' ..
            'testing rate: %dfps,  num. ep.: %d,  num. rewards: %d',
            step, step*opt.actrep, total_reward, agent.ep, agent.lr, time_dif,
            training_rate, eval_time, opt.actrep*opt.eval_steps/eval_time,
            nepisodes, nrewards))
    end

    if step % opt.save_freq == 0 or step == opt.steps then
        local s, a, r, s2, term = agent.valid_s, agent.valid_a, agent.valid_r,
            agent.valid_s2, agent.valid_term
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = nil, nil, nil, nil, nil, nil, nil
        local w, dw, g, g2, delta, delta2, deltas, tmp = agent.w, agent.dw,
            agent.g, agent.g2, agent.delta, agent.delta2, agent.deltas, agent.tmp
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = nil, nil, nil, nil, nil, nil, nil, nil

        local filename = foldername .. opt.name
        if opt.save_versions > 0 then
            filename = filename .. "_" .. math.floor(step / opt.save_versions)
        end
        filename = filename
        torch.save(filename .. ".t7", {agent = agent,
                                model = agent.network,
                                best_model = agent.best_network,
                                reward_history = reward_history,
                                reward_counts = reward_counts,
                                episode_counts = episode_counts,
                                time_history = time_history,
                                v_history = v_history,
                                td_history = td_history,
                                qmax_history = qmax_history,
                                arguments=opt})
        if opt.saveNetworkParams then
            local nets = {network=w:clone():float()}
            torch.save(filename..'.params.t7', nets, 'ascii')
        end
        agent.valid_s, agent.valid_a, agent.valid_r, agent.valid_s2,
            agent.valid_term = s, a, r, s2, term
        agent.w, agent.dw, agent.g, agent.g2, agent.delta, agent.delta2,
            agent.deltas, agent.tmp = w, dw, g, g2, delta, delta2, deltas, tmp
        print('Saved:', filename .. '.t7')
        io.flush()
        collectgarbage()
    end
end
