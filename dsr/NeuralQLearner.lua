if not dsr then
    require 'initenv'
end
require 'pprint'
require 'hdf5'
local nql = torch.class('dsr.NeuralQLearner')
local debug = require("debugger")

function nql:__init(args)

    self.name = args.name
    self.state_dim  = args.state_dim -- State dimensionality.
    self.actions    = args.actions
    
    if torch.type(self.actions)~=torch.type({1}) then
        self.actions = torch.totable(self.actions)
    end

    self.n_actions  = #self.actions
    self.verbose    = args.verbose
    self.best       = args.best
    self.srdims = args.srdims
    self.game_name = args.game_name
    --- epsilon annealing
    self.ep_start   = args.ep or 1
    self.ep         = self.ep_start -- Exploration probability.
    self.ep_end     = args.ep_end or self.ep
    self.ep_endt    = args.ep_endt or 1000000

    ---- learning rate annealing
    self.lr_start       = args.lr or 0 --Learning rate.
    self.lr             = self.lr_start
    self.R_lr           = self.lr_start
    self.lr_end         = args.lr_end or self.lr
    self.lr_endt        = args.lr_endt or 1000000
    self.wc             = args.wc or 0  -- L2 weight cost.
    self.minibatch_size = args.minibatch_size or 1
    self.valid_size     = args.valid_size or 500

    --- Q-learning parameters
    self.discount       = args.discount or 0.99 --Discount factor.
    self.update_freq    = args.update_freq or 1
    -- Number of points to replay per learning step.
    self.n_replay       = args.n_replay or 1
    -- Number of steps after which learning starts.
    self.learn_start    = args.learn_start or 0
     -- Size of the transition table.
    self.replay_memory  = args.replay_memory or 1000000
    self.hist_len       = args.hist_len or 1
    self.rescale_r      = args.rescale_r
    self.max_reward     = args.max_reward
    self.min_reward     = args.min_reward
    self.clip_delta     = args.clip_delta
    self.target_q       = args.target_q
    self.bestq          = 0

    self.gpu            = args.gpu

    self.ncols          = args.ncols or 1  -- number of color channels in input
    self.input_dims     = args.input_dims or {self.hist_len*self.ncols, 84, 84}
    self.preproc        = args.preproc  -- name of preprocessing network
    self.histType       = args.histType or "linear"  -- history type to use
    self.histSpacing    = args.histSpacing or 1
    self.nonTermProb    = args.nonTermProb or 1
    self.bufferSize     = args.bufferSize or 512
    self.transition_params = args.transition_params or {}

    self.network    = args.network or self:createNetwork()

    self.debugging_counter = 0

    self.num_samples = args.num_samples
    self.sample_collect = args.sample_collect
    -- check whether there is a network file
    local network_function
    if not (type(self.network) == 'string') then
        error("The type of the network provided in NeuralQLearner" ..
              " is not a string!")
    end

    local msg, err = pcall(require, self.network)
    if not msg then
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, self.network)
        if not err_msg then
            error("Could not find network file ")
        end
        if self.best and exp.best_model then
            self.network = exp.best_model
        else
            self.network = exp.model
        end
    else
        print('Creating Agent Network from ' .. self.network)
        self.network = err
        self.network = self:network()
    end

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
    else
        self.network:float()
    end

    -- Load preprocessing network.
    if not (type(self.preproc == 'string')) then
        error('The preprocessing is not a string')
    end
    msg, err = pcall(require, self.preproc)
    if not msg then
        error("Error loading preprocessing net")
    end
    self.preproc = err
    self.preproc = self:preproc()
    self.preproc:float()

    if self.gpu and self.gpu >= 0 then
        self.network:cuda()
        self.tensor_type = torch.CudaTensor
    else
        self.network:float()
        self.tensor_type = torch.FloatTensor
    end



    self.criterion = nn.MSECriterion()
    self.criterion.sizeAverage = false

    if self.gpu and self.gpu >= 0 then
        self.criterion:cuda()
    end

    -- Create transition table.
    ---- assuming the transition table always gets floating point input
    ---- (Foat or Cuda tensors) and always returns one of the two, as required
    ---- internally it always uses ByteTensors for states, scaling and
    ---- converting accordingly
    local transition_args = {
        stateDim = self.state_dim, numActions = self.n_actions,
        histLen = self.hist_len, gpu = self.gpu,
        maxSize = self.replay_memory, histType = self.histType,
        histSpacing = self.histSpacing, nonTermProb = self.nonTermProb,
        bufferSize = self.bufferSize
    }

    -- self.transitions = dsr.TransitionTable_priority(transition_args)
    self.transitions = dsr.TransitionTable(transition_args)


    self.numSteps = 0 -- Number of perceived states.
    self.R_numSteps = 0 -- counter for reward network
    self.lastState = nil
    self.lastAction = nil
    self.v_avg = 0 -- V running average.
    self.tderr_avg = 0 -- TD error running average.

    self.q_max = 1
    self.r_max = 1

    self.w, self.dw = self.network:getParameters()
    self.dw:zero()

    ---- for sr ------
    self.deltas = self.dw:clone():fill(0)
    self.tmp= self.dw:clone():fill(0)
    self.g  = self.dw:clone():fill(0)
    self.g2 = self.dw:clone():fill(0)

    ---- for r -------
    self.R_deltas = self.dw:clone():fill(0)
    self.R_tmp= self.dw:clone():fill(0)
    self.R_g  = self.dw:clone():fill(0)
    self.R_g2 = self.dw:clone():fill(0)


    if self.target_q then
        self.target_network = self.network:clone()
        self.w_target, self.dw_target = self.target_network:getParameters()
    end
    self.s2_buf = {}; self.reward_buf = {}
end


function nql:reset(state)
    if not state then
        return
    end
    self.best_network = state.best_network
    self.network = state.model
    self.w, self.dw = self.network:getParameters()
    self.dw:zero()
    self.numSteps = 0
    print("RESET STATE SUCCESFULLY")
end


function nql:preprocess(rawstate, gray)
    --print('rawstate', rawstate:sum())
    if self.preproc then
        return self.preproc:forward(rawstate:float(), gray)
                    :clone():reshape(self.state_dim)
    end
    
    return rawstate
end


function extract_node(model,id_name)
  for i,n in ipairs(model.forwardnodes) do
    if n.data.module then
      if  n.data.module.forwardnodes then
        ret =extract_node(n.data.module, id_name)
        if ret then
          return ret
        end
      end
    end
    if n.data.annotations.name== id_name then
      return n
    end
  end
end


function nql:segment(sr_M,k,threshold)
    -- n is number of sr's
    -- m is size of sr embedding

    size_M = #sr_M
    n,m = size_M[1], size_M[2]

    --  get singular value decomposition of m2_max
    u_svd, s_svd, v_svd = torch.svd(sr_M)
    second_singular_vector = u_svd[2]:clone()

    -- assign indicator vectors
    room_indicator = torch.Tensor(n):zero()
    room_indicator_below = torch.Tensor(n):zero()
    for i=1,n do
        if second_singular_vector[i] > threshold then
            room_indicator[i] = 1
        else
            room_indicator_below[i] = 1
        end 
    end
    return room_indicator
end


function nql:getQUpdate(args)
    local s, a, r, s2, term, delta
    local q, q2, q2_max

    s = args.s
    a = args.a
    r = args.r
    s2 = args.s2
    term = args.term
    term = term:clone():float():mul(-1):add(1)
    mode = args.mode -- update sr or R

    local ftrs, delta_m, delta_r, m2_max

    delta_m = {}
    for i=1,self.n_actions do
        delta_m[i] = torch.zeros(self.minibatch_size, self.srdims):float()
    end
    delta_r = torch.zeros(self.minibatch_size):float()
    delta_r = torch.reshape(delta_r, delta_r:size(1), 1)
    m2_max = torch.zeros(self.minibatch_size, self.srdims):float()
    local reconstruction = torch.zeros(self.minibatch_size, self.state_dim*self.hist_len)
    local sr_reconstruction = torch.zeros(self.minibatch_size, self.state_dim*self.hist_len)

    if mode == 'sr' then
        self.target_network:forward(s)
        ftrs = self.target_network.modules[#self.target_network.modules].modules[2].output:clone()

        -- Compute max_a M(s_2, a).
        m2, r2 = unpack(self.target_network:forward(s2))
        ftrs2 = self.target_network.modules[#self.target_network.modules].modules[2].output:clone()
        for i=1,#m2 do
            m2[i] = m2[i]:float()
        end

        ---- updates on SR -----
        -- find action a' that maximizes m2
        -- m2_max = m2:max(2)
        
        target_reward_weights = self.target_network.modules[#self.target_network.modules].modules[5].modules[2].modules[3].modules[1].modules[2].weight:clone()
        target_reward_weights = target_reward_weights:float():transpose(1,2)

        m2_max = m2[1]:clone()
        local batch_max_scores = torch.zeros(self.minibatch_size)
        local batch_max_indices = torch.ones(self.minibatch_size)
        local prev_batch_max_indices = batch_max_indices:clone()
        local prev_m2_max = m2_max:clone()

        batch_max_scores = m2[1]*target_reward_weights:squeeze()
        -- print('m2sum, ', m2:sum(), ' weight_sum: ', target_reward_weights:sum())
        for i=2,#m2 do -- i  is action ID
            current_max_scores = m2[i]*target_reward_weights:squeeze()
            local masking = torch.ge(current_max_scores, batch_max_scores)
            local original_masking = masking:clone():float()
            -- replicates masking feature size (64) times
            masking = torch.repeatTensor(masking, m2[i]:size()[2],1):transpose(1,2):float() 
            -- for each batch change only if current score is higher
            local inv_mask = masking:clone()
            m2_max = torch.cmul(prev_m2_max, inv_mask:mul(-1):add(1)) + torch.cmul(m2[i], masking)

            -- do the same for recording action indexes that change because of higher score
            local orig_inv_mask = original_masking:clone()

            batch_max_indices = torch.cmul(prev_batch_max_indices, orig_inv_mask:mul(-1):add(1)) + torch.cmul(torch.ones(self.minibatch_size)*i, original_masking) 

            prev_batch_max_indices = batch_max_indices:clone()
            prev_m2_max = m2_max:clone()
            batch_max_scores = m2_max*target_reward_weights:squeeze()
        end

        -- m2_max = m2[2]:clone()
        
        -- Compute m2 = (1-terminal) * gamma * max_a M(s2, a)
        local term_repeat = term:clone()
        term_repeat = torch.repeatTensor(term_repeat, m2[1]:size()[2],1):transpose(1,2):float()
        local m2_tmp = m2_max:clone():mul(self.discount):cmul(term_repeat)
        
        local term_original_repeate = term:clone():float()
        term_original_repeate = term_original_repeate:mul(-1):add(1)
        term_original_repeate = torch.repeatTensor(term_original_repeate, m2[1]:size()[2],1):transpose(1,2):float()
        m2_tmp:add(ftrs2:mul(self.discount):float():cmul(term_original_repeate))

        -- if self.rescale_r then
        --     delta_r:div(self.r_max)
        -- end

        -- q = Q(s,a)
        local m_all, r_unused = unpack(self.network:forward(s))
        -- local ftrs = self.network.modules[#self.network.modules].modules[2].output:clone()
        local m_pred = {}
        for i=1,#m_all do
            m_pred[i] = m_all[i]:clone():zero():float()
        end
        for i=1,self.minibatch_size do
            m_pred[a[i]][i] = m_all[a[i]][i]:clone():float()
            m_pred[a[i]][i]:mul(-1):add(m2_tmp[i])
        end

        ftrs = ftrs:float()
        ftrs_final = {}
        for i=1,#m_pred do
            ftrs_final[i] = ftrs:clone():zero():float()
        end

        for i=1,self.minibatch_size do
            ftrs_final[a[i]][i] = ftrs[i]:clone()
        end

        delta_m = ftrs_final
        for i=1,#m_pred do

            delta_m[i]:add(m_pred[i]:clone()) 
        end

        if self.clip_delta then
            for i=1,#delta_m do
                delta_m[i][delta_m[i]:ge(self.clip_delta)] = self.clip_delta
                delta_m[i][delta_m[i]:le(-self.clip_delta)] = -self.clip_delta
            end
        end
        if self.gpu >= 0 then 
            for i=1,#delta_m do
                delta_m[i] = delta_m[i]:cuda() 
                -- delta_m[i]:mul(0.01)
                end
        end

    else
        


        m2_max_clone = m2_max:clone()
        if self.gpu >= 0 then m2_max_clone = m2_max_clone:cuda() end

        sr_reconstruction = self.network.modules[#self.network.modules].modules[5].modules[2].modules[3].modules[2]:forward(m2_max_clone):clone()

        local m_unused, reward_channel = unpack(self.network:forward(s2))
        local rpred_s2 = reward_channel[1]:clone():float()
        reconstruction = reward_channel[2]:clone()

        --- updates on reward ----
        delta_r = rpred_s2:clone()
        delta_r:mul(-1)
        delta_r:add(r:clone():float())

        if self.clip_delta then
            delta_r[delta_r:ge(self.clip_delta)] = self.clip_delta
            delta_r[delta_r:le(-self.clip_delta)] = -self.clip_delta
        end

        if self.gpu >= 0 then 
            delta_r = delta_r:cuda()
        end

        delta_r = torch.reshape(delta_r, delta_r:size(1), 1)
        if false then --a[1]==2 and term[1] == 0 then-- term[1] == 0 then
            local num_displays = 12

            local s2_vis = s2[{{1,num_displays}}]:reshape(num_displays, self.hist_len*self.ncols, 84,84)
            s2_vis = s2_vis[{{},{1,self.ncols},{},{}}];disp.image(s2_vis, {win=1, title='observed'})

            local reconstruction_vis = reconstruction[{{1,num_displays}}]:reshape(num_displays, self.hist_len*self.ncols , 84,84)
            reconstruction_vis = reconstruction_vis[{{},{1,self.ncols},{},{}}];disp.image(reconstruction_vis, {win=2, title='predictions'})
        end

    end

    if self.gpu >= 0 then 
        reconstruction = reconstruction:cuda()
    end

    return delta_r, delta_m, m2_max, reconstruction
end



function nql:motionScaling(s2, grads)
    local scaling = 3
    local s2_reshaped = s2:reshape(self.minibatch_size, self.input_dims[1], self.input_dims[2], self.input_dims[3])
    local residual = s2_reshaped:clone():zero()

    for i=1,self.hist_len do
        if i == 1 then
            residual[{{},i,{},{}}] = s2_reshaped[{{}, i,{},{}}] - s2_reshaped[{{}, i+1,{},{}}]
        elseif i == self.hist_len then
            residual[{{},i,{},{}}] = s2_reshaped[{{}, i,{},{}}] - s2_reshaped[{{}, i-1,{},{}}]
        else
            local tmp1 = (s2_reshaped[{{}, i,{},{}}] - s2_reshaped[{{}, i-1,{},{}}]):abs()
            local tmp2 = (s2_reshaped[{{}, i,{},{}}] - s2_reshaped[{{}, i+1,{},{}}]):abs()
            residual[{{},i,{},{}}] = tmp1 + tmp2
        end
        residual[{{},i,{},{}}] = residual[{{},i,{},{}}]:abs()
        residual[{{},i,{},{}}] = torch.ge(residual[{{},i,{},{}}], 0.01) * (scaling-1)
        residual[{{},i,{},{}}] = residual[{{},i,{},{}}] + 1.0
    end

    residual = residual:reshape(self.minibatch_size*self.input_dims[1]*self.input_dims[2]*self.input_dims[3])
    grads = grads:cmul(residual)
    return grads
end


function nql:qLearnMinibatch(mode)
    -- Perform a minibatch Q-learning update:
    -- w += alpha * (r + gamma max Q(s2,a2) - Q(s,a)) * dQ(s,a)/dw
    assert(self.transitions:size() > self.minibatch_size)

    local s, a, r, s2, term = self.transitions:sample(self.minibatch_size)

    if self.game_name == "doom" and mode == "reward" then
        --prioritize s2 w.r.t reward
        local max_buffer_size = 10
        local buf_len = math.min(#self.s2_buf, max_buffer_size) 
        if buf_len > 0 then
            if buf_len >= max_buffer_size then
                local rindxs = torch.randperm(buf_len)
                for ii=1,buf_len do
                    s2[self.minibatch_size-buf_len+ii] = self.s2_buf[rindxs[ii]]
                    r[self.minibatch_size-buf_len+ii] = self.reward_buf[rindxs[ii]]
                end
            else
                for ii = 1,buf_len do
                    s2[self.minibatch_size-buf_len+ii] = self.s2_buf[ii]
                    r[self.minibatch_size-buf_len+ii] = self.reward_buf[ii]
                end
            end
        end
    end
    local delta_r, delta_m, m2_max, reconstruction = self:getQUpdate{s=s, a=a, r=r, s2=s2,
        term=term, update_qmax=true, mode=mode}


    -- zero gradients of parameters
    self.dw:zero()

    local s2_flatten = s2:clone()
    local bce_err = self.criterion:forward(s2_flatten, reconstruction)
    local gradCriterion = self.criterion:backward(s2_flatten, reconstruction)

    if self.hist_len > 1 then
        gradCriterion = self:motionScaling(s2_flatten, gradCriterion)
    end

    -- in getQUpdate, the last forward call is with s2, so backward delta_r
    local delta_m_tmp = {}
    for i=1,#delta_m do
        -- delta_m_tmp[i] = delta_m[i]:clone(); delta_m_tmp[i]:zero()
        delta_m_tmp[i] = delta_m[i]:clone():zero()
        if self.gpu >= 0 then delta_m_tmp[i] = delta_m_tmp[i]:cuda() end
    end
    local delta_r_tmp = delta_r:clone(); delta_r_tmp:zero()
    if self.gpu >= 0 then delta_r_tmp = delta_r_tmp:cuda() end

    if mode == 'reward' then
        self.network:backward(s2, {delta_m_tmp, {delta_r, gradCriterion} })
    end

    if mode == 'sr' then
        -- in getQUpdate, the last forward call is with s2, so backward delta_r
        -- self.network:forward(s)
        gradCriterion:zero()
        self.network:backward(s, {delta_m, {delta_r_tmp, gradCriterion}})
    end

    -- add weight cost to gradient
    self.dw:add(-self.wc, self.w)


    -- use gradients
    if mode == 'sr' then
        ------------------------ for sr ----------------------
        -- compute linearly annealed learning rate
        local t = math.max(0, self.numSteps - self.learn_start)
        self.lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                    self.lr_end
        self.lr = math.max(self.lr, self.lr_end)

        self.g:mul(0.95):add(0.05, self.dw)
        self.tmp:cmul(self.dw, self.dw)
        self.g2:mul(0.95):add(0.05, self.tmp)
        self.tmp:cmul(self.g, self.g)
        self.tmp:mul(-1)
        self.tmp:add(self.g2)
        self.tmp:add(0.01)
        self.tmp:sqrt()
        -- accumulate update
        self.deltas:mul(0):addcdiv(self.lr, self.dw, self.tmp)
        self.w:add(self.deltas)
    else
        ------------------------ for reward channel ----------------------
        -- compute linearly annealed learning rate
        local t = math.max(0, self.R_numSteps - self.learn_start)
        self.R_lr = (self.lr_start - self.lr_end) * (self.lr_endt - t)/self.lr_endt +
                    self.lr_end
        self.R_lr = math.max(self.R_lr, self.lr_end)

        self.R_g:mul(0.95):add(0.05, self.dw)
        self.R_tmp:cmul(self.dw, self.dw)
        self.R_g2:mul(0.95):add(0.05, self.R_tmp)
        self.R_tmp:cmul(self.R_g, self.R_g)
        self.R_tmp:mul(-1)
        self.R_tmp:add(self.R_g2)
        self.R_tmp:add(0.01)
        self.R_tmp:sqrt()
        -- accumulate update
        self.R_deltas:mul(0):addcdiv(self.R_lr, self.dw, self.R_tmp)
        self.w:add(self.R_deltas)
    end        
end


function nql:sample_validation_data()
    local s, a, r, s2, term = self.transitions:sample(self.valid_size)
    self.valid_s    = s:clone()
    self.valid_a    = a:clone()
    self.valid_r    = r:clone()
    self.valid_s2   = s2:clone()
    self.valid_term = term:clone()
end


function nql:compute_validation_statistics()
    local delta_r_tmp, delta_m, m2_max, recons = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term, mode='sr'}
    local delta_r, delta_m_tmp, m2_max_tmp = self:getQUpdate{s=self.valid_s,
        a=self.valid_a, r=self.valid_r, s2=self.valid_s2, term=self.valid_term, mode='reward'}

    self.v_avg = m2_max:mean()
    self.r_tderr_avg = delta_r[1]:clone():abs():mean()
    self.m_tderr_avg = delta_m[1]:clone():abs():mean()
    self.reconstruction_avg =  (recons - self.valid_s):pow(2):sqrt():mean()
    for i=2,#delta_m do
        self.m_tderr_avg = self.m_tderr_avg + delta_m[i]:clone():abs():mean()
    end
end


function nql:perceive(reward, rawstate, terminal, testing, testing_ep)

    local chunk_size = 100
    if self.sample_collect and self.numSteps >= self.num_samples and self.numSteps > 2*chunk_size then 
        -- do a forward pass and extract msa, where a is randomly chosen among set of actions A
        for i=1,desired_samples/chunk_size do

            local s, a, r, s2, term = self.transitions:sample(chunk_size)

            current_m, current_r = unpack(self.network:forward(s))
            
            local random_action_indices = torch.random(torch.zeros(chunk_size),1,4)

            ftrs_len = current_m[1][1]:size(1)
            m_values = {}
            s_values = {}
            for j=1,chunk_size do
                m_clone_svd = current_m[random_action_indices[j]][j]:reshape(1,ftrs_len):float()
                s_clone_svd = s[j]:reshape(1,self.state_dim):float()
                table.insert(m_values, m_clone_svd)
                table.insert(s_values, s_clone_svd)
            end
            
            merge_m = nn.JoinTable(1)
            current_m_tensor = merge_m:forward(m_values)
            merge_s = nn.JoinTable(1)
            current_s_tensor = merge_s:forward(s_values)

            m_tensor = m_tensor or current_m_tensor
            s_tensor = s_tensor or current_s_tensor

            merge_m = nn.JoinTable(1)
            m_tensor = merge_m:forward({m_tensor,current_m_tensor})
            merge_s = nn.JoinTable(1)
            s_tensor = merge_s:forward({s_tensor,current_s_tensor})

        end
        name = '../' .. self.game_name .. '.h5'
        print ("SAVING")
        myFile = hdf5.open(name, 'w')
        myFile:write('m_full_tensor', m_tensor)
        myFile:write('s_full_tensor', s_tensor)
        self.sample_collect = 0
    end

    -- Preprocess state (will be set to nil if terminal)
    local state = self:preprocess(rawstate, self.ncols==1):float()

    local curState

    if self.max_reward then
        reward = math.min(reward, self.max_reward)
    end
    if self.min_reward then
        reward = math.max(reward, self.min_reward)
    end
    if self.rescale_r then
        self.r_max = math.max(self.r_max, reward)
    end

    self.transitions:add_recent_state(state, terminal)

    local currentFullState = self.transitions:get_recent()

    --Store transition s, a, r, s'
    if self.lastState and not testing then
                self.transitions:add(self.lastState, self.lastAction, reward,
                             self.lastTerminal, priority)
    end

    if self.numSteps == self.learn_start+1 and not testing then
        self:sample_validation_data()
    end

    curState= self.transitions:get_recent()
    

    if reward > 0 then
        self.s2_buf[#self.s2_buf+1] = curState:clone()
        self.reward_buf[#self.reward_buf+1] = reward
    end


    curState = curState:resize(1, unpack(self.input_dims))

    -- Select action
    local actionIndex = 1
    if not terminal then
        actionIndex = self:eGreedy(curState, testing, testing_ep)
    end
    self.transitions:add_recent_action(actionIndex)

    --Do some Q-learning updates
    local reward_tr_start = 2000
    local reward_tr_steps = 12000 
    if self.game_name == "doom" then
        reward_tr_start = 15000
        reward_tr_steps =  50000
    end

    if not testing then
        if self.numSteps > reward_tr_start and self.numSteps %3000 == 1 then
            self.num_steps = self.num_steps or reward_tr_steps
            self.num_steps = self.num_steps/2
            print('Training reward network for ' ..  self.num_steps .. ' steps')
            for i=1,self.num_steps do
                xlua.progress(i, self.num_steps)
                self:qLearnMinibatch('reward') --update only reward
                self.R_numSteps = self.R_numSteps + 1
            end 
        end
    end

    if self.numSteps > self.learn_start and not testing and
        self.numSteps % self.update_freq == 0 then
        for i = 1, self.n_replay do
            self:qLearnMinibatch('sr') --update only SR
        end
    end

    if not testing then
        self.numSteps = self.numSteps + 1
    end

    self.lastState = state:clone()

    self.lastAction = actionIndex
    self.lastTerminal = terminal

    if self.target_q and self.numSteps % self.target_q == 1 then
        self.target_network = self.network:clone()
    end
    
    if not terminal then
        return actionIndex
    else
        return 0
    end
end


function nql:eGreedy(state, testing, testing_ep)
    self.ep = testing_ep or (self.ep_end +
                math.max(0, (self.ep_start - self.ep_end) * (self.ep_endt -
                math.max(0, self.numSteps - self.learn_start))/self.ep_endt))

    -- Epsilon greedy
    if torch.uniform() < self.ep then
        return torch.random(1, self.n_actions)
    else
        return self:greedy(state, testing)
    end

end


function nql:greedy(state, testing)

    -- Turn single state into minibatch.  Needed for convolutional nets.
    if state:dim() == 2 then
        assert(false, 'Input must be at least 3D')
        state = state:resize(1, state:size(1), state:size(2))
    end

    if self.gpu >= 0 then
        state = state:cuda()
    end

    local m_tab, r_tab = unpack(self.network:forward(state))
    m_tab_pred = {}

    for i=1,#m_tab do
        m_tab_pred[i] = m_tab[i]:float()
    end


    local reward_weights = self.network.modules[#self.network.modules].modules[5].modules[2].modules[3].modules[1].modules[2].weight:clone():float()
    local max_score = torch.dot(m_tab_pred[1], reward_weights)
    local besta = {1}

    -- Evaluate all other actions (with random tie-breaking)
    for a = 2, self.n_actions do
        local score = torch.dot(m_tab_pred[a], reward_weights)

        if score > max_score then
            besta = { a }
            max_score = score

        elseif score == max_score then
            besta[#besta+1] = a
        end
    end
    self.bestq = max_score
    local tie_break_r = torch.random(1, #besta)
    self.lastAction = besta[tie_break_r]
    return besta[tie_break_r]
end


function nql:createNetwork()
    local n_hid = 128
    local mlp = nn.Sequential()
    mlp:add(nn.Reshape(self.hist_len*self.ncols*self.state_dim))
    mlp:add(nn.Linear(self.hist_len*self.ncols*self.state_dim, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, n_hid))
    mlp:add(nn.Rectifier())
    mlp:add(nn.Linear(n_hid, self.n_actions))

    return mlp
end


function nql:_loadNet()
    local net = self.network
    if self.gpu then
        net:cuda()
    else
        net:float()
    end
    return net
end


function nql:init(arg)
    self.actions = arg.actions
    self.n_actions = #self.actions
    self.network = self:_loadNet()
    -- Generate targets.
    self.transitions:empty()
end


function nql:report()
    -- print(get_weight_norms(self.network))
    -- print(get_grad_norms(self.network))
end
