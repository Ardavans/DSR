--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require 'image'

local trans_priority = torch.class('dsr.TransitionTable_priority')


function trans_priority:__init(args)
    self.stateDim = args.stateDim
    self.numActions = args.numActions
    self.histLen = args.histLen
    self.maxSize = args.maxSize or 1024^2
    self.bufferSize = args.bufferSize or 1024
    self.histType = args.histType or "linear"
    self.histSpacing = args.histSpacing or 1
    self.zeroFrames = args.zeroFrames or 1
    self.nonTermProb = args.nonTermProb or 1
    self.nonEventProb = args.nonEventProb or 1
    self.gpu = args.gpu
    self.numEntries = 0
    self.insertIndex = 0
    self.ptrInsertIndex = 1 --new

    self.histIndices = {}
    local histLen = self.histLen
    if self.histType == "linear" then
        -- History is the last histLen frames.
        self.recentMemSize = self.histSpacing*histLen
        for i=1,histLen do
            self.histIndices[i] = i*self.histSpacing
        end
    elseif self.histType == "exp2" then
        -- The ith history frame is from 2^(i-1) frames ago.
        self.recentMemSize = 2^(histLen-1)
        self.histIndices[1] = 1
        for i=1,histLen-1 do
            self.histIndices[i+1] = self.histIndices[i] + 2^(7-i)
        end
    elseif self.histType == "exp1.25" then
        -- The ith history frame is from 1.25^(i-1) frames ago.
        self.histIndices[histLen] = 1
        for i=histLen-1,1,-1 do
            self.histIndices[i] = math.ceil(1.25*self.histIndices[i+1])+1
        end
        self.recentMemSize = self.histIndices[1]
        for i=1,histLen do
            self.histIndices[i] = self.recentMemSize - self.histIndices[i] + 1
        end
    end

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    -- self.r = torch.zeros(self.maxSize, 2)
    self.r = torch.zeros(self.maxSize)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    self.action_encodings = torch.eye(self.numActions)
    self.end_ptrs = {} --new
    self.dyn_ptrs = {} --new
    self.trace_indxs_with_extreward = {} --extrinsic reward (new)
    -- self.trace_indxs_with_intreward = {} --intrinsic reward
    
    -- self.subgoal_dims = args.subgoal_dims*9 --TODO (total number of objects)
    -- self.subgoal = torch.zeros(self.maxSize, self.subgoal_dims) 

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}
    -- self.recent_subgoal = {}

    local s_size = self.stateDim*histLen
    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    -- self.buf_r      = torch.zeros(self.bufferSize,2 )
    self.buf_r      = torch.zeros(self.bufferSize)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, s_size):fill(0)
    -- self.buf_subgoal = torch.zeros(self.bufferSize, self.subgoal_dims)
    -- self.buf_subgoal2 = torch.zeros(self.bufferSize, self.subgoal_dims)
    

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
        -- self.gpu_subgoal = self.buf_subgoal:float():cuda()
        -- self.gpu_subgoal2 = self.buf_subgoal2:float():cuda()
    end
end


function trans_priority:reset()
    self.numEntries = 0
    self.insertIndex = 0
    self.ptrInsertIndex = 1 --new
end


function trans_priority:size()
    return self.numEntries
end


function trans_priority:empty()
    return self.numEntries == 0
end


function trans_priority:fill_buffer()
    assert(self.numEntries >= self.bufferSize)
    -- clear CPU buffers
    self.buf_ind = 1
    local ind
    for buf_ind=1,self.bufferSize do
        -- local s, a, r, s2, term, subgoal, subgoal2 = self:sample_one(1)
        local s, a, r, s2, term = self:sample_one(1)
        self.buf_s[buf_ind]:copy(s)
        self.buf_a[buf_ind] = a
        -- self.buf_subgoal[buf_ind] = subgoal
        -- self.buf_subgoal2[buf_ind] = subgoal2
        self.buf_r[buf_ind] = r
        self.buf_s2[buf_ind]:copy(s2)
        self.buf_term[buf_ind] = term
    end
    self.buf_s  = self.buf_s:float():div(255)
    self.buf_s2 = self.buf_s2:float():div(255)
    if self.gpu and self.gpu >= 0 then
        self.gpu_s:copy(self.buf_s)
        self.gpu_s2:copy(self.buf_s2)
        -- self.gpu_subgoal:copy(self.buf_subgoal)
        -- self.gpu_subgoal2:copy(self.buf_subgoal2) 
    end
end

function trans_priority:get_size(tab)
    if tab == nil then return 0 end
    local Count = 0
    for Index, Value in pairs(tab) do
      Count = Count + 1
    end
    return Count
end

function trans_priority:get_canonical_indices()
    local indx;
    local index = -1
    while index <= 0 do
        indx = torch.random(#self.end_ptrs-1)
        index = self.dyn_ptrs[indx] - self.recentMemSize + 1
    end
    return indx, index
end

function trans_priority:sample_one()
    assert(self.numEntries > 1)
    assert(#self.end_ptrs == #self.dyn_ptrs)
    -- print(self.end_ptrs)
    local index = -1
    local indx

    --- choose to either select traces with external or internal reward
    local chosen_trace_indxs = self.trace_indxs_with_extreward
    -- if self:get_size(self.trace_indxs_with_extreward) == 0 then
    --     chosen_trace_indxs = self.trace_indxs_with_intreward
    -- else
    --     if torch.uniform() > 0.5 then
    --         chosen_trace_indxs = self.trace_indxs_with_intreward
    --     end
    -- end

    local eps = 0.8; 

    if torch.uniform() < eps or self:get_size(chosen_trace_indxs) <= 0 then 
        --randomly sample without prioritization
        indx, index = self:get_canonical_indices()
    else
        -- prioritize and pick from stored trans_priorityitions with rewards
        --this is only executed if #chosen_trace_indxs > 0, i.e. only if agent has received external reward
        while index <= 0  do 
            local keyset={}; local n=0;
            for k,v in pairs(chosen_trace_indxs) do
                if k <= self.maxSize - self.histLen + 1 then
                    n=n+1
                    keyset[n]=k    
                end
            end
            -- print('K:', keyset)
            if #keyset == 0 then 
                indx, index = self:get_canonical_indices()
                break
            end
            local mem_indx = keyset[torch.random(#keyset)]   
            -- print('mem_indx:', mem_indx)
            -- print('R:', chosen_trace_indxs)
            -- print('DYN:', self.dyn_ptrs)
            -- print('mem_indx:', mem_indx)
            -- print('END:', self.end_ptrs)
            for k,v in pairs(self.end_ptrs) do
                if v == mem_indx then
                    indx = k
                end
            end
            if indx then
                index = self.dyn_ptrs[indx] - self.recentMemSize + 1
            else
                indx, index = self:get_canonical_indices()
                break
            end
            -- this is a corner case: when there is only 2 eps (fix this TODO) with reward but index is zero
            if index <= 0 and self:get_size(chosen_trace_indxs) <= 2 then
                indx, index = self:get_canonical_indices()
                -- print('INDEX:', index)
                break
            end
        end
    end
    -- print(index, indx)
    self.dyn_ptrs[indx] = self.dyn_ptrs[indx] - 1
    if self.dyn_ptrs[indx] <= 0 or self.dyn_ptrs[indx] == self.end_ptrs[indx-1] then
        self.dyn_ptrs[indx] = self.end_ptrs[indx]
    end
    return self:get(index)
end



function trans_priority:sample(batch_size)
    local batch_size = batch_size or 1
    assert(batch_size < self.bufferSize)

    if not self.buf_ind or self.buf_ind + batch_size - 1 > self.bufferSize then
        self:fill_buffer()
    end

    local index = self.buf_ind

    self.buf_ind = self.buf_ind+batch_size
    local range = {{index, index+batch_size-1}}

    -- local buf_s, buf_s2, buf_a, buf_r, buf_term, buf_subgoal, buf_subgoal2 = self.buf_s, self.buf_s2,
    --     self.buf_a, self.buf_r, self.buf_term, self.buf_subgoal, self.buf_subgoal2
    local buf_s, buf_s2, buf_a, buf_r, buf_term = self.buf_s, self.buf_s2,
        self.buf_a, self.buf_r, self.buf_term
    if self.gpu and self.gpu >=0  then
        buf_s = self.gpu_s
        buf_s2 = self.gpu_s2
        -- buf_subgoal = self.gpu_subgoal
        -- buf_subgoal2 = self.gpu_subgoal2 
    end

    -- return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range], buf_subgoal[range], buf_subgoal2[range]
    return buf_s[range], buf_a[range], buf_r[range], buf_s2[range], buf_term[range]
end


function trans_priority:concatFrames(index, use_recent)
    if use_recent then
        -- s, t, subgoal = self.recent_s, self.recent_t, self.recent_subgoal[self.histLen]
        s, t = self.recent_s, self.recent_t
    else
        -- s, t, subgoal = self.s, self.t, self.subgoal[index]
        s, t = self.s, self.t

    end

    local fullstate = s[1].new()
    fullstate:resize(self.histLen, unpack(s[1]:size():totable()))

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            fullstate[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        fullstate[i]:copy(s[index+self.histIndices[i]-1])  
    end
    -- return fullstate, subgoal
    return fullstate
end


function trans_priority:concatActions(index, use_recent)
    local act_hist = torch.FloatTensor(self.histLen, self.numActions)
    if use_recent then
        a, t = self.recent_a, self.recent_t
    else
        a, t = self.a, self.t
    end

    -- Zero out frames from all but the most recent episode.
    local zero_out = false
    local episode_start = self.histLen

    for i=self.histLen-1,1,-1 do
        if not zero_out then
            for j=index+self.histIndices[i]-1,index+self.histIndices[i+1]-2 do
                if t[j] == 1 then
                    zero_out = true
                    break
                end
            end
        end

        if zero_out then
            act_hist[i]:zero()
        else
            episode_start = i
        end
    end

    if self.zeroFrames == 0 then
        episode_start = 1
    end

    -- Copy frames from the current episode.
    for i=episode_start,self.histLen do
        act_hist[i]:copy(self.action_encodings[a[index+self.histIndices[i]-1]])
    end

    return act_hist
end


function trans_priority:get_recent()
    -- Assumes that the most recent state has been added, but the action has not
    -- return self:concatFrames(1, true):float():div(255)
    
    -- local fullstate, subgoal = self:concatFrames(1,true)
    local fullstate = self:concatFrames(1,true)
    -- return fullstate:float():div(255), subgoal
    return fullstate:float():div(255)

end


function trans_priority:get(index)
    -- local s, subgoal = self:concatFrames(index)
    -- local s2, subgoal2 = self:concatFrames(index+1)
    local s = self:concatFrames(index)
    local s2 = self:concatFrames(index+1)
    local ar_index = index+self.recentMemSize-1
    -- print(index)
    -- return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1], self.subgoal[ar_index], self.subgoal[ar_index+1]
    return s, self.a[ar_index], self.r[ar_index], s2, self.t[ar_index+1]

end


function trans_priority:add(s, a, r, term)
    -- print('TT:', term, r)
    assert(s, 'State cannot be nil')
    assert(a, 'Action cannot be nil')
    assert(r, 'Reward cannot be nil')

    -- Incremenet until at full capacity
    if self.numEntries < self.maxSize then
        self.numEntries = self.numEntries + 1
    end

    -- Always insert at next index, then wrap around
    self.insertIndex = self.insertIndex + 1



    -- Overwrite oldest experience once at capacity
    if self.insertIndex > self.maxSize then
        self.insertIndex = 1
        self.ptrInsertIndex = 1
    end

    -- Overwrite (s,a,r,t) at insertIndex
    self.s[self.insertIndex] = s:clone():float():mul(255)
    self.a[self.insertIndex] = a
    self.r[self.insertIndex] = r
    -- self.subgoal[self.insertIndex] = subgoal

    if r > 0 then --if extrinsic reward is non-zero, record this!
        self.trace_indxs_with_extreward[self.insertIndex] = 1
    end

    -- local intrinsic_reward = r[2] - r[1]
    -- if intrinsic_reward > 0 then --if extrinsic reward is non-zero, record this!
    --     self.trace_indxs_with_intreward[self.insertIndex] = 1
    -- end
    
    if self.end_ptrs[self.ptrInsertIndex] == self.insertIndex then
        table.remove(self.end_ptrs,self.ptrInsertIndex)
        table.remove(self.dyn_ptrs,self.ptrInsertIndex)
        self.trace_indxs_with_extreward[self.insertIndex] = nil
        -- self.trace_indxs_with_intreward[self.insertIndex] = nil 
    end
    if term then
        self.t[self.insertIndex] = 1
        table.insert(self.end_ptrs, self.ptrInsertIndex, self.insertIndex)
        table.insert(self.dyn_ptrs, self.ptrInsertIndex, self.insertIndex)
        self.ptrInsertIndex = self.ptrInsertIndex + 1
    else
        self.t[self.insertIndex] = 0
    end
    -- print(#self.end_ptrs, term)
end


function trans_priority:add_recent_state(s, term)
    local s = s:clone():float():mul(255):byte()
    -- local subgoal = subgoal:clone()
    if #self.recent_s == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_s, s:clone():zero())
            table.insert(self.recent_t, 1)
            -- table.insert(self.recent_subgoal, subgoal:clone():zero())
        end
    end

    table.insert(self.recent_s, s)
    -- table.insert(self.recent_subgoal, subgoal)
    if term then
        table.insert(self.recent_t, 1)
    else
        table.insert(self.recent_t, 0)
    end

    -- Keep recentMemSize states.
    if #self.recent_s > self.recentMemSize then
        table.remove(self.recent_s, 1)
        table.remove(self.recent_t, 1)
    end
end


function trans_priority:add_recent_action(a)
    if #self.recent_a == 0 then
        for i=1,self.recentMemSize do
            table.insert(self.recent_a, 1)
        end
    end

    table.insert(self.recent_a, a)

    -- Keep recentMemSize steps.
    if #self.recent_a > self.recentMemSize then
        table.remove(self.recent_a, 1)
    end
end


--[[
Override the write function to serialize this class into a file.
We do not want to store anything into the file, just the necessary info
to create an empty trans_priorityition table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans_priority:write(file)
    file:writeObject({self.stateDim,
                      self.numActions,
                      self.histLen,
                      self.maxSize,
                      self.bufferSize,
                      self.numEntries,
                      self.insertIndex,
                      self.recentMemSize,
                      self.histIndices})
                      -- self.subgoal_dims})
end


--[[
Override the read function to desearialize this class from file.
Recreates an empty table.

@param file (FILE object ) @see torch.DiskFile
--]]
function trans_priority:read(file)
    -- local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices, subgoal_dims = unpack(file:readObject())
    local stateDim, numActions, histLen, maxSize, bufferSize, numEntries, insertIndex, recentMemSize, histIndices = unpack(file:readObject())
    self.stateDim = stateDim
    self.numActions = numActions
    self.histLen = histLen
    self.maxSize = maxSize
    self.bufferSize = bufferSize
    self.recentMemSize = recentMemSize
    self.histIndices = histIndices
    self.numEntries = 0
    self.insertIndex = 0
    -- self.subgoal_dims = subgoal_dims

    self.s = torch.ByteTensor(self.maxSize, self.stateDim):fill(0)
    self.a = torch.LongTensor(self.maxSize):fill(0)
    self.r = torch.zeros(self.maxSize, 2)
    self.t = torch.ByteTensor(self.maxSize):fill(0)
    -- self.subgoal = torch.zeros(self.maxSize, self.subgoal_dims)
    self.action_encodings = torch.eye(self.numActions)

    -- Tables for storing the last histLen states.  They are used for
    -- constructing the most recent agent state more easily.
    self.recent_s = {}
    self.recent_a = {}
    self.recent_t = {}

    self.buf_a      = torch.LongTensor(self.bufferSize):fill(0)
    self.buf_r      = torch.zeros(self.bufferSize, 2)
    self.buf_term   = torch.ByteTensor(self.bufferSize):fill(0)
    self.buf_s      = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    self.buf_s2     = torch.ByteTensor(self.bufferSize, self.stateDim * self.histLen):fill(0)
    -- self.buf_subgoal = torch.zeros(self.bufferSize, self.subgoal_dims)
    -- self.buf_subgoal2 = torch.zeros(self.bufferSize, self.subgoal_dims)

    if self.gpu and self.gpu >= 0 then
        self.gpu_s  = self.buf_s:float():cuda()
        self.gpu_s2 = self.buf_s2:float():cuda()
        -- self.gpu_subgoal = self.buf_subgoal:float():cuda()
        -- self.gpu_subgoal2 = self.buf_subgoal2:float():cuda() 
    end
end