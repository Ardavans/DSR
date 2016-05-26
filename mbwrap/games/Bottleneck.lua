local Bottleneck, parent = torch.class('Bottleneck', 'MazeBase')

math = require ('math')

function Bottleneck:__init(opts, vocab)
    opts.height = 7
    opts.width = 7
    parent.__init(self, opts, vocab)

    block_locs = {{1,4}, {3,4}, {4,4}, {4,5}, {4,6}, {4,7}, {5,2}, {5,3}, {5,4}, {6,4}}

    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end
    goal_one = {3,7}

    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 6, 6)
        end
        self.agent = self.agents[1]
    end
end

function Bottleneck:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function Bottleneck:get_reward()
    -- if self.finished then
    --     return -self.costs.goal
    -- else
    --     return parent.get_reward(self)
    -- end
    if self.finished then
        return 1
    else
        return -1
    end
end
