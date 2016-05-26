local Bottleneck2Rooms, parent = torch.class('Bottleneck2Rooms', 'MazeBase')

math = require ('math')

function Bottleneck2Rooms:__init(opts, vocab)
    opts.height = 5
    opts.width = 5
    parent.__init(self, opts, vocab)

    block_locs = {{1,3}, {2,3}, {3,3}, {5,3}}

    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end
    goal_one = {1,1}

    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 1, 5)
        end
        self.agent = self.agents[1]
    end
end

function Bottleneck2Rooms:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function Bottleneck2Rooms:get_reward()
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
