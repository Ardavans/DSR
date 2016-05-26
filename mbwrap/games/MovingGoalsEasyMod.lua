local MovingGoalsEasyMod, parent = torch.class('MovingGoalsEasyMod', 'MazeBase')

math = require ('math')

function MovingGoalsEasyMod:__init(opts, vocab)
    opts.height = 4
    opts.width = 4
    parent.__init(self, opts, vocab)

    water_locs = {{1,1}}
    for i = 1,#water_locs do
        self:place_item({type = 'water'},water_locs[i][1],water_locs[i][2])
    end

    block_locs = {{2,2}, {3,3}} -- {1,1},{1,2}
    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end

    goal_one = {2,3}
    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 4, 1)
        end
        self.agent = self.agents[1]
    end
end

function MovingGoalsEasyMod:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function MovingGoalsEasyMod:get_reward()
    if self.finished then
        return self.costs.goal
    else
        return -parent.get_reward(self)
    end
end
