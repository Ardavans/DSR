local MovingGoalsMedium, parent = torch.class('MovingGoalsMedium', 'MazeBase')

math = require ('math')

function MovingGoalsMedium:__init(opts, vocab)
    opts.height = 7
    opts.width = 7
    parent.__init(self, opts, vocab)

    water_locs = {{1,1},{3,2},{3,3},{3,4},{3,5},{4,2},{4,3}}
    for i = 1,#water_locs do
        self:place_item({type = 'water'},water_locs[i][1],water_locs[i][2])
    end

    block_locs = {{2,2}, {2,3}, {2,5},{4,4},{4,5},{4,6},{4,7},{5,3},{6,4},{6,6}} -- {1,1},{1,2}
    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end

    goal_one = {3,6}
    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 7, 1)
        end
        self.agent = self.agents[1]
    end
end

function MovingGoalsMedium:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function MovingGoalsMedium:get_reward()
    if self.finished then
        return self.costs.goal
    else
        return -parent.get_reward(self)
    end
end
