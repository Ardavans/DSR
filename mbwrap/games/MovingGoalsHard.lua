local MovingGoalsHard, parent = torch.class('MovingGoalsHard', 'MazeBase')

math = require ('math')

function MovingGoalsHard:__init(opts, vocab)
    opts.height = 10
    opts.width = 10
    parent.__init(self, opts, vocab)

    water_locs = {{1,6},{2,2},{2,3},{2,4},{2,8},{2,9},{5,2},{7,3},{8,4}}
    for i = 1,#water_locs do
        self:place_item({type = 'water'},water_locs[i][1],water_locs[i][2])
    end

    block_locs = {{3,6},{4,4},{4,5},{4,7},{4,8},{4,9},{5,8},{6,4},{6,5},{6,7},{6,9},{7,6},{7,9},{7,10},{9,2},{9,3},{9,4},{9,6},{9,8},{9,9}} -- {1,1},{1,2}
    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end

    goal_one = {5,9}
    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 10, 1)
        end
        self.agent = self.agents[1]
    end
end

function MovingGoalsHard:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function MovingGoalsHard:get_reward()
    if self.finished then
        return self.costs.goal
    else
        return -parent.get_reward(self)
    end
end
