local MovingGoals, parent = torch.class('MovingGoals', 'MazeBase')

math = require ('math')

function MovingGoals:__init(opts, vocab)
    opts.height = 3
    opts.width = 3
    parent.__init(self, opts, vocab)
    --block_locs = {{3,5}, {4,5}, {5,5}, {6,5}, {7,5}, {8,5}, {3,6}, {3,7}, {3,8}, {8,6}, {8,7}, {8,8}}
    --block_locs = {{1,1}, {1,2}, {1,3}, {2,1}, {2,3}, {3,1}}
    block_locs = {{2,1}, {2,2}} -- {1,1},{1,2}
    for i = 1,#block_locs do
        self:place_item({type = 'block'},block_locs[i][1],block_locs[i][2])
    end
    goal_one = {3,3}

    self.goal = self:place_item({type = 'goal', name = 'goal' .. 1}, goal_one[1], goal_one[2])


    if self.agent == nil then
        self.agents = {}
        for i = 1, self.nagents do
            self.agents[i] =  self:place_item({type = 'agent'}, 1, 1)
        end
        self.agent = self.agents[1]
    end
end

function MovingGoals:update()
    parent.update(self)
    if not self.finished then
        if self.goal.loc.y == self.agent.loc.y and self.goal.loc.x == self.agent.loc.x then
            self.finished = true
        end
    end
end

function MovingGoals:get_reward()
    -- if self.finished then
    --     return -self.costs.goal
    -- else
    --     return parent.get_reward(self)
    -- end
    if self.finished then
        return 1
    else
        return -1---1
    end
end
