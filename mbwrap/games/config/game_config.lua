if not g_opts then g_opts = {} end

g_opts.multigames = {}
-------------------
--some shared RangeOpts
--current min, current max, min max, max max, increment
local mapH = torch.Tensor{5,10,5,10,1}
local mapW = torch.Tensor{5,10,5,10,1}
local blockspct = torch.Tensor{0,.2,0,.2,.01}
local waterpct = torch.Tensor{0,.2,0,.2,.01}


-------------------
--some shared StaticOpts
local sso = {}
-------------- costs:
sso.costs = {}
sso.costs.goal = 0
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.2
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 1
sso.flag_visited = 1
sso.enable_boundary = 0
sso.enable_corners = 1
sso.max_attributes = g_opts.max_attributes or 6

-------------------------------------------------------
-- MultiGoals:
local MultiGoalsRangeOpts = {}
MultiGoalsRangeOpts.mapH = mapH:clone()
MultiGoalsRangeOpts.mapW = mapW:clone()
MultiGoalsRangeOpts.blockspct = blockspct:clone()
MultiGoalsRangeOpts.waterpct = waterpct:clone()
MultiGoalsRangeOpts.ngoals = torch.Tensor{2,6,3,6,1}
MultiGoalsRangeOpts.ngoals_active = torch.Tensor{1,3,1,3,1}

local MultiGoalsStaticOpts = {}
for i,j in pairs(sso) do MultiGoalsStaticOpts[i] = j end


MultiGoalsOpts ={}
MultiGoalsOpts.RangeOpts = MultiGoalsRangeOpts
MultiGoalsOpts.StaticOpts = MultiGoalsStaticOpts

g_opts.multigames.MultiGoals = MultiGoalsOpts


-------------------------------------------------------
-- CondGoals:
local CondGoalsRangeOpts = {}
CondGoalsRangeOpts.mapH = mapH:clone()
CondGoalsRangeOpts.mapW = mapW:clone()
CondGoalsRangeOpts.blockspct = blockspct:clone()
CondGoalsRangeOpts.waterpct = waterpct:clone()
CondGoalsRangeOpts.ngoals = torch.Tensor{2,6,3,6,1}
CondGoalsRangeOpts.ncolors = torch.Tensor{2,6,3,6,1}

local CondGoalsStaticOpts = {}
for i,j in pairs(sso) do CondGoalsStaticOpts[i] = j end


CondGoalsOpts ={}
CondGoalsOpts.RangeOpts = CondGoalsRangeOpts
CondGoalsOpts.StaticOpts = CondGoalsStaticOpts

g_opts.multigames.CondGoals = CondGoalsOpts


-------------------------------------------------------
-- Exclusion:
local ExclusionRangeOpts = {}
ExclusionRangeOpts.mapH = mapH:clone()
ExclusionRangeOpts.mapW = mapW:clone()
ExclusionRangeOpts.blockspct = blockspct:clone()
ExclusionRangeOpts.waterpct = waterpct:clone()
ExclusionRangeOpts.ngoals = torch.Tensor{2,6,3,6,1}
ExclusionRangeOpts.ngoals_active = torch.Tensor{1,3,1,3,0}

local ExclusionStaticOpts = {}
for i,j in pairs(sso) do ExclusionStaticOpts[i] = j end


ExclusionOpts ={}
ExclusionOpts.RangeOpts = ExclusionRangeOpts
ExclusionOpts.StaticOpts = ExclusionStaticOpts

g_opts.multigames.Exclusion = ExclusionOpts

-------------------------------------------------------
-- Switches:
local SwitchesRangeOpts = {}
SwitchesRangeOpts.mapH = mapH:clone()
SwitchesRangeOpts.mapW = mapW:clone()
SwitchesRangeOpts.blockspct = blockspct:clone()
SwitchesRangeOpts.waterpct = waterpct:clone()
SwitchesRangeOpts.nswitches = torch.Tensor{1,5,1,5,0}
SwitchesRangeOpts.ncolors = torch.Tensor{1,6,1,6,1}

local SwitchesStaticOpts = {}
for i,j in pairs(sso) do SwitchesStaticOpts[i] = j end


SwitchesOpts ={}
SwitchesOpts.RangeOpts = SwitchesRangeOpts
SwitchesOpts.StaticOpts = SwitchesStaticOpts

g_opts.multigames.Switches = SwitchesOpts

-------------------------------------------------------
-- LightKey:
local LightKeyRangeOpts = {}
LightKeyRangeOpts.mapH = mapH:clone()
LightKeyRangeOpts.mapW = mapW:clone()
LightKeyRangeOpts.blockspct = blockspct:clone()
LightKeyRangeOpts.waterpct = waterpct:clone()

local LightKeyStaticOpts = {}
for i,j in pairs(sso) do LightKeyStaticOpts[i] = j end

LightKeyOpts ={}
LightKeyOpts.RangeOpts = LightKeyRangeOpts
LightKeyOpts.StaticOpts = LightKeyStaticOpts

g_opts.multigames.LightKey = LightKeyOpts




-------------------------------------------------------
-- Goto:
local GotoRangeOpts = {}
GotoRangeOpts.mapH = mapH:clone()
GotoRangeOpts.mapW = mapW:clone()
GotoRangeOpts.blockspct = blockspct:clone()
GotoRangeOpts.waterpct = waterpct:clone()

local GotoStaticOpts = {}
for i,j in pairs(sso) do GotoStaticOpts[i] = j end

GotoOpts ={}
GotoOpts.RangeOpts = GotoRangeOpts
GotoOpts.StaticOpts = GotoStaticOpts


g_opts.multigames.Goto = GotoOpts

-------------------------------------------------------
-- GotoHidden:
local GotoHiddenRangeOpts = {}
GotoHiddenRangeOpts.mapH = mapH:clone()
GotoHiddenRangeOpts.mapW = mapW:clone()
GotoHiddenRangeOpts.blockspct = blockspct:clone()
GotoHiddenRangeOpts.waterpct = waterpct:clone()
GotoHiddenRangeOpts.ngoals = torch.Tensor{1,6,3,6,1}

local GotoHiddenStaticOpts = {}
for i,j in pairs(sso) do GotoHiddenStaticOpts[i] = j end

GotoHiddenOpts ={}
GotoHiddenOpts.RangeOpts = GotoHiddenRangeOpts
GotoHiddenOpts.StaticOpts = GotoHiddenStaticOpts


g_opts.multigames.GotoHidden = GotoHiddenOpts


-------------------------------------------------------
-- PushBlock:

--note:  these are not the shared range opts!!!
local PushBlockRangeOpts = {}
PushBlockRangeOpts.mapH = torch.Tensor{3,7,3,7,1}
PushBlockRangeOpts.mapW = torch.Tensor{3,7,3,7,1}
PushBlockRangeOpts.blockspct = torch.Tensor{0,0.1,0,.1,.01}
PushBlockRangeOpts.waterpct = torch.Tensor{0,0.1,0,.1,.01}

local PushBlockStaticOpts = {}
for i,j in pairs(sso) do PushBlockStaticOpts[i] = j end

PushBlockOpts ={}
PushBlockOpts.RangeOpts = PushBlockRangeOpts
PushBlockOpts.StaticOpts = PushBlockStaticOpts

g_opts.multigames.PushBlock = PushBlockOpts


-------------------------------------------------------
-- PushBlockCardinal:

--note:  these are not the shared range opts!!!
local PushBlockCardinalRangeOpts = {}
PushBlockCardinalRangeOpts.mapH = torch.Tensor{3,7,3,7,1}
PushBlockCardinalRangeOpts.mapW = torch.Tensor{3,7,3,7,1}
PushBlockCardinalRangeOpts.blockspct = torch.Tensor{0,0.1,0,.1,.01}
PushBlockCardinalRangeOpts.waterpct = torch.Tensor{0,0.1,0,.1,.01}

local PushBlockCardinalStaticOpts = {}
for i,j in pairs(sso) do PushBlockCardinalStaticOpts[i] = j end

PushBlockCardinalOpts ={}
PushBlockCardinalOpts.RangeOpts = PushBlockCardinalRangeOpts
PushBlockCardinalOpts.StaticOpts = PushBlockCardinalStaticOpts

g_opts.multigames.PushBlockCardinal = PushBlockCardinalOpts


-------------------------------------------------------
-- BlockedDoor:
local BlockedDoorRangeOpts = {}
BlockedDoorRangeOpts.mapH = mapH:clone()
BlockedDoorRangeOpts.mapW = mapW:clone()
BlockedDoorRangeOpts.blockspct = blockspct:clone()
BlockedDoorRangeOpts.waterpct = waterpct:clone()

local BlockedDoorStaticOpts = {}
for i,j in pairs(sso) do BlockedDoorStaticOpts[i] = j end

BlockedDoorOpts ={}
BlockedDoorOpts.RangeOpts = BlockedDoorRangeOpts
BlockedDoorOpts.StaticOpts = BlockedDoorStaticOpts

g_opts.multigames.BlockedDoor = BlockedDoorOpts


-------------------------------------------------------
-- MovingGoals:
sso.costs = {}
sso.costs.goal = 0
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.2
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsRangeOpts = {}
MovingGoalsRangeOpts.mapH = mapH:clone()
MovingGoalsRangeOpts.mapW = mapW:clone()
MovingGoalsRangeOpts.blockspct = blockspct:clone()
MovingGoalsRangeOpts.waterpct = waterpct:clone()

local MovingGoalsStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsStaticOpts[i] = j end

MovingGoalsOpts ={}
MovingGoalsOpts.RangeOpts = MovingGoalsRangeOpts
MovingGoalsOpts.StaticOpts = MovingGoalsStaticOpts

g_opts.multigames.MovingGoals = MovingGoalsOpts

-------------------------------------------------------
--Bottleneck:
sso.costs = {}
sso.costs.goal = 0
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.2
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local BottleneckRangeOpts = {}
BottleneckRangeOpts.mapH = mapH:clone()
BottleneckRangeOpts.mapW = mapW:clone()
BottleneckRangeOpts.blockspct = blockspct:clone()
BottleneckRangeOpts.waterpct = waterpct:clone()

local BottleneckStaticOpts = {}
for i,j in pairs(sso) do BottleneckStaticOpts[i] = j end

BottleneckOpts ={}
BottleneckOpts.RangeOpts = BottleneckRangeOpts
BottleneckOpts.StaticOpts = BottleneckStaticOpts

g_opts.multigames.Bottleneck = BottleneckOpts

-------------------------------------------------------
--Bottleneck2Rooms:
sso.costs = {}
sso.costs.goal = 0
sso.costs.empty = 0.1
sso.costs.block = 1000
sso.costs.water = 0.2
sso.costs.corner = 0
sso.costs.step = 0.1
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local Bottleneck2RoomsRangeOpts = {}
Bottleneck2RoomsRangeOpts.mapH = mapH:clone()
Bottleneck2RoomsRangeOpts.mapW = mapW:clone()
Bottleneck2RoomsRangeOpts.blockspct = blockspct:clone()
Bottleneck2RoomsRangeOpts.waterpct = waterpct:clone()

local Bottleneck2RoomsStaticOpts = {}
for i,j in pairs(sso) do Bottleneck2RoomsStaticOpts[i] = j end

Bottleneck2RoomsOpts ={}
Bottleneck2RoomsOpts.RangeOpts = Bottleneck2RoomsRangeOpts
Bottleneck2RoomsOpts.StaticOpts = Bottleneck2RoomsStaticOpts

g_opts.multigames.Bottleneck2Rooms = Bottleneck2RoomsOpts


-------------------------------------------------------
-- MovingGoalsEasy:
sso.costs = {}
sso.costs.goal = 1
sso.costs.empty = 0
sso.costs.block = 1000
sso.costs.water = -0.5
sso.costs.corner = 0
sso.costs.step = -0.5
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsEasyRangeOpts = {}
MovingGoalsEasyRangeOpts.mapH = mapH:clone()
MovingGoalsEasyRangeOpts.mapW = mapW:clone()
MovingGoalsEasyRangeOpts.blockspct = blockspct:clone()
MovingGoalsEasyRangeOpts.waterpct = waterpct:clone()

local MovingGoalsEasyStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsEasyStaticOpts[i] = j end

MovingGoalsEasyOpts ={}
MovingGoalsEasyOpts.RangeOpts = MovingGoalsEasyRangeOpts
MovingGoalsEasyOpts.StaticOpts = MovingGoalsEasyStaticOpts

g_opts.multigames.MovingGoalsEasy = MovingGoalsEasyOpts

-------------------------------------------------------
-- MovingGoalsMedium:
sso.costs = {}
sso.costs.goal = 1
sso.costs.empty = 0
sso.costs.block = 1000
sso.costs.water = -0.5
sso.costs.corner = 0
sso.costs.step = -0.5
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsMediumRangeOpts = {}
MovingGoalsMediumRangeOpts.mapH = mapH:clone()
MovingGoalsMediumRangeOpts.mapW = mapW:clone()
MovingGoalsMediumRangeOpts.blockspct = blockspct:clone()
MovingGoalsMediumRangeOpts.waterpct = waterpct:clone()

local MovingGoalsMediumStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsMediumStaticOpts[i] = j end

MovingGoalsMediumOpts ={}
MovingGoalsMediumOpts.RangeOpts = MovingGoalsMediumRangeOpts
MovingGoalsMediumOpts.StaticOpts = MovingGoalsMediumStaticOpts

g_opts.multigames.MovingGoalsMedium = MovingGoalsMediumOpts


-------------------------------------------------------
-- MovingGoalsHard:
sso.costs = {}
sso.costs.goal = 1
sso.costs.empty = 0
sso.costs.block = 1000
sso.costs.water = -0.5
sso.costs.corner = 0
sso.costs.step = -0.5
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsHardRangeOpts = {}
MovingGoalsHardRangeOpts.mapH = mapH:clone()
MovingGoalsHardRangeOpts.mapW = mapW:clone()
MovingGoalsHardRangeOpts.blockspct = blockspct:clone()
MovingGoalsHardRangeOpts.waterpct = waterpct:clone()

local MovingGoalsHardStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsHardStaticOpts[i] = j end

MovingGoalsHardOpts = {}
MovingGoalsHardOpts.RangeOpts = MovingGoalsHardRangeOpts
MovingGoalsHardOpts.StaticOpts = MovingGoalsHardStaticOpts

g_opts.multigames.MovingGoalsHard = MovingGoalsHardOpts


-------------------------------------------------------
-- MovingGoalsEasyMod:
sso.costs = {}
sso.costs.goal = 1
sso.costs.empty = 0
sso.costs.block = 1000
sso.costs.water = -0.5
sso.costs.corner = 0
sso.costs.step = -0.5
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsEasyModRangeOpts = {}
MovingGoalsEasyModRangeOpts.mapH = mapH:clone()
MovingGoalsEasyModRangeOpts.mapW = mapW:clone()
MovingGoalsEasyModRangeOpts.blockspct = blockspct:clone()
MovingGoalsEasyModRangeOpts.waterpct = waterpct:clone()

local MovingGoalsEasyModStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsEasyModStaticOpts[i] = j end

MovingGoalsEasyModOpts ={}
MovingGoalsEasyModOpts.RangeOpts = MovingGoalsEasyModRangeOpts
MovingGoalsEasyModOpts.StaticOpts = MovingGoalsEasyModStaticOpts

g_opts.multigames.MovingGoalsEasyMod = MovingGoalsEasyModOpts

-------------------------------------------------------
-- MovingGoalsMediumMod:
sso.costs = {}
sso.costs.goal = 1
sso.costs.empty = 0
sso.costs.block = 1000
sso.costs.water = -0.5
sso.costs.corner = 0
sso.costs.step = -0.5
sso.costs.pushableblock = 1000
---------------------
sso.crumb_action = 0
sso.push_action = 0
sso.flag_visited = 0
sso.enable_boundary = 0
sso.enable_corners = 0
sso.max_attributes = g_opts.max_attributes or 6


local MovingGoalsMediumModRangeOpts = {}
MovingGoalsMediumModRangeOpts.mapH = mapH:clone()
MovingGoalsMediumModRangeOpts.mapW = mapW:clone()
MovingGoalsMediumModRangeOpts.blockspct = blockspct:clone()
MovingGoalsMediumModRangeOpts.waterpct = waterpct:clone()

local MovingGoalsMediumModStaticOpts = {}
for i,j in pairs(sso) do MovingGoalsMediumModStaticOpts[i] = j end

MovingGoalsMediumModOpts ={}
MovingGoalsMediumModOpts.RangeOpts = MovingGoalsMediumModRangeOpts
MovingGoalsMediumModOpts.StaticOpts = MovingGoalsMediumModStaticOpts

g_opts.multigames.MovingGoalsMediumMod = MovingGoalsMediumModOpts


return g_opts
