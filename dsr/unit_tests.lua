-- unit test 

-- require 'cutorch'
-- require 'nn'

require 'convnet'

args = {}
args.hist_len = 1
args.ncols = 1
args.gpu = -1
args.srdims = 256
args.n_actions = 3
args.verbose = 0
args.state_dim = 84*84
args.input_dims = {args.hist_len, 84, 84}
args.n_units        = {32, 64, 64}
args.filter_size    = {8, 4, 3}
args.filter_stride  = {4, 2, 1}
args.n_hid          = {512}
args.nl             = nn.Rectifier

net = create_network(args)

s = torch.zeros(10,1,84,84)

print(net:forward(s))


-- grads = {{torch.zeros(10,args.srdims),torch.zeros(10,args.srdims),torch.zeros(10,args.srdims)}, 
-- 						{torch.zeros(10,1), torch.zeros(10,7056)}}
-- print(net:backward(s, grads):size())