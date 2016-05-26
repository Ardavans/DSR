require "initenv"

function create_network(args)

    local net = nn.Sequential()
    net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer
    local convLayer = nn.SpatialConvolution

    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end

    local nel
    if args.gpu >= 0 then
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end


    local sr_splitter = nn.Sequential()
        local sr_output_ftrs_dimensions = args.srdims-- 10
        sr_splitter:add(nn.Linear(last_layer_size, sr_output_ftrs_dimensions))
        sr_splitter:add(nn.Tanh())--args.nl())
    
        sr_splitter:add(nn.Replicate(2)) -- one goes to reward and other to SR
        sr_splitter:add(nn.SplitTable(1))
        sr_fork = nn.ParallelTable();
            mnet = nn.Sequential()
                mnet:add(nn.GradScale(0)) --don't backprop through SR representation down to features. only reward should do this!
                -- mnet:add(nn.Dropout(0.2))
                mnet:add(nn.Replicate(args.n_actions))
                mnet:add(nn.SplitTable(1))
                mnet_subnets = nn.ParallelTable()
                    for i=1,args.n_actions do
                        mnet_fork = nn.Sequential()
                        mnet_fork:add(nn.Linear(sr_output_ftrs_dimensions, 512))
                        mnet_fork:add(args.nl())
                        mnet_fork:add(nn.Linear(512, 256))
                        mnet_fork:add(args.nl())
                        mnet_fork:add(nn.Linear(256, sr_output_ftrs_dimensions))

                        mnet_subnets:add(mnet_fork)
                    end
            mnet:add(mnet_subnets)
            sr_fork:add(mnet) -- SR prediction

            rnet = nn.Sequential()
            -- rnet:add(nn.Identity())
                rnet:add(nn.Replicate(2))
                rnet:add(nn.SplitTable(1))
                rnet_fork = nn.ParallelTable()
                    -- extrinsic net
                    extrinsic_net = nn.Sequential()
                    extrinsic_net:add(nn.GradScale(1))
                    extrinsic_net:add(nn.Linear(sr_output_ftrs_dimensions, 1))
                    extrinsic_net:add(nn.Identity())
                    rnet_fork:add(extrinsic_net)
                    -- decoder
                    decoder = nn.Sequential()
                    decoder:add(nn.Reshape(sr_output_ftrs_dimensions, 1,1))
                    decoder:add(nn.SpatialFullConvolution(sr_output_ftrs_dimensions, 8*64, 4, 4))
                    decoder:add(nn.ReLU(true))
                    decoder:add(nn.SpatialFullConvolution(8*64, 4*64, 4, 4, 2, 2))
                    decoder:add(nn.ReLU(true))
                    decoder:add(nn.SpatialFullConvolution(4*64, 2*64, 4, 4, 2, 2,1,1))
                    decoder:add(nn.ReLU(true))
                    decoder:add(nn.SpatialFullConvolution(2*64, 64, 4, 4, 2, 2))
                    decoder:add(nn.ReLU(true))
                    decoder:add(nn.SpatialFullConvolution(64, args.ncols*args.hist_len, 4, 4, 2, 2,1,1))
                    -- decoder:add(nn.Tanh())
                    decoder:add(nn.Reshape(args.state_dim*args.hist_len))
                    decoder:add(nn.GradScale(1))

                    rnet_fork:add(decoder)
            rnet:add(rnet_fork)
            sr_fork:add(rnet) -- reward prediction
        sr_splitter:add(sr_fork) 
    net:add(sr_splitter)

    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end
    return net
end
