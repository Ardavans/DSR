function [] = subgoal_discovery(gamma, n_samples, sNcut, sArea, n_runs, exp_name)
addpath('./NcutImageSegment');
%% global variables 
%gamma = 0.5
%n_samples = 2000
%sNcut = 0.14; sArea = 500;
%n_runs = 10;

total_samples = 15000;

%% loading data

filename = '../../Doom_data.h5';
datasetname_m = '/m_full_tensor';
datasetname_s = '/s_full_tensor'; 

data_m = h5read(filename,datasetname_m)'; % SR data
data_s = h5read(filename,datasetname_s)'; % raw state data

cut_indices_cntvec = zeros(total_samples,1);
for seed = 1:n_runs
    rng(seed);
    rando = randperm(total_samples);
    rng(seed);
    ind_map_shuffled = randperm(total_samples);
    
    data_m_processed = data_m(rando(1:n_samples),:);
    
    W = exp(-gamma*squareform(pdist(data_m_processed)));
    W = double(W);

    Seg = rando(1:n_samples)'; % the first segment has whole nodes. [1 2 3 ... N]'
    [Seg Cut Id Ncut] = NcutPartition(Seg, W, sNcut, sArea, [], 'ROOT');
    cut_indices = unique(vertcat(Cut{:}));
    size(cut_indices,1)
    for i=1:size(cut_indices,1)
        cut_indices(i) = find(ind_map_shuffled == cut_indices(i));
    end
    
    %cut_indices = ind_map_shuffled(cut_indices);
    cut_indices_cntvec([cut_indices]) = cut_indices_cntvec([cut_indices]) + 1;
end
exp_name = 1;
[vals, subgoals] = sort(cut_indices_cntvec, 'descend');
save(['results/' num2str(exp_name) '.mat'], 'vals', 'subgoals');
end
