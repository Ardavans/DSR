%% loading data

filename_m = '/home/cocosci/simanta/DeepSR_synth_exp/sample_data/Doom_data.h5';
filename_s = '/home/cocosci/simanta/DeepSR_synth_exp/sample_data/Doom_data.h5';

datasetname_m = '/m_full_tensor';
datasetname_s = '/s_full_tensor';

data_m = h5read(filename_m,datasetname_m)'; % SR data
data_s = h5read(filename_s,datasetname_s)'; % raw state data

files=dir('/home/cocosci/simanta/DeepSR_subgoal_cuts/results/*.mat')
count2 = 0;
for file = files'
    count2 = count2 + 1
    matfile=['/home/cocosci/simanta/DeepSR_subgoal_cuts/results/' file.name];
    matfile
    top_k = 4;
    load(matfile)
    size(data_s)

    count = 1;
    graphed = 1;
    sample = data_s(subgoals(count),:);
    found_states = [sample];
    sum_arr = permute(reshape(sample, 84, 84, 3), [2,1,3]);
    subplot(4,2,graphed);
    subimage(sum_arr),title(strcat('subgoal-',int2str(count)));
    imwrite(sum_arr, strcat('/home/cocosci/simanta/DeepSR_subgoal_cuts/doom_',int2str(count2), '_', int2str(count), '.png'));
    graphed = graphed + 1;
    while graphed <= top_k
        sample = data_s(subgoals(count),:);
        size(found_states)
        if ismember(sample,found_states, 'rows')
            count = count + 1;
            continue
        end
        found_states = [found_states;sample];
        sum_arr = permute(reshape(sample, 84, 84, 3), [2,1,3]);
        subplot(4,2,graphed);
        subimage(sum_arr),title(strcat('subgoal-',int2str(count)));
        imwrite(sum_arr, strcat('/home/cocosci/simanta/DeepSR_subgoal_cuts/doom_',int2str(count2), '_', int2str(count), '.png'));
        graphed = graphed + 1;
    end
    k = waitforbuttonpress 
end