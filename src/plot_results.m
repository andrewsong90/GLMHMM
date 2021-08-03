clc
close all
clear

%% Reading the files


%Define filenames for template videos
% filename = './../results/2021_08_02_23_51_08_state2_lambda0.05_iter200/result.mat';
% info = load(filename);

filename = './../results/2021_08_03_10_15_27_state1_lambda0.05_iter2/result.mat';
info = load(filename);

numOfanimals_train = size(info.animal_list,1);

for idx = 1:numOfanimals_train
    
    name = convertCharsToStrings(info.animal_list(idx,:,:));
    fieldname = strcat('prob_emission_',int2str(idx-1),'_train');
    
    figure;
    tiledlayout(3,1)
    ax1 = nexttile;
    % Predicted observation
    numOfbehaviors = size(info.(fieldname),1);
    hold on 
    
    for behav_idx = 1:numOfbehaviors
        plot(info.(fieldname)(behav_idx,:))
    end
    
    title(name, 'Interpreter', 'none')
    
    % Most likely trajectory
    ax2 = nexttile;
    hold on;
    traj = info.trajectory_train{1,1};
    plot(traj)
    ylabel('States')

    % True behavior
    ax3 = nexttile;
    behavior = info.behavior{1,1};
    plot(behavior)
    ylabel('True behavior')
    xlabel('Time')

    linkaxes([ax1 ax2 ax3],'x')
end