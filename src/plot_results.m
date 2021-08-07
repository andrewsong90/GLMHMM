clc
close all
clear

%% Reading the files
filename = './../results/state4/result.mat';
info = load(filename);

numOfanimals_train = size(info.animal_list,1);

display("=============================")
display("Animals used for training ...")
for idx = 1:numOfanimals_train
    
    name = convertCharsToStrings(info.animal_list(idx,:,:));
    fieldname = strcat('prob_emission_',int2str(idx-1),'_train');
    display(name)
    
    figure;
    tiledlayout(3,1)
    ax1 = nexttile;
    % Predicted observation
    numOfbehaviors = size(info.(fieldname),1);
    hold on 
    
    for behav_idx = 1:numOfbehaviors
        plot((info.train_stamp(idx,1)+1:info.train_stamp(idx,2))/info.fs, info.(fieldname)(behav_idx,:))
    end
    legend('0: Idle','1: Adult_enter', '2: Adult_exit', '3: Approach_pup', '4: Investigate_pup', '5: Retrieve_pup', '6: Sitting_with_pup', 'Interpreter', 'none')
    ylabel('P(behavior)')
    
    title(name, 'Interpreter', 'none')
    
    % Most likely trajectory
    ax2 = nexttile;
    hold on;
    traj = info.trajectory_train{1,idx};
    plot((info.train_stamp(idx,1)+1:info.train_stamp(idx,2))/info.fs,traj)
    ylabel('States')

    % True behavior
    ax3 = nexttile;
    behavior = info.behavior{1,idx};
    plot((info.train_stamp(idx,1)+1:info.train_stamp(idx,2))/info.fs, behavior)
    ylabel('True behavior')
    xlabel('Time (s)')

    linkaxes([ax1 ax2 ax3],'x')
end

display("=============================")
display("Animals used for testing ...")
for idx = 1:numOfanimals_train
    
    name = convertCharsToStrings(info.animal_list_test(idx,:,:));
    fieldname = strcat('prob_emission_',int2str(idx-1),'_test');
    display(name)
    
    figure;
    tiledlayout(3,1)
    ax1 = nexttile;
    % Predicted observation
    numOfbehaviors = size(info.(fieldname),1);
    hold on 
    
    for behav_idx = 1:numOfbehaviors
        plot((info.test_stamp(idx,1)+1:info.test_stamp(idx,2))/info.fs, info.(fieldname)(behav_idx,:))
    end
    legend('0: Idle','1: Adult_enter', '2: Adult_exit', '3: Approach_pup', '4: Investigate_pup', '5: Retrieve_pup', '6: Sitting_with_pup', 'Interpreter', 'none')
    ylabel('P(behavior)')
    
    title(name, 'Interpreter', 'none')
    
    % Most likely trajectory
    ax2 = nexttile;
    hold on;
    traj = info.trajectory_test{1,idx};
    plot((info.test_stamp(idx,1)+1:info.test_stamp(idx,2))/info.fs,traj)
    ylabel('States')

    % True behavior
    ax3 = nexttile;
    behavior = info.behavior_test{1,idx};
    plot((info.test_stamp(idx,1)+1:info.test_stamp(idx,2))/info.fs, behavior)
    ylabel('True behavior')
    xlabel('Time (s)')

    linkaxes([ax1 ax2 ax3],'x')
end