clear all


f1 = csvread(strcat("ArchivedData\1_ActionAndValueLosses25MCTS2batch.csv"));
a_losses = downsample(f1(:, 1), 1);
v_losses = downsample(f1(:, 2), 1);
tot_losses = downsample(f1(:, 3), 1);

batches = [1:numel(tot_losses)];

figure
plot(batches, a_losses)
hold on
plot(batches, v_losses)
plot(batches, tot_losses)
legend("action losses", "value losses", "total losses")
hold off

% greedy policy..............................................

f2 = csvread("ArchivedData\1_Challenger Losses25MCTS2batch.csv");
figure
f2(f2==0) = nan;
plot(f2')

f3 = csvread(strcat("ArchivedData\1_Current Losses25MCTS2batch.csv"));
figure
f3(f3==0) = nan;
plot(f3')

clear all


for iter = 1:2
    
    f1 = csvread(strcat("Data\ActionAndValueLosses", int2str(iter), ".csv"));
    a_losses = downsample(f1(:, 1), 1);
    v_losses = downsample(f1(:, 2), 1);
    tot_losses = downsample(f1(:, 3), 1);
    
    batches = [1:numel(tot_losses)];
    
    figure
    plot(batches, a_losses)
    hold on
    plot(batches, v_losses)
    plot(batches, tot_losses)
    legend("action losses", "value losses", "total losses")
    hold off
    
    % greedy policy..............................................
    
    f2 = csvread(strcat("Data\TrainingExamples", int2str(iter-1), ".csv"));
    figure
    f2(f2==0) = nan;
    plot(f2')
    
    f3 = csvread(strcat("Data\Challenger Losses", int2str(iter), ".csv"));
    figure
    f3(f3==0) = nan;
    plot(f3')
end
