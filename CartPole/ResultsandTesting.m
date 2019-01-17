clear all

f = csvread("Data\ActionAndValueLosses1.csv");
a_losses = downsample(f(:, 1), 1);
v_losses = downsample(f(:, 2), 1);
tot_losses = downsample(f(:, 3), 1);

batches = [1:numel(tot_losses)];

figure
plot(batches, a_losses)
hold on
plot(batches, v_losses)
plot(batches, tot_losses)
legend("action losses", "value losses", "total losses")
hold off

clear all
% greedy policy..............................................

f = csvread("Current Losses1.csv");
figure
f(f==0) = nan;
plot(f')
