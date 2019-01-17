clear all

f = csvread("ActionAndValueLosses1.csv");
a_losses = f(:, 1);
v_losses = f(:, 2);
tot_losses = f(:, 3);

batches = [1:numel(tot_losses)];

figure
plot(batches, a_losses)
hold on
plot(batches, v_losses)
plot(batches, tot_losses)
legend("a losses", "v losses", "tot losses")
hold off

clear all
% greedy policy..............................................

f = csvread("Current Losses1.csv");
figure
f(f==0) = nan;
plot(f')
