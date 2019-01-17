clear all
for iter = 1:2
    iter
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
    
    f2 = csvread(strcat("Data\Current Losses", int2str(iter), ".csv"));
    figure
    f2(f2==0) = nan;
    plot(f2')
    
    f3 = csvread(strcat("Data\Challenger Losses", int2str(iter), ".csv"));
    figure
    f3(f3==0) = nan;
    plot(f3')
end
