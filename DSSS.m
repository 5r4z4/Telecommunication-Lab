clc; 
clear; 
close all;

% 1. Input Data & PN Code
data = [1 -1 1];              % Original Data
pn = [1 -1 1 1 -1 1 -1];      % 7-chip PN sequence

% 2. DSSS Spreading
dsss = [];
for i = 1:length(data)
    dsss = [dsss, data(i) * pn];
end

% 3. Plotting
figure;

subplot(3, 1, 1);
stairs(repelem(data, length(pn)));
title('1. Original Input Data');
xlabel('Chip Index'); ylabel('Amplitude');
ylim([-1.5 1.5]); grid on;

subplot(3, 1, 2);
stairs(repmat(pn, 1, length(data)));
title('2. PN Sequence / Chipping Code');
xlabel('Chip Index'); ylabel('Amplitude');
ylim([-1.5 1.5]); grid on;

subplot(3, 1, 3);
stairs(dsss);
title('3. DSSS Spread Signal');
xlabel('Chip Index'); 
ylabel('Amplitude');
ylim([-1.5 1.5]); grid on;