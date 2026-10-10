clc;
clear;
close all;

% Data
data = [1 0 1 1 0 1];

% Frequencies
freq = [10 20 30 40];

% Time
t = 0:0.001:0.1;

% Frequency hopping sequence
hop = [1 3 2 4 1 3];

signal = [];

% FHSS signal generation
for i = 1:length(data)

    f = freq(hop(i));

    if data(i) == 1
        s = sin(2*pi*f*t);
    else
        s = -sin(2*pi*f*t);
    end

    signal = [signal s];
end

% Plot
figure;

plot(signal);
title('Frequency Hopping Spread Spectrum (FHSS)');
xlabel('Time');
ylabel('Amplitude');
grid on