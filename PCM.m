% LAB1--PCM---------
clc;
clear;
close all;

fm = 5;
fs = 90;

% Sine signal
t = 0:0.001:1;
msg = sin(2*pi*fm*t);

% Sampling
n = 0:1/fs:1;
sample = sin(2*pi*fm*n);

% PCM
q = round((sample+1)*7/2);
pcm = dec2bin(q,3);

% Demodulation
d = bin2dec(pcm);
d = 2*d/7 - 1;
demod = interp1(n,d,t,'spline');

% Plot
subplot(4,1,1);
plot(t,msg);
title('Original Sine Signal');

subplot(4,1,2);
stem(n,sample);
title('Sampled Signal');

subplot(4,1,3);
stairs(n,q);
title('Quantization');

subplot(4,1,4);
plot(t,demod);
title('Reconstructed Signal');