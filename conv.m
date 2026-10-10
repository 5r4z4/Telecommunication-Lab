clc;
clear;
close all;

% Input data
data = [1 0 1 1 0 1];

% Convolutional encoder
trellis = poly2trellis(3, [7 5]);

% Encoding
encoded = convenc(data, trellis);

% Decoding
decoded = vitdec(encoded, trellis, 5, 'trunc', 'hard');

% Display results
disp('Original Data:');
disp(data);

disp('Encoded Data:');
disp(encoded);

disp('Decoded Data:');
disp(decoded);

% Plot
figure;

subplot(3,1,1);
stairs(data);
title('Original Data');
ylim([-0.5 1.5]);
grid on;

subplot(3,1,2);
stairs(encoded);
title('Encoded Data');
ylim([-0.5 1.5]);
grid on;

subplot(3,1,3);
stairs(decoded);
title('Decoded Data');
ylim([-0.5 1.5]);
grid on;