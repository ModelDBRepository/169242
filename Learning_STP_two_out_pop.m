
%% Recurrent network of N neurons divided into 2 oppulations learns to optimize synapses by learning STP and STDP through an error-driven mechanism

% This code calls the functions:
    % 1. Euler_integration_conductance_based_IF_multi_synapses - neuron integration
    % 2. sym_measure - symmetry measure evaluation
    % 3. confplot - shaded plot
    % 4. writePDF1000ppi - print 1000ppi .pdf
    % 5. TM_single_synapse - compute synaptic traces from TM model (and calls the function Euler_integration_IF_with_external_current)

clear all
close all

%% General parameters

N_inputs1 = 30;
N_inputs2 = 30;
N_outputs1 = 10;
N_outputs2 = 10;

N_inputs = N_inputs1 + N_inputs2;
N_outputs = N_outputs1 + N_outputs2;

N = N_inputs + N_outputs;    %number of neurons

high = 30;  %Hz
low = 5;    %Hz

temporal = 1;   %switching regime variable

%% Computational parameters
dt_simul = 1e-3;    %s

time_simul = 5e1;   %s
n_sample = (round(time_simul / dt_simul) ) + 1;

timesteps_for_firing_rate = 500;
exp_decay_mean_firing_rate = 0.001;


%% Plotting parameters
numericFontSize = 25;
axesFontSize = 30;
lineThickness = 2;
markLine = 1;
markSize = 12;


%% Neuron Parameters - from Carvalho Buonomano
E = 2.9e-2;         %V
g_L = 0.1e-6;       %S
tau_g = 1e-2;       %s
V_th = 1e-3;        %V

refr_period = 10e-3;    %s ---> maximum frequency: 100Hz
freq_max = 1 / refr_period;


%% STP Parameters
tau_f_lower = 1e-3;
tau_f_upper = 900e-3;
tau_f_init = tau_f_lower + (tau_f_upper-tau_f_lower) * rand(N, N);
tau_f_init = tau_f_init - diag(diag(tau_f_init));

tau_d_lower = 100e-3;
tau_d_upper = 900e-3;
tau_d_init = tau_d_lower + (tau_d_upper-tau_d_lower) * rand(N, N);
tau_d_init = tau_d_init - diag(diag(tau_d_init));

U_init_lower = 0.05;
U_init_upper = 0.95;
U_init = U_init_lower + (U_init_upper-U_init_lower) * rand(N, N);
U_init = U_init - diag(diag(U_init));
U_lower = 0.05;
U_upper = 0.95;


%% STDP Learning parameters
% Triplet model - From Pfister & Gerstner: Triplets in STDP models. Nearest-spike
taupos  = 16.8 * 1E-3;      %time constant for r1 in s
tauneg  = 33.7 * 1E-3;      %time constant for o1 in s
taux    = 575 * 1E-3;       %time constant for r2 in s
tauy    = 47 * 1E-3;        %time constant for o2 in s
A2pos   = 4.6E-3;           %amplitude of weight change for LTP (pair interaction)
A3pos   = 9.1E-3;           %amplitude of weight change for LTP (triplet interaction)
A2neg   = 3.0E-3;           %amplitude of weight change for LTD (pair interaction)
A3neg   = 7.5E-9;           %amplitude of weight change for LTD (triplet interaction)

learning_rate = 2;

w_max = 1;      %maximum synaptic weight
w_max_between = 0.1;   %maximum synaptic weight between different populations
w_min = 0.001;  %minimum synaptic weight


%% Input neurons selection
neurons_label = 1 : N;
input_neurons_logical = zeros(N, 1);

temp = randperm(N); %permuting indices

input_neurons_label = sort(temp(1 : N_inputs1+N_inputs2));
input_neurons_logical(input_neurons_label) = 1; %identifying neurons in the input pull

input1_neurons_label = input_neurons_label(1:N_inputs1);
input1_neurons_logical(input1_neurons_label) = 1; %identifying neurons in the input pull
input2_neurons_label = input_neurons_label(N_inputs1+1 : N_inputs1+N_inputs2);
input2_neurons_logical(input2_neurons_label) = 1; %identifying neurons in the input pull


%% Output neurons selection
output1_neurons_logical = zeros(N, 1);
output2_neurons_logical = zeros(N, 1);

temp2 = neurons_label;
temp2(input_neurons_label) = [];   %excluding neurons that are in the input pool only

perm_index = randperm(size(temp2, 2));  %permuting indices
temp = temp2(perm_index);

output1_neurons_label = sort(temp(1 : N_outputs1));   %identifying neurons in the output pool
output1_neurons_logical(output1_neurons_label) = 1;

output2_neurons_label = sort(temp(N_outputs1 + 1 : N_outputs1 + N_outputs2));   %identifying neurons in the output pool
output2_neurons_logical(output2_neurons_label) = 1;

output_neurons_label = [output1_neurons_label, output2_neurons_label];

target_firing_rate = zeros(N, 1);
target_firing_rate(output1_neurons_label) = high;
target_firing_rate(input1_neurons_label) = high;
target_firing_rate(output2_neurons_label) = low;
target_firing_rate(input2_neurons_label) = low;


%% Input signal

%---Temporal code. Neurons receive external input and fire in a precise sequence with the same rate
rho_input = 10;                                 % Stimulus rate in Hz
delay = 1 / (rho_input * N_inputs);             % Time delay between the stimulus injection for two consecutive neurons in s 
input_time_1 =  100 * 1E-3;                     % Stimulus injection time for neuron 1. Dynamics begins after 100ms to be sure that adding gaussian noise does not give negative time
jitter_amplitude = 0.1 * delay;                 % 10% of the delay
next_input_times = input_time_1 + (0:(N_inputs-1))' * delay + ( jitter_amplitude * randn(N_inputs, 1) ); % Gaussian noise with sd = (10% of the delay) --> The inversion in the order of input injection between two consecutive neurons is a very unlikely event

dV_input = 2 * V_th;


%% Variables
t = 0 : dt_simul : time_simul;
t = round(1000.*t)./1000; %this approximates the time steps to the order of ms (the sampling time is 1 ms)

V = zeros(N, n_sample);
time_from_last_spike = zeros(N, 1);
neurons_spike_logical = zeros(N, timesteps_for_firing_rate);

firing_rates = zeros(N, n_sample);
mean_firing_rate = zeros(N, n_sample);
error = zeros(N, floor(n_sample));

g = zeros(N, N);
F = zeros(N, N);
D = zeros(N, N); 
U = zeros(N, N);

o1 = zeros(N,1);    % Post synaptic variables. Note: its value depends only on the activity of the post (the pre triggers only, without changing) ---> We need the same number as neurons, not synapses
r1 = zeros(N,1);    % Pre synaptic variables. Note: the same argument above holds here.
o2 = zeros(N,1);    % Post synaptic variables. 
r2 = zeros(N,1);    % Pre synaptic variables. 

w_evolution_out1_1 = zeros(N_inputs, n_sample);
w_evolution_out1_2 = zeros(N_inputs, n_sample);
w_evolution_out2_1 = zeros(N_inputs, n_sample);
w_evolution_out2_2 = zeros(N_inputs, n_sample);

w_evolution_out11 = zeros(N_outputs1, n_sample);
w_evolution_out22 = zeros(N_outputs2, n_sample);
w_evolution_in11 = zeros(N_inputs1, n_sample);
w_evolution_in22 = zeros(N_inputs2, n_sample);
w_evolution_out12 = zeros(N_outputs1, n_sample);
w_evolution_out21 = zeros(N_outputs2, n_sample);
w_evolution_in12 = zeros(N_inputs1, n_sample);
w_evolution_in21 = zeros(N_inputs2, n_sample);
w_evolution_out1in1 = zeros(N_outputs1, n_sample);
w_evolution_in1out1 = zeros(N_inputs1, n_sample);
w_evolution_out2in2 = zeros(N_outputs2, n_sample);
w_evolution_in2out2 = zeros(N_inputs2, n_sample);

w_evolution_out11_mean = zeros(1, n_sample);
w_evolution_out22_mean = zeros(1, n_sample);
w_evolution_in11_mean = zeros(1, n_sample);
w_evolution_in22_mean = zeros(1, n_sample);
w_evolution_out12_mean = zeros(1, n_sample);
w_evolution_out21_mean = zeros(1, n_sample);
w_evolution_in12_mean = zeros(1, n_sample);
w_evolution_in21_mean = zeros(1, n_sample);
w_evolution_out1in1_mean = zeros(1, n_sample);
w_evolution_in1out1_mean = zeros(1, n_sample);
w_evolution_out2in2_mean = zeros(1, n_sample);
w_evolution_in2out2_mean = zeros(1, n_sample);

s_evolution = zeros(1, n_sample);
s_out_evolution = zeros(1, n_sample);
s_in_evolution = zeros(1, n_sample);
s_out1_evolution = zeros(1, n_sample);
s_out2_evolution = zeros(1, n_sample);
s_in1_evolution = zeros(1, n_sample);
s_in2_evolution = zeros(1, n_sample);
s_out12_evolution = zeros(1, n_sample);
s_in12_evolution = zeros(1, n_sample);
s_out1_in1_evolution = zeros(1, n_sample);
s_out2_in2_evolution = zeros(1, n_sample);

tau_f_evolution_out1 = zeros(N_outputs1, floor(n_sample));
tau_d_evolution_out1 = zeros(N_outputs1, floor(n_sample));
U_evolution_out1 = zeros(N_outputs1, floor(n_sample));
tau_f_evolution_out1_mean = zeros( 1, floor(n_sample));
tau_d_evolution_out1_mean = zeros( 1, floor(n_sample));
U_evolution_out1_mean = zeros( 1, floor(n_sample));
tau_f_evolution_out1_sd = zeros( 1, floor(n_sample));
tau_d_evolution_out1_sd = zeros( 1, floor(n_sample));
U_evolution_out1_sd = zeros( 1, floor(n_sample));

tau_f_evolution_out2 = zeros(N_outputs2, floor(n_sample));
tau_d_evolution_out2 = zeros(N_outputs2, floor(n_sample));
U_evolution_out2 = zeros(N_outputs2, floor(n_sample));
tau_f_evolution_out2_mean = zeros( 1, floor(n_sample));
tau_d_evolution_out2_mean = zeros( 1, floor(n_sample));
U_evolution_out2_mean = zeros( 1, floor(n_sample));
tau_f_evolution_out2_sd = zeros( 1, floor(n_sample));
tau_d_evolution_out2_sd = zeros( 1, floor(n_sample));
U_evolution_out2_sd = zeros( 1, floor(n_sample));

% tau_f_out11_evolution = zeros( N_outputs1, floor(n_sample));
% tau_d_out11_evolution = zeros( N_outputs1, floor(n_sample));
% U_out11_evolution = zeros( N_outputs1, floor(n_sample));
% tau_f_out11_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out11_evolution_mean = zeros( 1, floor(n_sample));
% U_out11_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_out12_evolution = zeros( N_outputs2, floor(n_sample));
% tau_d_out12_evolution = zeros( N_outputs2, floor(n_sample));
% U_out12_evolution = zeros( N_outputs2, floor(n_sample));
% tau_f_out12_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out12_evolution_mean = zeros( 1, floor(n_sample));
% U_out12_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_out22_evolution = zeros( N_outputs2, floor(n_sample));
% tau_d_out22_evolution = zeros( N_outputs2, floor(n_sample));
% U_out22_evolution = zeros( N_outputs2, floor(n_sample));
% tau_f_out22_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out22_evolution_mean = zeros( 1, floor(n_sample));
% U_out22_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_out21_evolution = zeros( N_outputs1, floor(n_sample));
% tau_d_out21_evolution = zeros( N_outputs1, floor(n_sample));
% U_out21_evolution = zeros( N_outputs1, floor(n_sample));
% tau_f_out21_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out21_evolution_mean = zeros( 1, floor(n_sample));
% U_out21_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in11_evolution = zeros( N_inputs1, floor(n_sample));
% tau_d_in11_evolution = zeros( N_inputs1, floor(n_sample));
% U_in11_evolution = zeros( N_inputs1, floor(n_sample));
% tau_f_in11_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in11_evolution_mean = zeros( 1, floor(n_sample));
% U_in11_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in12_evolution = zeros( N_inputs2, floor(n_sample));
% tau_d_in12_evolution = zeros( N_inputs2, floor(n_sample));
% U_in12_evolution = zeros( N_inputs2, floor(n_sample));
% tau_f_in12_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in12_evolution_mean = zeros( 1, floor(n_sample));
% U_in12_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in22_evolution = zeros( N_inputs2, floor(n_sample));
% tau_d_in22_evolution = zeros( N_inputs2, floor(n_sample));
% U_in22_evolution = zeros( N_inputs2, floor(n_sample));
% tau_f_in22_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in22_evolution_mean = zeros( 1, floor(n_sample));
% U_in22_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in21_evolution = zeros( N_inputs1, floor(n_sample));
% tau_d_in21_evolution = zeros( N_inputs1, floor(n_sample));
% U_in21_evolution = zeros( N_inputs1, floor(n_sample));
% tau_f_in21_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in21_evolution_mean = zeros( 1, floor(n_sample));
% U_in21_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_out1_in1_evolution = zeros( N_inputs1, floor(n_sample));
% tau_d_out1_in1_evolution = zeros( N_inputs1, floor(n_sample));
% U_out1_in1_evolution = zeros( N_inputs1, floor(n_sample));
% tau_f_out1_in1_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out1_in1_evolution_mean = zeros( 1, floor(n_sample));
% U_out1_in1_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in1_out1_evolution = zeros( N_outputs1, floor(n_sample));
% tau_d_in1_out1_evolution = zeros( N_outputs1, floor(n_sample));
% U_in1_out1_evolution = zeros( N_outputs1, floor(n_sample));
% tau_f_in1_out1_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in1_out1_evolution_mean = zeros( 1, floor(n_sample));
% U_in1_out1_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_out2_in2_evolution = zeros( N_inputs2, floor(n_sample));
% tau_d_out2_in2_evolution = zeros( N_inputs2, floor(n_sample));
% U_out2_in2_evolution = zeros( N_inputs2, floor(n_sample));
% tau_f_out2_in2_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_out2_in2_evolution_mean = zeros( 1, floor(n_sample));
% U_out2_in2_evolution_mean = zeros( 1, floor(n_sample));
% 
% tau_f_in2_out2_evolution = zeros( N_outputs2, floor(n_sample));
% tau_d_in2_out2_evolution = zeros( N_outputs2, floor(n_sample));
% U_in2_out2_evolution = zeros( N_outputs2, floor(n_sample));
% tau_f_in2_out2_evolution_mean = zeros( 1, floor(n_sample));
% tau_d_in2_out2_evolution_mean = zeros( 1, floor(n_sample));
% U_in2_out2_evolution_mean = zeros( 1, floor(n_sample));


%% Initial state
V(:, 1) = 0;

time_from_last_spike(:) = refr_period * 1e+3;  %renormalisation in ms units
neurons_previously_fired_logical = (V(:, 1) >= V_th);

F(:, :) = U_init;
F = F - diag(diag(F));

D(:, :) = 1;
D = D - diag(diag(D));

U(:, :) = U_init;
U = U - diag(diag(U));

tau_f(:, :) = tau_f_init;
tau_f = tau_f - diag(diag(tau_f));

tau_d(:, :) = tau_d_init;
tau_d = tau_d - diag(diag(tau_d));

%Connectivity matrix
w_in = (w_max - w_min) * rand(N, N);   %initialization of the weights
w_in = w_in - diag(diag(w_in));        %eliminate self-interaction

w_in(output2_neurons_label, input1_neurons_label) = 0;
w_in(input1_neurons_label, output2_neurons_label) = 0;
w_in(output1_neurons_label, input2_neurons_label) = 0;
w_in(input2_neurons_label, output1_neurons_label) = 0;

w_in(input1_neurons_label, input2_neurons_label) = w_in(input1_neurons_label, input2_neurons_label) * w_max_between;
w_in(input2_neurons_label, input1_neurons_label) = w_in(input2_neurons_label, input1_neurons_label) * w_max_between;
w_in(output1_neurons_label, output2_neurons_label) = w_in(output1_neurons_label, output2_neurons_label) * w_max_between;
w_in(output2_neurons_label, output1_neurons_label) = w_in(output2_neurons_label, output1_neurons_label) * w_max_between;
%w_in(input1_neurons_label, input2_neurons_label) = w_in(input1_neurons_label, input2_neurons_label) * 0.;
%w_in(input2_neurons_label, input1_neurons_label) = w_in(input2_neurons_label, input1_neurons_label) * 0.;
%w_in(output1_neurons_label, output2_neurons_label) = w_in(output1_neurons_label, output2_neurons_label) * 0.;
%w_in(output2_neurons_label, output1_neurons_label) = w_in(output2_neurons_label, output1_neurons_label) * 0.;

w = w_in;

%for the STP rule
% w_norm = w;
% w_norm(input1_neurons_label, input2_neurons_label) = w(input1_neurons_label, input2_neurons_label) / w_max_between;
% w_norm(input2_neurons_label, input1_neurons_label) = w(input2_neurons_label, input1_neurons_label) / w_max_between;
% w_norm(output1_neurons_label, output2_neurons_label) = w(output1_neurons_label, output2_neurons_label) / w_max_between;
% w_norm(output2_neurons_label, output1_neurons_label) = w(output2_neurons_label, output1_neurons_label) / w_max_between;

%store weigths information
w_evolution_out1_1(:, 1) = w_in(output1_neurons_label(1), input_neurons_label)';
w_evolution_out1_2(:, 1) = w_in(output1_neurons_label(2), input_neurons_label)';
w_evolution_out2_1(:, 1) = w_in(output2_neurons_label(1), input_neurons_label)';
w_evolution_out2_2(:, 1) = w_in(output2_neurons_label(2), input_neurons_label)';

%average wieght onto each neuron and onto a group
temp = w_in(output1_neurons_label, output1_neurons_label)';
temp(logical(eye(size(temp)))) = [];
w_evolution_out11(:, 1) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
w_evolution_out11_mean(1) = mean(w_evolution_out11(:, 1));

temp = w_in(output2_neurons_label, output2_neurons_label)';
temp(logical(eye(size(temp)))) = [];
w_evolution_out22(:, 1) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
w_evolution_out22_mean(1) = mean(w_evolution_out22(:, 1));

temp = w_in(input1_neurons_label, input1_neurons_label)';
temp(logical(eye(size(temp)))) = [];
w_evolution_in11(:, 1) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
w_evolution_in11_mean(1) = mean(w_evolution_in11(:, 1));

temp = w_in(input2_neurons_label, input2_neurons_label)';
temp(logical(eye(size(temp)))) = [];
w_evolution_in22(:, 1) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
w_evolution_in22_mean(1) = mean(w_evolution_in22(:, 1));

w_evolution_out12(:, 1) = mean(w_in(output1_neurons_label, output2_neurons_label), 2);
w_evolution_out12_mean(1) = mean(w_evolution_out12(:, 1));

w_evolution_out21(:, 1) = mean(w_in(output2_neurons_label, output1_neurons_label), 2);
w_evolution_out21_mean(1) = mean(w_evolution_out21(:, 1));

w_evolution_in12(:, 1) = mean(w_in(input1_neurons_label, input2_neurons_label), 2);
w_evolution_in12_mean(1) = mean(w_evolution_in12(:, 1));

w_evolution_in21(:, 1) = mean(w_in(input2_neurons_label, input1_neurons_label), 2);
w_evolution_in21_mean(1) = mean(w_evolution_in21(:, 1));

w_evolution_out1in1(:, 1) = mean(w_in(output1_neurons_label, input1_neurons_label), 2);
w_evolution_out1in1_mean(1) = mean(w_evolution_out1in1(:, 1));

w_evolution_in1out1(:, 1) = mean(w_in(input1_neurons_label, output1_neurons_label), 2);
w_evolution_in1out1_mean(1) = mean(w_evolution_in1out1(:, 1));

w_evolution_out2in2(:, 1) = mean(w_in(output2_neurons_label, input2_neurons_label), 2);
w_evolution_out2in2_mean(1) = mean(w_evolution_out2in2(:, 1));

w_evolution_in2out2(:, 1) = mean(w_in(input2_neurons_label, output2_neurons_label), 2);
w_evolution_in2out2_mean(1) = mean(w_evolution_in2out2(:, 1));

%symmetry measure
s_evolution(1) = sym_measure(w);
s_out_evolution(1) = sym_measure(w(output_neurons_label, output_neurons_label));
s_in_evolution(1) = sym_measure(w(input_neurons_label, input_neurons_label));
s_out1_evolution(1) = sym_measure(w(output1_neurons_label, output1_neurons_label));
s_out2_evolution(1) = sym_measure(w(output2_neurons_label, output2_neurons_label));
s_in1_evolution(1) = sym_measure(w(input1_neurons_label, input1_neurons_label));
s_in2_evolution(1) = sym_measure(w(input2_neurons_label, input2_neurons_label));

w_temp = w;
w_temp(input_neurons_label, :) = 0;
w_temp(:, input_neurons_label) = 0;
w_temp(output1_neurons_label, output1_neurons_label) = 0;
w_temp(output2_neurons_label, output2_neurons_label) = 0;
s_out12_evolution(1) = sym_measure(w_temp);

w_temp = w;
w_temp(output_neurons_label, :) = 0;
w_temp(:, output_neurons_label) = 0;
w_temp(input1_neurons_label, input1_neurons_label) = 0;
w_temp(input2_neurons_label, input2_neurons_label) = 0;
s_in12_evolution(1) = sym_measure(w_temp);

w_temp = w;
w_temp(output2_neurons_label, :) = 0;
w_temp(:, output2_neurons_label) = 0;
w_temp(input2_neurons_label, :) = 0;
w_temp(:, input2_neurons_label) = 0;
w_temp(input1_neurons_label, input1_neurons_label) = 0;
w_temp(output1_neurons_label, output1_neurons_label) = 0;
s_out1_in1_evolution(1) = sym_measure(w_temp);

w_temp = w;
w_temp(output1_neurons_label, :) = 0;
w_temp(:, output1_neurons_label) = 0;
w_temp(input1_neurons_label, :) = 0;
w_temp(:, input1_neurons_label) = 0;
w_temp(input2_neurons_label, input2_neurons_label) = 0;
w_temp(output2_neurons_label, output2_neurons_label) = 0;
s_out2_in2_evolution(1) = sym_measure(w_temp);


%STP parameters for each synapse

%output1+input1+output2 ---> output1
temp = tau_f(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
temp(logical(temp==0)) = [];
tau_f_evolution_out1(:, 1) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
tau_f_evolution_out1_mean(1) = mean(tau_f_evolution_out1(:, 1));
tau_f_evolution_out1_sd(1) = std(tau_f_evolution_out1(:, 1));

temp = tau_d(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
temp(logical(temp==0)) = [];
tau_d_evolution_out1(:, 1) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
tau_d_evolution_out1_mean(1) = mean(tau_d_evolution_out1(:, 1));
tau_d_evolution_out1_sd(1) = std(tau_d_evolution_out1(:, 1));

temp = U(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
temp(logical(temp==0)) = [];
U_evolution_out1(:, 1) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
U_evolution_out1_mean(1) = mean(U_evolution_out1(:, 1));
U_evolution_out1_sd(1) = std(U_evolution_out1(:, 1));


%output2+input2+output1 ---> output2
temp = tau_f(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
temp(logical(temp==0)) = [];
tau_f_evolution_out2(:, 1) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
tau_f_evolution_out2_mean(1) = mean(tau_f_evolution_out2(:, 1));
tau_f_evolution_out2_sd(1) = std(tau_f_evolution_out2(:, 1));

temp = tau_d(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
temp(logical(temp==0)) = [];
tau_d_evolution_out2(:, 1) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
tau_d_evolution_out2_mean(1) = mean(tau_d_evolution_out2(:, 1));
tau_d_evolution_out2_sd(1) = std(tau_d_evolution_out2(:, 1));

temp = U(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
temp(logical(temp==0)) = [];
U_evolution_out2(:, 1) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
U_evolution_out2_mean(1) = mean(U_evolution_out2(:, 1));
U_evolution_out2_sd(1) = std(U_evolution_out2(:, 1));


% %OUTPUTS
% %output1 ---> output2
% temp = tau_f(output1_neurons_label, output1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_f_out11_evolution(:, 1) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
% tau_f_out11_evolution_mean(1) = mean(tau_f_out11_evolution(:, 1));
% 
% temp = tau_d(output1_neurons_label, output1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_d_out11_evolution(:, 1) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
% tau_d_out11_evolution_mean(1) = mean(tau_d_out11_evolution(:, 1));
% 
% temp = U(output1_neurons_label, output1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% U_out11_evolution(:, 1) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
% U_out11_evolution_mean(1) = mean(U_out11_evolution(:, 1));
% 
% 
% %output2 ---> output1
% temp = tau_f(output1_neurons_label, output2_neurons_label)';
% tau_f_out12_evolution(:, 1) = mean(temp, 1)';
% tau_f_out12_evolution_mean(1) = mean(tau_f_out12_evolution(:, 1));
% 
% temp = tau_d(output1_neurons_label, output2_neurons_label)';
% tau_d_out12_evolution(:, 1) = mean(temp, 1)';
% tau_d_out12_evolution_mean(1) = mean(tau_d_out12_evolution(:, 1));
% 
% temp = U(output1_neurons_label, output2_neurons_label)';
% U_out12_evolution(:, 1) = mean(temp, 1)';
% U_out12_evolution_mean(1) = mean(U_out12_evolution(:, 1));
% 
% 
% %output2 ---> output2
% temp = tau_f(output2_neurons_label, output2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_f_out22_evolution(:, 1) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
% tau_f_out22_evolution_mean(1) = mean(tau_f_out22_evolution(:, 1));
% 
% temp = tau_d(output2_neurons_label, output2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_d_out22_evolution(:, 1) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
% tau_d_out22_evolution_mean(1) = mean(tau_d_out22_evolution(:, 1));
% 
% temp = U(output2_neurons_label, output2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% U_out22_evolution(:, 1) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
% U_out22_evolution_mean(1) = mean(U_out22_evolution(:, 1));
% 
% 
% %output1 ---> output2
% temp = tau_f(output2_neurons_label, output1_neurons_label)';
% tau_f_out21_evolution(:, 1) = mean(temp, 1)';
% tau_f_out21_evolution_mean(1) = mean(tau_f_out21_evolution(:, 1));
% 
% temp = tau_d(output2_neurons_label, output1_neurons_label)';
% tau_d_out21_evolution(:, 1) = mean(temp, 1)';
% tau_d_out21_evolution_mean(1) = mean(tau_d_out21_evolution(:, 1));
% 
% temp = U(output2_neurons_label, output1_neurons_label)';
% U_out21_evolution(:, 1) = mean(temp, 1)';
% U_out21_evolution_mean(1) = mean(U_out21_evolution(:, 1));
% 
% 
% %INPUTS
% %input1 ---> input1
% temp = tau_f(input1_neurons_label, input1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_f_in11_evolution(:, 1) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
% tau_f_in11_evolution_mean(1) = mean(tau_f_in11_evolution(:, 1));
% 
% temp = tau_d(input1_neurons_label, input1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_d_in11_evolution(:, 1) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
% tau_d_in11_evolution_mean(1) = mean(tau_d_in11_evolution(:, 1));
% 
% temp = U(input1_neurons_label, input1_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% U_in11_evolution(:, 1) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
% U_in11_evolution_mean(1) = mean(U_in11_evolution(:, 1));
% 
% 
% %input2 ---> input1
% temp = tau_f(input1_neurons_label, input2_neurons_label)';
% tau_f_in12_evolution(:, 1) = mean(temp, 1)';
% tau_f_in12_evolution_mean(1) = mean(tau_f_in12_evolution(:, 1));
% 
% temp = tau_d(input1_neurons_label, input2_neurons_label)';
% tau_d_in12_evolution(:, 1) = mean(temp, 1)';
% tau_d_in12_evolution_mean(1) = mean(tau_d_in12_evolution(:, 1));
% 
% temp = U(input1_neurons_label, input2_neurons_label)';
% U_in12_evolution(:, 1) = mean(temp, 1)';
% U_in12_evolution_mean(1) = mean(U_in12_evolution(:, 1));
% 
% 
% %input2 ---> input2
% temp = tau_f(input2_neurons_label, input2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_f_in22_evolution(:, 1) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
% tau_f_in22_evolution_mean(1) = mean(tau_f_in22_evolution(:, 1));
% 
% temp = tau_d(input2_neurons_label, input2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% tau_d_in22_evolution(:, 1) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
% tau_d_in22_evolution_mean(1) = mean(tau_d_in22_evolution(:, 1));
% 
% temp = U(input2_neurons_label, input2_neurons_label)';
% temp(logical(eye(size(temp)))) = [];
% U_in22_evolution(:, 1) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
% U_in22_evolution_mean(1) = mean(U_in22_evolution(:, 1));
% 
% 
% %input1 ---> input2
% temp = tau_f(input2_neurons_label, input1_neurons_label)';
% tau_f_in21_evolution(:, 1) = mean(temp, 1)';
% tau_f_in21_evolution_mean(1) = mean(tau_f_in21_evolution(:, 1));
% 
% temp = tau_d(input2_neurons_label, input1_neurons_label)';
% tau_d_in21_evolution(:, 1) = mean(temp, 1)';
% tau_d_in21_evolution_mean(1) = mean(tau_d_in21_evolution(:, 1));
% 
% temp = U(input2_neurons_label, input1_neurons_label)';
% U_in21_evolution(:, 1) = mean(temp, 1)';
% U_in21_evolution_mean(1) = mean(U_in21_evolution(:, 1));
% 
% 
% %MIXED INPUTS-OUTPUTS
% %input1 ---> output1
% temp = tau_f(output1_neurons_label, input1_neurons_label)';
% tau_f_out1_in1_evolution(:, 1) = mean(temp, 1)';
% tau_f_out1_in1_evolution_mean(1) = mean(tau_f_out1_in1_evolution(:, 1));
% 
% temp = tau_d(output1_neurons_label, input1_neurons_label)';
% tau_d_out1_in1_evolution(:, 1) = mean(temp, 1)';
% tau_d_out1_in1_evolution_mean(1) = mean(tau_d_out1_in1_evolution(:, 1));
% 
% temp = U(output1_neurons_label, input1_neurons_label)';
% U_out1_in1_evolution(:, 1) = mean(temp, 1)';
% U_out1_in1_evolution_mean(1) = mean(U_out1_in1_evolution(:, 1));
% 
% 
% %output1 ---> input1
% temp = tau_f(input1_neurons_label, output1_neurons_label)';
% tau_f_in1_out1_evolution(:, 1) = mean(temp, 1)';
% tau_f_in1_out1_evolution_mean(1) = mean(tau_f_in1_out1_evolution(:, 1));
% 
% temp = tau_d(input1_neurons_label, output1_neurons_label)';
% tau_d_in1_out1_evolution(:, 1) = mean(temp, 1)';
% tau_d_in1_out1_evolution_mean(1) = mean(tau_d_in1_out1_evolution(:, 1));
% 
% temp = U(input1_neurons_label, output1_neurons_label)';
% U_in1_out1_evolution(:, 1) = mean(temp, 1)';
% U_in1_out1_evolution_mean(1) = mean(U_in1_out1_evolution(:, 1));
% 
% %input2 ---> output2
% temp = tau_f(output2_neurons_label, input2_neurons_label)';
% tau_f_out2_in2_evolution(:, 1) = mean(temp, 1)';
% tau_f_out2_in2_evolution_mean(1) = mean(tau_f_out2_in2_evolution(:, 1));
% 
% temp = tau_d(output2_neurons_label, input2_neurons_label)';
% tau_d_out2_in2_evolution(:, 1) = mean(temp, 1)';
% tau_d_out2_in2_evolution_mean(1) = mean(tau_d_out2_in2_evolution(:, 1));
% 
% temp = U(output2_neurons_label, input2_neurons_label)';
% U_out2_in2_evolution(:, 1) = mean(temp, 1)';
% U_out2_in2_evolution_mean(1) = mean(U_out2_in2_evolution(:, 1));
% 
% 
% %output2 ---> input2
% temp = tau_f(input2_neurons_label, output2_neurons_label)';
% tau_f_in2_out2_evolution(:, 1) = mean(temp, 1)';
% tau_f_in2_out2_evolution_mean(1) = mean(tau_f_in2_out2_evolution(:, 1));
% 
% temp = tau_d(input2_neurons_label, output2_neurons_label)';
% tau_d_in2_out2_evolution(:, 1) = mean(temp, 1)';
% tau_d_in2_out2_evolution_mean(1) = mean(tau_d_in2_out2_evolution(:, 1));
% 
% temp = U(input2_neurons_label, output2_neurons_label)';
% U_in2_out2_evolution(:, 1) = mean(temp, 1)';
% U_in2_out2_evolution_mean(1) = mean(U_in2_out2_evolution(:, 1));


%% Learning

regime_counter = 1;
for i = 2 : n_sample       
   
    if i > timesteps_for_firing_rate
        neurons_spike_logical = circshift(neurons_spike_logical, [0, -1]);
        neurons_spike_logical(:, timesteps_for_firing_rate) = 0;
    end
            
    % compute the voltage of neurons
    V(:, i) = Euler_integration_conductance_based_IF_multi_synapses( V(:,i-1), E, g, tau_g, g_L, dt_simul, N);        
       
    % decay of synaptic variables for those synapses who were not involving in any firing event at the previous timestep
    g = g .* exp( -dt_simul / tau_g); 
    g = g - diag(diag(g));   
    
    D = 1 - (1 - D) .* exp( -dt_simul ./ tau_d);
    D = D - diag(diag(D));
    
    F = U + (F - U) .* exp( -dt_simul ./ tau_f);
    F = F - diag(diag(F));
    
    % Synaptic variables decay
    o1 = o1 * exp(-dt_simul / tauneg);  % Update post synaptic variable: exponential decay
    r1 = r1 * exp(-dt_simul / taupos);  % Update pre synaptic variable: exponential decay
    o2 = o2 * exp(-dt_simul / tauy);    % Update post synaptic variable: exponential decay
    r2 = r2 * exp(-dt_simul / taux);    % Update pre synaptic variable: exponential decay
                    
    % apply the input if it is the case
    V(input_neurons_label, i) = V(input_neurons_label, i) + dV_input .* (next_input_times <= t(i));    %update only the voltage of the neurons (among the input neurons, selected through input_neurons_label) that received the input, selected through input_times <= t(i)
    
    % set to zero the voltage of those neurons who have fired at the previous timestep
    V(:, i) = V(:, i) .* (~neurons_previously_fired_logical);
    
    % refractoriness implementation - step 1
    V(:, i) = V(:, i) .* (time_from_last_spike(:) >= (refr_period * 1e+3));
    
    % evaluate if there is some neuron firing
    neurons_currently_firing_logical = (V(:, i) >= V_th);
    
    % refractoriness implementation - step 2
    time_from_last_spike(neurons_currently_firing_logical) = 0;
        
    if sum(V(:, i) >= V_th) > 0        
        
        V(:, i) = V(:, i) .* (~neurons_currently_firing_logical) + V_th .* (neurons_currently_firing_logical);
                
        % apply the effect of the (eventual) spike at the previous timestep on those synapses who were involving in any firing event at the previous timestep
        g = g + w .* D .* F .* repmat(neurons_currently_firing_logical', N, 1);
        g = g - diag(diag(g));
                
        D = D - D .* F .* repmat(neurons_currently_firing_logical', N, 1);
        D = D - diag(diag(D));

        F = F + U .* (1 - F) .* repmat(neurons_currently_firing_logical', N, 1);
        F = F - diag(diag(F));
        
        neurons_currently_firing_logical_index = find(neurons_currently_firing_logical);
        if i > timesteps_for_firing_rate  %learning starts not at the beginning of the simulation
            k = 1;
            while (k <= size(neurons_currently_firing_logical_index,1)) && (size(neurons_currently_firing_logical_index,2) ~= 0)
                w(:, neurons_currently_firing_logical_index(k)) = w(:, neurons_currently_firing_logical_index(k)) - learning_rate * o1 * (A2neg + A3neg * r2(neurons_currently_firing_logical_index(k)));        % Depression of synaptic weights from the neurons who have fired (post-pre)
                w(neurons_currently_firing_logical_index(k), :) = w(neurons_currently_firing_logical_index(k), :) + learning_rate * (r1 * (A2pos + A3pos * o2(neurons_currently_firing_logical_index(k))))';     % Potentiation of synaptic weights onto the neurons who have fired (pre-post)
                w = w - diag(diag(w));                      % Eliminate self-interaction                            
                k = k + 1;
            end
        end
            
        if i <= timesteps_for_firing_rate
            neurons_spike_logical(:, i) = neurons_currently_firing_logical;
        else
            neurons_spike_logical(:, timesteps_for_firing_rate) = neurons_currently_firing_logical;
        end
        
    end
    
    % Weights bounds
%     w(input1_neurons_label, input2_neurons_label) = w(input1_neurons_label, input2_neurons_label) .* (w(input1_neurons_label, input2_neurons_label) <= w_max_between) + w_max_between * (w(input1_neurons_label, input2_neurons_label) > w_max_between);
%     w(input2_neurons_label, input1_neurons_label) = w(input2_neurons_label, input1_neurons_label) .* (w(input2_neurons_label, input1_neurons_label) <= w_max_between) + w_max_between * (w(input2_neurons_label, input1_neurons_label) > w_max_between);
%     w(output1_neurons_label, output2_neurons_label) = w(output1_neurons_label, output2_neurons_label) .* (w(output1_neurons_label, output2_neurons_label) <= w_max_between) + w_max_between * (w(output1_neurons_label, output2_neurons_label) > w_max_between);
%     w(output2_neurons_label, output1_neurons_label) = w(output2_neurons_label, output1_neurons_label) .* (w(output2_neurons_label, output1_neurons_label) <= w_max_between) + w_max_between * (w(output2_neurons_label, output1_neurons_label) > w_max_between);
     
    w = w .* (w >= w_min) + w_min * (w < w_min);    % Lower bound
    w = w .* (w <= w_max) + w_max * (w > w_max);    % Upper bound
            
    %w(input1_neurons_label, input2_neurons_label) = 0.;
    %w(input2_neurons_label, input1_neurons_label) = 0.;
    %w(output1_neurons_label, output2_neurons_label) = 0.;
    %w(output2_neurons_label, output1_neurons_label) = 0.;
    
    w(output2_neurons_label, input1_neurons_label) = 0;
    w(input1_neurons_label, output2_neurons_label) = 0;
    w(output1_neurons_label, input2_neurons_label) = 0;
    w(input2_neurons_label, output1_neurons_label) = 0;
    
    w = w - diag(diag(w));        %eliminate self-interaction
    
    % For STP rule
%     w_norm = w;
%     w_norm(input1_neurons_label, input2_neurons_label) = w(input1_neurons_label, input2_neurons_label) / w_max_between;
%     w_norm(input2_neurons_label, input1_neurons_label) = w(input2_neurons_label, input1_neurons_label) / w_max_between;
%     w_norm(output1_neurons_label, output2_neurons_label) = w(output1_neurons_label, output2_neurons_label) / w_max_between;
%     w_norm(output2_neurons_label, output1_neurons_label) = w(output2_neurons_label, output1_neurons_label) / w_max_between;    
    
    % Symmetry measure
    s_evolution(i) = sym_measure(w);
    s_out_evolution(i) = sym_measure(w(output_neurons_label, output_neurons_label));
    s_in_evolution(i) = sym_measure(w(input_neurons_label, input_neurons_label));
    s_out1_evolution(i) = sym_measure(w(output1_neurons_label, output1_neurons_label));
    s_out2_evolution(i) = sym_measure(w(output2_neurons_label, output2_neurons_label));
    s_in1_evolution(i) = sym_measure(w(input1_neurons_label, input1_neurons_label));
    s_in2_evolution(i) = sym_measure(w(input2_neurons_label, input2_neurons_label));
    
    w_temp = w;
    w_temp(input_neurons_label, :) = 0;
    w_temp(:, input_neurons_label) = 0;
    w_temp(output1_neurons_label, output1_neurons_label) = 0;
    w_temp(output2_neurons_label, output2_neurons_label) = 0;        
    s_out12_evolution(i) = sym_measure(w_temp);
    
    w_temp = w;
    w_temp(output_neurons_label, :) = 0;
    w_temp(:, output_neurons_label) = 0;
    w_temp(input1_neurons_label, input1_neurons_label) = 0;
    w_temp(input2_neurons_label, input2_neurons_label) = 0;
    s_in12_evolution(i) = sym_measure(w_temp);

    w_temp = w;
    w_temp(output2_neurons_label, :) = 0;
    w_temp(:, output2_neurons_label) = 0;
    w_temp(input2_neurons_label, :) = 0;
    w_temp(:, input2_neurons_label) = 0;
    w_temp(input1_neurons_label, input1_neurons_label) = 0;
    w_temp(output1_neurons_label, output1_neurons_label) = 0;
    s_out1_in1_evolution(i) = sym_measure(w_temp);

    w_temp = w;
    w_temp(output1_neurons_label, :) = 0;
    w_temp(:, output1_neurons_label) = 0;
    w_temp(input1_neurons_label, :) = 0;
    w_temp(:, input1_neurons_label) = 0;
    w_temp(input2_neurons_label, input2_neurons_label) = 0;
    w_temp(output2_neurons_label, output2_neurons_label) = 0;
    s_out2_in2_evolution(i) = sym_measure(w_temp);    
         
    % Storing single neuron's weights
    w_evolution_out1_1(:, i) = w(output1_neurons_label(1), input_neurons_label)';
    w_evolution_out1_2(:, i) = w(output1_neurons_label(2), input_neurons_label)';
    w_evolution_out2_1(:, i) = w(output2_neurons_label(1), input_neurons_label)';
    w_evolution_out2_2(:, i) = w(output2_neurons_label(2), input_neurons_label)';
    
    % Storing mean weights
    temp = w(output1_neurons_label, output1_neurons_label)';
    temp(logical(eye(size(temp)))) = [];
    w_evolution_out11(:, i) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
    w_evolution_out11_mean(i) = mean(w_evolution_out11(:, i));

    temp = w(output2_neurons_label, output2_neurons_label)';
    temp(logical(eye(size(temp)))) = [];
    w_evolution_out22(:, i) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
    w_evolution_out22_mean(i) = mean(w_evolution_out22(:, i));

    temp = w(input1_neurons_label, input1_neurons_label)';
    temp(logical(eye(size(temp)))) = [];
    w_evolution_in11(:, i) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
    w_evolution_in11_mean(i) = mean(w_evolution_in11(:, i));

    temp = w(input2_neurons_label, input2_neurons_label)';
    temp(logical(eye(size(temp)))) = [];
    w_evolution_in22(:, i) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
    w_evolution_in22_mean(i) = mean(w_evolution_in22(:, i));

    w_evolution_out12(:, i) = mean(w(output1_neurons_label, output2_neurons_label), 2);
    w_evolution_out12_mean(i) = mean(w_evolution_out12(:, i));

    w_evolution_out21(:, i) = mean(w(output2_neurons_label, output1_neurons_label), 2);
    w_evolution_out21_mean(i) = mean(w_evolution_out21(:, i));

    w_evolution_in12(:, i) = mean(w(input1_neurons_label, input2_neurons_label), 2);
    w_evolution_in12_mean(i) = mean(w_evolution_in12(:, i));

    w_evolution_in21(:, i) = mean(w(input2_neurons_label, input1_neurons_label), 2);
    w_evolution_in21_mean(i) = mean(w_evolution_in21(:, i));

    w_evolution_out1in1(:, i) = mean(w(output1_neurons_label, input1_neurons_label), 2);
    w_evolution_out1in1_mean(i) = mean(w_evolution_out1in1(:, i));

    w_evolution_in1out1(:, i) = mean(w(input1_neurons_label, output1_neurons_label), 2);
    w_evolution_in1out1_mean(i) = mean(w_evolution_in1out1(:, i));

    w_evolution_out2in2(:, i) = mean(w(output2_neurons_label, input2_neurons_label), 2);
    w_evolution_out2in2_mean(i) = mean(w_evolution_out2in2(:, i));

    w_evolution_in2out2(:, i) = mean(w(input2_neurons_label, output2_neurons_label), 2);
    w_evolution_in2out2_mean(i) = mean(w_evolution_in2out2(:, i));
        
    % Synaptic variables increase
    o1 = (neurons_currently_firing_logical==0) .* o1 + (neurons_currently_firing_logical~=0); % Post synaptic variable: saturation to 1 for those neurons that have fired
    r1 = (neurons_currently_firing_logical==0) .* r1 + (neurons_currently_firing_logical~=0); % Pre synaptic variable: saturation to 1 for those neurons that have fired
    o2 = (neurons_currently_firing_logical==0) .* o2 + (neurons_currently_firing_logical~=0); % Post synaptic variable: saturation to 1 for those neurons that have fired
    r2 = (neurons_currently_firing_logical==0) .* r2 + (neurons_currently_firing_logical~=0); % Pre synaptic variable: saturation to 1 for those neurons that have fired
           
    % update firing variable for neurons
    neurons_previously_fired_logical = neurons_currently_firing_logical;
        
    % update firing variable for inputs
    next_input_times = next_input_times + ( (1/rho_input) + ( jitter_amplitude * rand(N_inputs, 1) - (jitter_amplitude/2) ) ) .* (next_input_times <= t(i));    % the (jitter_amplitude/2) is to center the firing times on the unjittered times
    
    % refractoriness implementation - step 3
    time_from_last_spike = time_from_last_spike + 1;
    
    % store information to compute firing rates
    if i >= timesteps_for_firing_rate
        
        % normal unweighted average
        %firing_rates(:, i) = mean(neurons_spike_logical, 2) * 1e3;
        
        % exponential decay average (running average)
        if i == timesteps_for_firing_rate
            firing_rates(:, i) = mean(neurons_spike_logical, 2) * 1e3;
        else
            firing_rates(:, i) = firing_rates(:, i-1) .* (1-exp_decay_mean_firing_rate) + neurons_spike_logical(:, end) * 1e3 .* exp_decay_mean_firing_rate;
        end
                
        mean_firing_rate(output1_neurons_label, i) = mean(firing_rates(output1_neurons_label, i));
        mean_firing_rate(output2_neurons_label, i) = mean(firing_rates(output2_neurons_label, i));
        mean_firing_rate(input1_neurons_label, i) = mean(firing_rates(input1_neurons_label, i));
        mean_firing_rate(input2_neurons_label, i) = mean(firing_rates(input2_neurons_label, i));
    end
    
    if i > timesteps_for_firing_rate  %learning starts not at the beginning of the simulation            
        % global error
        error(:, i) = error(:, i-1) .* ( ~neurons_currently_firing_logical) + (target_firing_rate - mean_firing_rate(:, i) ) .* neurons_currently_firing_logical;
        
        U_squared = U.^2;
        U_squared = U_squared + eye(N,N);  %to avoid NAN in the diagonal when in the STP rule we divide by U.^U
        tau_d_squared = tau_d.^2;
        tau_d_squared = tau_d_squared + eye(N,N);
                
        % error-dependent learning rules
        U = U - 0.2 * (1 + ((repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N )).^2) .* ( w ./ (U_squared) ) .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );         
        tau_d = tau_d - 0.2 * (1 + ((repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N )).^2) .* ( w ./ (tau_d_squared) ) .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );
        tau_f = tau_f + 0.2 * (1 + ((repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N )).^2) .* w .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );        
        
        % STP rule for synaptic strength
        w = w + learning_rate * ( 1 ./ (abs(tau_d_squared)) ) .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );   
        
        % boundaries                           
        U = (U >= U_upper ) .* U_upper + (U < U_upper ) .* U;
        U = (U >= U_lower ) .* U + (U < U_lower ) .* U_lower;
        U = U - diag(diag(U));
        tau_d = (tau_d >= tau_d_lower ) .* tau_d + (tau_d < tau_d_lower ) .* tau_d_lower;
        tau_d = (tau_d >= tau_d_upper ) .* tau_d_upper + (tau_d < tau_d_upper ) .* tau_d;
        tau_d = tau_d - diag(diag(tau_d));
        tau_f = (tau_f >= tau_f_lower ) .* tau_f + (tau_f < tau_f_lower ) .* tau_f_lower;
        tau_f = (tau_f >= tau_f_upper ) .* tau_f_upper + (tau_f < tau_f_upper ) .* tau_f;
        tau_f = tau_f - diag(diag(tau_f));
        
        % store STP parameters of single neurons
        
        %output1+input1+output2 ---> output1
        temp = tau_f(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
        temp(logical(temp==0)) = [];
        tau_f_evolution_out1(:, i) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
        tau_f_evolution_out1_mean(i) = mean(tau_f_evolution_out1(:, i));
        tau_f_evolution_out1_sd(i) = std(tau_f_evolution_out1(:, i));

        temp = tau_d(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
        temp(logical(temp==0)) = [];
        tau_d_evolution_out1(:, i) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
        tau_d_evolution_out1_mean(i) = mean(tau_d_evolution_out1(:, i));
        tau_d_evolution_out1_sd(i) = std(tau_d_evolution_out1(:, i));

        temp = U(output1_neurons_label, [output1_neurons_label, input1_neurons_label, output2_neurons_label])';
        temp(logical(temp==0)) = [];
        U_evolution_out1(:, i) = mean(reshape(temp, N_outputs1+N_inputs1+N_outputs2-1, N_outputs1))';
        U_evolution_out1_mean(i) = mean(U_evolution_out1(:, i));
        U_evolution_out1_sd(i) = std(U_evolution_out1(:, i));


        %output2+input2+output1 ---> output2
        temp = tau_f(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
        temp(logical(temp==0)) = [];
        tau_f_evolution_out2(:, i) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
        tau_f_evolution_out2_mean(i) = mean(tau_f_evolution_out2(:, i));
        tau_f_evolution_out2_sd(i) = std(tau_f_evolution_out2(:, i));

        temp = tau_d(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
        temp(logical(temp==0)) = [];
        tau_d_evolution_out2(:, i) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
        tau_d_evolution_out2_mean(i) = mean(tau_d_evolution_out2(:, i));
        tau_d_evolution_out2_sd(i) = std(tau_d_evolution_out2(:, i));

        temp = U(output2_neurons_label, [output2_neurons_label, input2_neurons_label, output1_neurons_label])';
        temp(logical(temp==0)) = [];
        U_evolution_out2(:, i) = mean(reshape(temp, N_outputs2+N_inputs2+N_outputs1-1, N_outputs2))';
        U_evolution_out2_mean(i) = mean(U_evolution_out2(:, i));
        U_evolution_out2_sd(i) = std(U_evolution_out2(:, i));
        
        
%         %OUTPUTS
%         %output1 ---> output1
%         temp = tau_f(output1_neurons_label, output1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_f_out11_evolution(:, i) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
%         tau_f_out11_evolution_mean(i) = mean(tau_f_out11_evolution(:, i));
%         
%         temp = tau_d(output1_neurons_label, output1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_d_out11_evolution(:, i) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
%         tau_d_out11_evolution_mean(i) = mean(tau_d_out11_evolution(:, i));
% 
%         temp = U(output1_neurons_label, output1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         U_out11_evolution(:, i) = mean(reshape(temp, N_outputs1-1, N_outputs1))';
%         U_out11_evolution_mean(i) = mean(U_out11_evolution(:, i));
% 
% 
%         %output2 ---> output1
%         temp = tau_f(output1_neurons_label, output2_neurons_label)';
%         tau_f_out12_evolution(:, i) = mean(temp, 1);
%         tau_f_out12_evolution_mean(i) = mean(tau_f_out12_evolution(:, i));
%         
%         temp = tau_d(output1_neurons_label, output2_neurons_label)';
%         tau_d_out12_evolution(:, i) = mean(temp, 1);
%         tau_d_out12_evolution_mean(i) = mean(tau_d_out12_evolution(:, i));
% 
%         temp = U(output1_neurons_label, output2_neurons_label)';            
%         U_out12_evolution(:, i) = mean(temp, 1)';
%         U_out12_evolution_mean(i) = mean(U_out12_evolution(:, i));
% 
% 
%         %output2 ---> output2
%         temp = tau_f(output2_neurons_label, output2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_f_out22_evolution(:, i) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
%         tau_f_out22_evolution_mean(i) = mean(tau_f_out22_evolution(:, i));
%         
%         temp = tau_d(output2_neurons_label, output2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_d_out22_evolution(:, i) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
%         tau_d_out22_evolution_mean(i) = mean(tau_d_out22_evolution(:, i));
% 
%         temp = U(output2_neurons_label, output2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         U_out22_evolution(:, i) = mean(reshape(temp, N_outputs2-1, N_outputs2))';
%         U_out22_evolution_mean(i) = mean(U_out22_evolution(:, i));
% 
% 
%         %output1 ---> output2
%         temp = tau_f(output2_neurons_label, output1_neurons_label)';
%         tau_f_out21_evolution(:, i) = mean(temp, 1)';
%         tau_f_out21_evolution_mean(i) = mean(tau_f_out21_evolution(:, i));
%         
%         temp = tau_d(output2_neurons_label, output1_neurons_label)';
%         tau_d_out21_evolution(:, i) = mean(temp, 1)';
%         tau_d_out21_evolution_mean(i) = mean(tau_d_out21_evolution(:, i));
% 
%         temp = U(output2_neurons_label, output1_neurons_label)';
%         U_out21_evolution(:, i) = mean(temp, 1)';
%         U_out21_evolution_mean(i) = mean(U_out21_evolution(:, i));
% 
% 
%         %INPUTS
%         %input1 ---> input1
%         temp = tau_f(input1_neurons_label, input1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_f_in11_evolution(:, i) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
%         tau_f_in11_evolution_mean(i) = mean(tau_f_in11_evolution(:, i));
%         
%         temp = tau_d(input1_neurons_label, input1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_d_in11_evolution(:, i) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
%         tau_d_in11_evolution_mean(i) = mean(tau_d_in11_evolution(:, i));
% 
%         temp = U(input1_neurons_label, input1_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         U_in11_evolution(:, i) = mean(reshape(temp, N_inputs1-1, N_inputs1))';
%         U_in11_evolution_mean(i) = mean(U_in11_evolution(:, i));
% 
% 
%         %input2 ---> input1
%         temp = tau_f(input1_neurons_label, input2_neurons_label)';
%         tau_f_in12_evolution(:, i) = mean(temp, 1)';
%         tau_f_in12_evolution_mean(i) = mean(tau_f_in12_evolution(:, i));
%         
%         temp = tau_d(input1_neurons_label, input2_neurons_label)';
%         tau_d_in12_evolution(:, i) = mean(temp, 1)';
%         tau_d_in12_evolution_mean(i) = mean(tau_d_in12_evolution(:, i));
% 
%         temp = U(input1_neurons_label, input2_neurons_label)';
%         U_in12_evolution(:, i) = mean(temp, 1)';
%         U_in12_evolution_mean(i) = mean(U_in12_evolution(:, i));
% 
% 
%         %input2 ---> input2
%         temp = tau_f(input2_neurons_label, input2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_f_in22_evolution(:, i) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
%         tau_f_in22_evolution_mean(i) = mean(tau_f_in22_evolution(:, i));
%         
%         temp = tau_d(input2_neurons_label, input2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         tau_d_in22_evolution(:, i) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
%         tau_d_in22_evolution_mean(i) = mean(tau_d_in22_evolution(:, i));
% 
%         temp = U(input2_neurons_label, input2_neurons_label)';
%         temp(logical(eye(size(temp)))) = [];
%         U_in22_evolution(:, i) = mean(reshape(temp, N_inputs2-1, N_inputs2))';
%         U_in22_evolution_mean(i) = mean(U_in22_evolution(:, i));
% 
% 
%         %input1 ---> input2
%         temp = tau_f(input2_neurons_label, input1_neurons_label)';
%         tau_f_in21_evolution(:, i) = mean(temp, 1)';
%         tau_f_in21_evolution_mean(i) = mean(tau_f_in21_evolution(:, i));
%         
%         temp = tau_d(input2_neurons_label, input1_neurons_label)';
%         tau_d_in21_evolution(:, i) = mean(temp, 1)';
%         tau_d_in21_evolution_mean(i) = mean(tau_d_in21_evolution(:, i));
% 
%         temp = U(input2_neurons_label, input1_neurons_label)';
%         U_in21_evolution(:, i) = mean(temp, 1)';
%         U_in21_evolution_mean(i) = mean(U_in21_evolution(:, i));
% 
% 
%         %MIXED INPUTS-OUTPUTS
%         %input1 ---> output1
%         temp = tau_f(output1_neurons_label, input1_neurons_label)';
%         tau_f_out1_in1_evolution(:, i) = mean(temp, 1)';
%         tau_f_out1_in1_evolution_mean(i) = mean(tau_f_out1_in1_evolution(:, i));
%         
%         temp = tau_d(output1_neurons_label, input1_neurons_label)';
%         tau_d_out1_in1_evolution(:, i) = mean(temp, 1)';
%         tau_d_out1_in1_evolution_mean(i) = mean(tau_d_out1_in1_evolution(:, i));
% 
%         temp = U(output1_neurons_label, input1_neurons_label)';
%         U_out1_in1_evolution(:, i) = mean(temp, 1)';
%         U_out1_in1_evolution_mean(i) = mean(U_out1_in1_evolution(:, i));
% 
% 
%         %output1 ---> input1
%         temp = tau_f(input1_neurons_label, output1_neurons_label)';
%         tau_f_in1_out1_evolution(:, i) = mean(temp, 1)';
%         tau_f_in1_out1_evolution_mean(i) = mean(tau_f_in1_out1_evolution(:, i));
%         
%         temp = tau_d(input1_neurons_label, output1_neurons_label)';
%         tau_d_in1_out1_evolution(:, i) = mean(temp, 1)';
%         tau_d_in1_out1_evolution_mean(i) = mean(tau_d_in1_out1_evolution(:, i));
% 
%         temp = U(input1_neurons_label, output1_neurons_label)';
%         U_in1_out1_evolution(:, i) = mean(temp, 1)';
%         U_in1_out1_evolution_mean(i) = mean(U_in1_out1_evolution(:, i));
% 
%         %input2 ---> output2
%         
%         temp = tau_f(output2_neurons_label, input2_neurons_label)';
%         tau_f_out2_in2_evolution(:, i) = mean(temp, 1)';
%         tau_f_out2_in2_evolution_mean(i) = mean(tau_f_out2_in2_evolution(:, i));
%         
%         temp = tau_d(output2_neurons_label, input2_neurons_label)';
%         tau_d_out2_in2_evolution(:, i) = mean(temp, 1)';
%         tau_d_out2_in2_evolution_mean(i) = mean(tau_d_out2_in2_evolution(:, i));
% 
%         temp = U(output2_neurons_label, input2_neurons_label)';
%         U_out2_in2_evolution(:, i) = mean(temp, 1)';
%         U_out2_in2_evolution_mean(i) = mean(U_out2_in2_evolution(:, i));
% 
% 
%         %output2 ---> input2
%         temp = tau_f(input2_neurons_label, output2_neurons_label)';
%         tau_f_in2_out2_evolution(:, i) = mean(temp, 1)';
%         tau_f_in2_out2_evolution_mean(i) = mean(tau_f_in2_out2_evolution(:, i));
%         
%         temp = tau_d(input2_neurons_label, output2_neurons_label)';
%         tau_d_in2_out2_evolution(:, i) = mean(temp, 1)';
%         tau_d_in2_out2_evolution_mean(i) = mean(tau_d_in2_out2_evolution(:, i));
% 
%         temp = U(input2_neurons_label, output2_neurons_label)';
%         U_in2_out2_evolution(:, i) = mean(temp, 1)';
%         U_in2_out2_evolution_mean(i) = mean(U_in2_out2_evolution(:, i));
    end
    
    display(i)
    
end

w_out1 = w(output1_neurons_label, output1_neurons_label);
s_out1 = sym_measure(w_out1);
w_out2 = w(output2_neurons_label, output2_neurons_label);
s_out2 = sym_measure(w_out2);

w_in1 = w(input1_neurons_label, input1_neurons_label);
s_in1 = sym_measure(w_in1);
w_in2 = w(input2_neurons_label, input2_neurons_label);
s_in2 = sym_measure(w_in2);

syms u
theoretical_mean_unif_noprune = 0.6137;
theoretical_variance_unif_noprune = 0.0017;
f = (1 / sqrt(2*pi*theoretical_variance_unif_noprune)) * exp( - (u - theoretical_mean_unif_noprune)^2 / (2*theoretical_variance_unif_noprune) );

p_s_out1 = 2 * ( double(int(f, 0, s_out1)) * (s_out1 < theoretical_mean_unif_noprune) + double(int(f, s_out1, 1)) * (s_out1 >= theoretical_mean_unif_noprune) );
p_s_out2 = 2 * ( double(int(f, 0, s_out2)) * (s_out2 < theoretical_mean_unif_noprune) + double(int(f, s_out2, 1)) * (s_out2 >= theoretical_mean_unif_noprune) );
p_s_in1 = 2 * ( double(int(f, 0, s_in1)) * (s_in1 < theoretical_mean_unif_noprune) + double(int(f, s_in1, 1)) * (s_in1 >= theoretical_mean_unif_noprune) );
p_s_in2 = 2 * ( double(int(f, 0, s_in2)) * (s_in2 < theoretical_mean_unif_noprune) + double(int(f, s_in2, 1)) * (s_in2 >= theoretical_mean_unif_noprune) );


%% Save, Print and plot

sprintf('Symmetry measure for the first output population of %d neurons: s=%f\n', N_outputs1, s_out1)
sprintf('Symmetry measure for the second output population of %d neurons: s=%f\n', N_outputs2, s_out2)
sprintf('Symmetry measure for the first input population of %d neurons: s=%f\n', N_inputs1, s_in1)
sprintf('Symmetry measure for the second input population of %d neurons: s=%f\n', N_inputs2, s_in2)


figure(1);
plot(t(timesteps_for_firing_rate+1:end), mean(abs(error(output1_neurons_label,timesteps_for_firing_rate+1:end))), 'LineWidth', lineThickness, 'Color', 'k');
hold on
plot(t(timesteps_for_firing_rate+1:end), mean(abs(error(output2_neurons_label,timesteps_for_firing_rate+1:end))), 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Mean error (Hz)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Error_out');


figure(2);
confplot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1) );
hold on
confplot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output2_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1) );
hold on
p1 = plot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output1_neurons_label, timesteps_for_firing_rate+1:end-1), 1), 'LineWidth', lineThickness-1);
set(p1, 'Color', [1 1 1] * 0.);
p2 = plot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output2_neurons_label, timesteps_for_firing_rate+1:end-1), 1), 'LineWidth', lineThickness-1);
set(p2, 'Color', [1 1 1] * 0.7);
plot(0:1:50, high, 'LineWidth', lineThickness, 'LineStyle', '-.', 'Color', [ 1 1 1 ] .* .5)
plot(0:1:50, low, 'LineWidth', lineThickness, 'LineStyle', '-.', 'Color', [ 1 1 1 ] .* .5)
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Mean firing rate (Hz)','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);
ylim([0, 35]);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Mean_firing_rate_out');


figure(3);
plot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(input1_neurons_label, timesteps_for_firing_rate+1:end-1), 1), 'LineWidth', lineThickness, 'Color', 'k')
hold on
plot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(input2_neurons_label, timesteps_for_firing_rate+1:end-1), 1), 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.7)
plot(0:1:50, high, 'LineWidth', lineThickness-1, 'LineStyle', '-.', 'Color', 'k')
plot(0:1:50, low, 'LineWidth', lineThickness-1, 'LineStyle', '-.', 'Color', 'k')
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Mean firing rate input (Hz)','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Mean_firing_rate_in');


figure(4);
confplot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out1_mean(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out1_sd(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out1_sd(timesteps_for_firing_rate+1:end-1));
hold on
confplot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out2_mean(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out2_sd(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out2_sd(timesteps_for_firing_rate+1:end-1));
hold on
p1 = plot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out1_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p1, 'Color', [1 1 1] * 0.);
p2 = plot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out2_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p2, 'Color', [1 1 1] * 0.7);
axis([0, 50, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_f');


figure(5);
confplot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out2_mean(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out2_sd(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out2_sd(timesteps_for_firing_rate+1:end-1));
hold on
confplot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out1_mean(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out1_sd(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out1_sd(timesteps_for_firing_rate+1:end-1));
hold on
p1 = plot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out1_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p1, 'Color', [1 1 1] * 0.);
p2 = plot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out2_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p2, 'Color', [1 1 1] * 0.7);
axis([0, 50, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d');


figure(6);
confplot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out2_mean(timesteps_for_firing_rate+1:end-1), U_evolution_out2_sd(timesteps_for_firing_rate+1:end-1), U_evolution_out2_sd(timesteps_for_firing_rate+1:end-1));
hold on
confplot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out1_mean(timesteps_for_firing_rate+1:end-1), U_evolution_out1_sd(timesteps_for_firing_rate+1:end-1), U_evolution_out1_sd(timesteps_for_firing_rate+1:end-1));
hold on
p1 = plot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out1_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p1, 'Color', [1 1 1] * 0.);
p2 = plot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out2_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p2, 'Color', [1 1 1] * 0.7);
axis([0, 50, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U');


% figure(4);
% plot(t, U_out11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
%  
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_out1');
% 
% 
% figure(5);
% plot(t, tau_d_out11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_out1');
% 
% 
% figure(6);
% plot(t, tau_f_out11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_out1');
% 
% 
% figure(7);
% plot(t, U_out22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_out2');
% 
% 
% figure(8);
% plot(t, tau_d_out22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_out2');
% 
% 
% figure(9);
% plot(t, tau_f_out22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_out2');
% 
% 
% figure(10);
% plot(t, U_in11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_in12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_in1');
% 
% 
% figure(11);
% plot(t, tau_d_in11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_in12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_in1');
% 
% 
% figure(12);
% plot(t, tau_f_in11_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_in12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_in1');
% 
% 
% figure(13);
% plot(t, U_in22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_in21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_in2');
% 
% 
% figure(14);
% plot(t, tau_d_in22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_in21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_in2');
% 
% 
% figure(15);
% plot(t, tau_f_in22_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_in21_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_in2');
% 
% 
% figure(16);
% plot(t, U_out1_in1_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_in1_out1_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_inout1');
% 
% 
% figure(17);
% plot(t, tau_d_out1_in1_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_in1_out1_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_inout1');
% 
% 
% figure(18);
% plot(t, tau_f_out1_in1_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_in1_out1_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_inout1');
% 
% 
% figure(19);
% plot(t, U_out2_in2_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_in2_out2_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_inout2');
% 
% 
% figure(20);
% plot(t, tau_d_out2_in2_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_in2_out2_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_inout2');
% 
% 
% figure(21);
% plot(t, tau_f_out2_in2_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_in2_out2_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_inout2');
% 
% 
% figure(6);
% plot(t, w_evolution_out1_1(:,:), 'LineWidth', lineThickness);
% xlabel('t(s)','fontsize',axesFontSize)
% ylabel('w from inputs to output 1','fontsize',axesFontSize)
% set(gca,'fontsize',numericFontSize);
% 
% print(gcf, '-depsc2', '-loose', 'Full_w_in_out1'); % Print the figure in eps (first option) and uncropped (second object)
% writeFig300ppi(gcf, 'Full_w_in_out1');
%
% 
% figure(7);
% plot(t, w_evolution_out2_1(:,:), 'LineWidth', lineThickness);
% xlabel('t(s)','fontsize',axesFontSize)
% ylabel('w from inputs to output 8','fontsize',axesFontSize)
% set(gca,'fontsize',numericFontSize);
% 
% print(gcf, '-depsc2', '-loose', 'Full_w_in_out8'); % Print the figure in eps (first option) and uncropped (second object)
% writeFig300ppi(gcf, 'Full_w_in_out8');
%
%
% figure(22);
% plot(t, w_evolution_out12_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, w_evolution_out21_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_w_mean_out12');
% 
% 
% figure(23);
% plot(t, w_evolution_in12_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, w_evolution_in21_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_w_mean_in12');
% 
% 
% figure(24);
% plot(t, w_evolution_out1in1_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, w_evolution_in1out1_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_w_mean_inout1');
% 
% 
% figure(25);
% plot(t, w_evolution_out2in2_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, w_evolution_in2out2_mean, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_w_mean_inout2');


% figure(26);
% plot(t, s_evolution, 'LineWidth', lineThickness, 'Color', 'k');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for entire network','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_s');
% 
% 
% figure(27);
% plot(t, s_out_evolution, 'LineWidth', lineThickness, 'Color', 'k');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for non-input neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_s_noinput');


figure(28);
plot(t(timesteps_for_firing_rate+1:end-1), s_out1_evolution(timesteps_for_firing_rate+1:end-1), 'LineWidth', lineThickness, 'Color', 'k');
hold on
plot(t(timesteps_for_firing_rate+1:end-1), s_out2_evolution(timesteps_for_firing_rate+1:end-1), 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.7);
axis([0, 50, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Symmetry index s','fontsize',axesFontSize);
%lg = legend('high','low');
%set(lg, 'Position', [0.7,0.5,0.25,0.08])
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'S_output');


% figure(29);
% plot(t, s_in1_evolution, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, s_in2_evolution, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% hold on
% plot(t, s_in12_evolution, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-.');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_s_input_singles');
% 
% 
% figure(30);
% plot(t, s_out1_in1_evolution, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '-');
% hold on
% plot(t, s_out2_in2_evolution, 'LineWidth', lineThickness, 'Color', 'k', 'LineStyle', '--');
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s for output neurons','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_s_input_output');


figure(31);
imagesc(w_out1)
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_end_output1');


figure(32);
imagesc(w_out2)
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_end_output2');


figure(33);
imagesc(w_in1)
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_end_input1');


figure(34);
imagesc(w_in2)
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_end_input2');


% figure(35);
% plot(t, U_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, U_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_U_out_lateral_source');
% 
% 
% figure(36);
% plot(t, tau_d_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_d_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_d_out_lateral_source');
% 
% 
% figure(37);
% plot(t, tau_f_out21_evolution_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, tau_f_out12_evolution_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
% set(gca,'fontsize', numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_tau_f_out_lateral_source');
% 
% 
% figure(38);
% plot(t, w_evolution_out21_mean, 'LineWidth', lineThickness, 'Color', 'k');
% hold on
% plot(t, w_evolution_out12_mean, 'LineWidth', lineThickness, 'Color', [1 1 1] .* 0.6);
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('s','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Full_w_mean_out_lateral_source');


x = 0.02:0.04:1;
% tau_d mean+-mse values
temp = tau_d(output_neurons_label, output_neurons_label);
temp(logical(temp==0)) = [];
tau_d_out12_to_out12_mean_end = mean(temp);
tau_d_out12_to_out12_sd_end = std(temp)./sqrt(size(temp,2));
figure(39)
hist(temp,x)
p = findobj(gca,'Type','patch');
set(p,'FaceColor','k')
xlab = xlabel('Tau_d (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
%writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out_distribution');

temp = tau_d(output1_neurons_label, [output1_neurons_label, output2_neurons_label]);
temp(logical(temp==0)) = [];
tau_d_out12_to_out1_mean_end = mean(temp);
tau_d_out12_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(40)
%hist(temp,x)

temp = tau_d(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
tau_d_out1_to_out1_mean_end = mean(temp);
tau_d_out1_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(41)
[n1, x1] = hist(temp,x);

temp = tau_d(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
tau_d_out2_to_out1_mean_end = mean(temp);
tau_d_out2_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(42)
[n2, x2] = hist(temp,x);

temp = tau_d(output2_neurons_label, [output2_neurons_label, output1_neurons_label]);
temp(logical(temp==0)) = [];
tau_d_out21_to_out2_mean_end = mean(temp);
tau_d_out21_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(43)
%hist(temp,25)

temp = tau_d(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
tau_d_out2_to_out2_mean_end = mean(temp);
tau_d_out2_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(44)
[n3, x3] = hist(temp,x);

temp = tau_d(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
tau_d_out1_to_out2_mean_end = mean(temp);
tau_d_out1_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(45)
[n4, x4] = hist(temp,x);

figure(46)
n_tot = [n3',n4',n1',n2'];
h = bar3(x,n_tot);
set(h(1),'Facecolor', [1 1 1] .* .9)
set(h(2),'Facecolor', [1 1 1] .* .6)
set(h(3),'Facecolor', [1 1 1] .* .3)
set(h(4),'Facecolor', [1 1 1] .* .0)
ylab = ylabel('Tau_d (s)','fontsize',axesFontSize);
zlab = zlabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
legend('2 --> 2','1 --> 2','1 --> 1','2 --> 1', 'location', [0.5 0.5 0.25 0.15])
writePDF1000ppi(gcf, numericFontSize, axesFontSize, ylab, zlab, 'Tau_d_out_distribution');


% tau_f mean+-mse values
temp = tau_f(output_neurons_label, output_neurons_label);
temp(logical(temp==0)) = [];
tau_f_out12_to_out12_mean_end = mean(temp);
tau_f_out12_to_out12_sd_end = std(temp)./sqrt(size(temp,2));
figure(47)
hist(temp,x)
p = findobj(gca,'Type','patch');
set(p,'FaceColor','k')
xlab = xlabel('Tau_f (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
%writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_f_out_distribution');

temp = tau_f(output1_neurons_label, [output1_neurons_label, output2_neurons_label]);
temp(logical(temp==0)) = [];
tau_f_out12_to_out1_mean_end = mean(temp);
tau_f_out12_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(48)
%hist(temp,x)

temp = tau_f(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
tau_f_out1_to_out1_mean_end = mean(temp);
tau_f_out1_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(49)
[n1, x1] = hist(temp,x);

temp = tau_f(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
tau_f_out2_to_out1_mean_end = mean(temp);
tau_f_out2_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(50)
[n2, x2] = hist(temp,x);

temp = tau_f(output2_neurons_label, [output2_neurons_label, output1_neurons_label]);
temp(logical(temp==0)) = [];
tau_f_out21_to_out2_mean_end = mean(temp);
tau_f_out21_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(51)
%hist(temp,x)

temp = tau_f(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
tau_f_out2_to_out2_mean_end = mean(temp);
tau_f_out2_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(52)
[n3, x3] = hist(temp,x);

temp = tau_f(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
tau_f_out1_to_out2_mean_end = mean(temp);
tau_f_out1_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(53)
[n4, x4] = hist(temp,x);

figure(54)
n_tot = [n3',n4',n1',n2'];
h = bar3(x,n_tot);
set(h(1),'Facecolor', [1 1 1] .* .9)
set(h(2),'Facecolor', [1 1 1] .* .6)
set(h(3),'Facecolor', [1 1 1] .* .3)
set(h(4),'Facecolor', [1 1 1] .* .0)
ylab = ylabel('Tau_f (s)','fontsize',axesFontSize);
zlab = zlabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
legend('2 --> 2','1 --> 2','1 --> 1','2 --> 1', 'location', [0.35 0.65 0.25 0.15])
writePDF1000ppi(gcf, numericFontSize, axesFontSize, ylab, zlab, 'Tau_f_out_distribution');


% U mean+-mse values
temp = U(output_neurons_label, output_neurons_label);
temp(logical(temp==0)) = [];
U_out12_to_out12_mean_end = mean(temp);
U_out12_to_out12_sd_end = std(temp)./sqrt(size(temp,2));
figure(55)
hist(temp,x)
p = findobj(gca,'Type','patch');
set(p,'FaceColor','k')
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
%writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out_distribution');

temp = U(output1_neurons_label, [output1_neurons_label, output2_neurons_label]);
temp(logical(temp==0)) = [];
U_out12_to_out1_mean_end = mean(temp);
U_out12_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(56)
%hist(temp,x);

temp = U(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
U_out1_to_out1_mean_end = mean(temp);
U_out1_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(57)
[n1, x1] = hist(temp,x);

temp = U(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
U_out2_to_out1_mean_end = mean(temp);
U_out2_to_out1_sd_end = std(temp)./sqrt(size(temp,2));
%figure(58)
[n2, x2] = hist(temp,x);

temp = U(output2_neurons_label, [output2_neurons_label, output1_neurons_label]);
temp(logical(temp==0)) = [];
U_out21_to_out2_mean_end = mean(temp);
U_out21_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(59)
%hist(temp,x);

temp = U(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
U_out2_to_out2_mean_end = mean(temp);
U_out2_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(60)
[n3, x3] = hist(temp,x);

temp = U(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
U_out1_to_out2_mean_end = mean(temp);
U_out1_to_out2_sd_end = std(temp)./sqrt(size(temp,2));
%figure(61)
[n4, x4] = hist(temp,x);

figure(62)
n_tot = [n3',n4',n1',n2'];
h = bar3(x,n_tot);
set(h(1),'Facecolor', [1 1 1] .* .9)
set(h(2),'Facecolor', [1 1 1] .* .6)
set(h(3),'Facecolor', [1 1 1] .* .3)
set(h(4),'Facecolor', [1 1 1] .* .0)
ylab = ylabel('U','fontsize',axesFontSize);
zlab = zlabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
legend('2 --> 2','1 --> 2','1 --> 1','2 --> 1', 'location', [0.5 0.5 0.25 0.15])
writePDF1000ppi(gcf, numericFontSize, axesFontSize, ylab, zlab, 'U_out_distribution');


temp = tau_d_init(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
tau_d_out1_to_out1_mean_init = mean(temp);

temp = tau_d_init(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
tau_d_out2_to_out1_mean_init = mean(temp);

temp = tau_d_init(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
tau_d_out2_to_out2_mean_init = mean(temp);

temp = tau_d_init(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
tau_d_out1_to_out2_mean_init = mean(temp);


temp = tau_f_init(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
tau_f_out1_to_out1_mean_init = mean(temp);

temp = tau_f_init(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
tau_f_out2_to_out1_mean_init = mean(temp);

temp = tau_f_init(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
tau_f_out2_to_out2_mean_init = mean(temp);

temp = tau_f_init(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
tau_f_out1_to_out2_mean_init = mean(temp);


temp = U_init(output1_neurons_label, output1_neurons_label);
temp(logical(temp==0)) = [];
U_out1_to_out1_mean_init = mean(temp);

temp = U_init(output1_neurons_label, output2_neurons_label);
temp = reshape(temp, 1, (N_outputs1*N_outputs2));
U_out2_to_out1_mean_init = mean(temp);

temp = U_init(output2_neurons_label, output2_neurons_label);
temp(logical(temp==0)) = [];
U_out2_to_out2_mean_init = mean(temp);

temp = U_init(output2_neurons_label, output1_neurons_label);
temp = reshape(temp, 1, (N_outputs2*N_outputs1));
U_out1_to_out2_mean_init = mean(temp);

TM_single_synapse(tau_f_out2_to_out2_mean_end, tau_d_out2_to_out2_mean_end, U_out2_to_out2_mean_end, 12, 3.5, 0.8, 3, 'TM_trace_post_out2_to_out2_end', 'TM_trace_pre_out2_to_out2_end')
TM_single_synapse(tau_f_out1_to_out2_mean_end, tau_d_out1_to_out2_mean_end, U_out1_to_out2_mean_end, 12, 3.5, 0.8, 3, 'TM_trace_post_out1_to_out2_end', 'TM_trace_pre_out1_to_out2_end')
TM_single_synapse(tau_f_out1_to_out1_mean_end, tau_d_out1_to_out1_mean_end, U_out1_to_out1_mean_end, 12, 3.5, 0.8, 3, 'TM_trace_post_out1_to_out1_end', 'TM_trace_pre_out1_to_out1_end')
TM_single_synapse(tau_f_out2_to_out1_mean_end, tau_d_out2_to_out1_mean_end, U_out2_to_out1_mean_end, 12, 3.5, 0.8, 3, 'TM_trace_post_out2_to_out1_end', 'TM_trace_pre_out2_to_out1_end')

%save Data_Learning_STP_two_out_pop

