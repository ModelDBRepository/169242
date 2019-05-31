
%% Recurrent network of N neurons learns to optimize synapses by learning STP and STDP through an error-driven mechanism

% This code calls the functions:
    % 1. Euler_integration_conductance_based_IF_multi_synapses - neuron integration
    % 2. sym_measure - symmetry measure evaluation
    % 3. confplot - shaded plot
    % 4. writePDF1000ppi - print 1000ppi .pdf
    % 5. TM_single_synapse - compute synaptic traces from TM model (and calls the function Euler_integration_IF_with_external_current)
    
clear all
close all

%% General parameters

N_inputs = 30;
N_outputs = 10;

input_output_max_intersection = 0;

N = N_inputs + N_outputs;    %number of neurons

high = 30;  %Hz
low = 5;    %Hz

temporal = 1;   %switching regime variable


%% Computational parameters
dt_simul = 1e-3;    %s

time_simul_single_regime = 1e2;     %s
n_single_regime = (round(time_simul_single_regime / dt_simul) );

n_regimes = 4;
n_sample = n_regimes * n_single_regime + 1;
time_simul = n_regimes * time_simul_single_regime;

iter_end_phases = [1, (time_simul_single_regime : time_simul_single_regime : n_regimes*time_simul_single_regime) ./ dt_simul ];
freq_end_phases = [low, low, low, low, low];
time_simul_traces = [2.7, 2.7, 2.7, 2.7, 2.7];  %s
time_end_train = [1.25, 1.25, 1.25, 1.25, 1.25];  %s
time_extra_spike = [2.5, 2.5, 2.5, 2.5, 2.5];  %s
figure_name_post = char('TM_trace_post_init', 'TM_trace_post_end1', 'TM_trace_post_end2', 'TM_trace_post_end3', 'TM_trace_post_end4');
figure_name_pre = char('TM_trace_pre_init', 'TM_trace_pre_end1', 'TM_trace_pre_end2', 'TM_trace_pre_end3', 'TM_trace_pre_end4');

timesteps_for_firing_rate = 500;
exp_decay_mean_firing_rate = 0.001;


%% Plotting parameters
numericFontSize = 25;
axesFontSize = 30;
lineThickness = 2;
markLine = 1;
markSize = 12;


%% Neuron Parameters
%from Carvalho Buonomano
E = 3e-2;           %V
g_L = 0.1e-6;       %S
tau_g = 1e-2;       %s
V_th = 1e-3;        %V

refr_period = 10e-3;    %s ---> maximum frequency: 100Hz
freq_max = 1 / refr_period;


%% STP Parameters
tau_f_lower = 1e-3;
tau_f_upper = 900e-3;
tau_f = tau_f_lower + (tau_f_upper-tau_f_lower) * rand(N, N);
tau_f = tau_f - diag(diag(tau_f));

tau_d_lower = 100e-3;
tau_d_upper = 900e-3;
tau_d = tau_d_lower + (tau_d_upper-tau_d_lower) * rand(N, N);
tau_d = tau_d - diag(diag(tau_d));

U_first_lower = 0.05;
U_first_upper = 0.95;
U_first = U_first_lower + (U_first_upper-U_first_lower) * rand(N, N);
U_first = U_first - diag(diag(U_first));
U_lower = 0.05;
U_upper = 0.95;


%% STDP Learning parameters
% Triplet model - From Pfister & Gerstner: Triplets in STDP models. Visual cortex nn data set
taupos  = 16.8 * 1E-3;      %time constant for r1 in s
tauneg  = 33.7 * 1E-3;      %time constant for o1 in s
taux    = 575 * 1E-3;       %time constant for r2 in s
tauy    = 47 * 1E-3;        %time constant for o2 in s
A2pos   = 4.6E-3;           %amplitude of weight change for LTP (pair interaction)
A3pos   = 9.1E-3;           %amplitude of weight change for LTP (triplet interaction)
A2neg   = 3.0E-3;           %amplitude of weight change for LTD (pair interaction)
A3neg   = 7.5E-9;           %amplitude of weight change for LTD (triplet interaction)

learning_rate = 4;          %value for the first regime (temporal)

w_max = 1;          %maximum synaptic weight
w_min = 0.001;      %minimum synaptic weight


%% Input neurons selection
neurons_label = 1 : N;
input_neurons_logical = zeros(N, 1);

temp = randperm(N); %permuting indices

input_neurons_label = sort(temp(1 : N_inputs));
input_neurons_logical(input_neurons_label) = 1; %identifying neurons in the input pull


%% Output neurons selection
output_neurons_logical = zeros(N, 1);

temp = input_neurons_label;
temp(1:round(N_inputs*(input_output_max_intersection))) = [];   %neurons that are in the input pool only so they have to excluded from the possible output pull

temp2 = neurons_label;
temp2(temp) = [];   %excluding neurons that are in the input pull only

perm_index = randperm(size(temp2, 2));  %permuting indices
temp = temp2(perm_index);

output_neurons_label = sort(temp(1 : N_outputs));   %identifying neurons in the output pull
output_neurons_logical(output_neurons_label) = 1;


%% Input signal

%---Temporal code. Neurons receive external input and fire in a precise sequence with the same rate
rho_input = 10;                                 % Stimulus rate in Hz
delay = 1 / (rho_input * N_inputs);             % Time delay between the stimulus injection for two consecutive neurons in s 
input_time_1 =  100 * 1E-3;                     % Stimulus injection time for neuron 1. Dynamics begins after 100ms to be sure that adding gaussian noise does not give negative time
jitter_amplitude = 0.1 * delay;                 % 10% of the delay
next_input_times = input_time_1 + (0:(N_inputs-1))' * delay + ( jitter_amplitude * randn(N_inputs, 1) ); % Gaussian noise with sd = (10% of the delay) --> The inversion in the order of input injection between two consecutive neurons is a very unlikely event

if temporal == 1    
    target_firing_rate = low;
else  
    target_firing_rate = high;   
end

dV_input = 2 * V_th;


%% Variables
t = 0 : dt_simul : time_simul;
t = round(1000.*t)./1000; %this approximates the time steps to the order of ms (the sampling time is 1 ms)

V = zeros(N, n_sample);
time_from_last_spike = zeros(N, 1);
neurons_spike_logical = zeros(N, timesteps_for_firing_rate);

firing_rates = zeros(N, floor(n_sample));
mean_firing_rate = zeros(N, floor(n_sample));
error = zeros(N, floor(n_sample));

g = zeros(N, N);
F = zeros(N, N);
D = zeros(N, N); 
U = zeros(N, N);

o1 = zeros(N,1);    % Post synaptic variables. Note: its value depends only on the activity of the post (the pre triggers only, without changing) ---> We need the same number as neurons, not synapses
r1 = zeros(N,1);    % Pre synaptic variables. Note: the same argument above holds here.
o2 = zeros(N,1);    % Post synaptic variables. 
r2 = zeros(N,1);    % Pre synaptic variables. 

w_evolution_1 = zeros(N_inputs, floor(n_sample));
%w_evolution_2 = zeros(N_inputs, floor(n_sample));
%w_evolution_3 = zeros(N_inputs, floor(n_sample));
%w_evolution_4 = zeros(N_inputs, floor(n_sample));
%w_evolution_5 = zeros(N_inputs, floor(n_sample));
%w_evolution_6 = zeros(N_inputs, floor(n_sample));
%w_evolution_7 = zeros(N_inputs, floor(n_sample));
w_evolution_8 = zeros(N_inputs, floor(n_sample));
%w_evolution_9 = zeros(N_inputs, floor(n_sample));
%w_evolution_10 = zeros(N_inputs, floor(n_sample));

w_out = cell(1, 4);  % In each element of the cell store connectivity matrix of output neurons at the end of each regime
s_out = cell(1, 4);  % In each element of the cell store corresponding symmetry value
p_s_out = cell(1, 4);  % In each element of the cell store corresponding p-value

s_evolution = zeros(1, floor(n_sample));
s_out_evolution = zeros(1, floor(n_sample));

%output population
tau_f_evolution_out = zeros( N_outputs, floor(n_sample));
tau_d_evolution_out = zeros( N_outputs, floor(n_sample));
U_evolution_out = zeros( N_outputs, floor(n_sample));

%entire network
tau_f_evolution_out_mean = zeros( 1, floor(n_sample));
tau_f_evolution_out_sd = zeros( 1, floor(n_sample));
tau_d_evolution_out_mean = zeros( 1, floor(n_sample));
tau_d_evolution_out_sd = zeros( 1, floor(n_sample));
U_evolution_out_mean = zeros( 1, floor(n_sample));
U_evolution_out_sd = zeros( 1, floor(n_sample));

%synaptic values of the output at the beginning and at the end of each phase
U_output_phase_end = cell(1, 5);
tau_d_output_phase_end = cell(1, 5);


%% Initial state
V(:, 1) = 0;

time_from_last_spike(:) = refr_period * 1e+3;  %renormalisation in ms units
neurons_previously_fired_logical = (V(:, 1) >= V_th);

F(:, :) = U_first;
F = F - diag(diag(F));

D(:, :) = 1;
D = D - diag(diag(D));

U(:, :) = U_first;
U = U - diag(diag(U));

w_in = (w_max - w_min) * rand(N, N);   %initialization of the weights
w_in = w_in - diag(diag(w_in));        %eliminate self-interaction

w = w_in;

w_evolution_1(:, 1) = w_in(output_neurons_label(1), input_neurons_label)';
%w_evolution_2(:, 1) = w_in(output_neurons_label(2), input_neurons_label)';
%w_evolution_3(:, 1) = w_in(output_neurons_label(3), input_neurons_label)';
%w_evolution_4(:, 1) = w_in(output_neurons_label(4), input_neurons_label)';
%w_evolution_5(:, 1) = w_in(output_neurons_label(5), input_neurons_label)';
%w_evolution_6(:, 1) = w_in(output_neurons_label(6), input_neurons_label)';
%w_evolution_7(:, 1) = w_in(output_neurons_label(7), input_neurons_label)';
w_evolution_8(:, 1) = w_in(output_neurons_label(8), input_neurons_label)';
%w_evolution_9(:, 1) = w_in(output_neurons_label(9), input_neurons_label)';
%w_evolution_10(:, 1) = w_in(output_neurons_label(10), input_neurons_label)';

s_evolution(1) = sym_measure(w);
s_out_evolution(1) = sym_measure(w(output_neurons_label, output_neurons_label));

temp = tau_f(output_neurons_label, :)';
temp(logical(temp==0)) = [];
tau_f_evolution_out(:, 1) = mean(reshape(temp, N-1, N_outputs))';
tau_f_evolution_out_mean(1) = mean(tau_f_evolution_out(:, 1));
tau_f_evolution_out_sd(1) = std(tau_f_evolution_out(:, 1));

temp = tau_d(output_neurons_label, :)';
temp(logical(temp==0)) = [];
tau_d_evolution_out(:, 1) = mean(reshape(temp, N-1, N_outputs))';
tau_d_evolution_out_mean(1) = mean(tau_d_evolution_out(:, 1));
tau_d_evolution_out_sd(1) = std(tau_d_evolution_out(:, 1));

temp = U(output_neurons_label, :)';
temp(logical(temp==0)) = [];
U_evolution_out(:, 1) = mean(reshape(temp, N-1, N_outputs))';
U_evolution_out_mean(1) = mean(U_evolution_out(:, 1));
U_evolution_out_sd(1) = std(U_evolution_out(:, 1));

temp = U(output_neurons_label,:);
temp(logical(temp==0)) = [];
U_output_phase_end{1} = temp;
        
temp = tau_d(output_neurons_label,:);
temp(logical(temp==0)) = [];
tau_d_output_phase_end{1} = temp;

        
%% Learning
regime_counter = 1;
for i = 2 : n_sample
    
    if i == (regime_counter * n_single_regime) + 1
        temporal = ~temporal;
                
        target_firing_rate = low * temporal + high * (~temporal);
        
        learning_rate =  2 * temporal + 1 * (~temporal);
        
        w_out{regime_counter} = w(output_neurons_label, output_neurons_label);
        s_out{regime_counter} = sym_measure(w(output_neurons_label, output_neurons_label));
        
        temp = U(output_neurons_label,:);
        temp(logical(temp==0)) = [];
        U_output_phase_end{regime_counter+1} = temp;
        
        temp = tau_d(output_neurons_label,:);
        temp(logical(temp==0)) = [];
        tau_d_output_phase_end{regime_counter+1} = temp;
        
        regime_counter = regime_counter + 1;
            
    end
    
    if regime_counter > n_regimes
        break;
    end
    
    if i > timesteps_for_firing_rate
        neurons_spike_logical = circshift(neurons_spike_logical, [0, -1]);
        neurons_spike_logical(:, timesteps_for_firing_rate) = 0;
    end
    
    % compute the voltage of neurons
    V(:, i) = Euler_integration_conductance_based_IF_multi_synapses( V(:,i-1), E, g, tau_g, g_L, dt_simul, N);        %exponential decay of the membrane potential    
         
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
    w = w .* (w >= w_min) + w_min * (w < w_min);    % Lower bound
    w = w .* (w <= w_max) + w_max * (w > w_max);    % Upper bound
    
    w = w - diag(diag(w));        %eliminate self-interaction
    
    % Symmetry measure
    s_evolution(i) = sym_measure(w);
    s_out_evolution(i) = sym_measure(w(output_neurons_label, output_neurons_label));
    
    % Storing a single neuron's weights
    w_evolution_1(:, i) = w(output_neurons_label(1), input_neurons_label)';
    %w_evolution_2(:, i) = w(output_neurons_label(2), input_neurons_label)';
    %w_evolution_3(:, i) = w(output_neurons_label(3), input_neurons_label)';
    %w_evolution_4(:, i) = w(output_neurons_label(4), input_neurons_label)';
    %w_evolution_5(:, i) = w(output_neurons_label(5), input_neurons_label)';
    %w_evolution_6(:, i) = w(output_neurons_label(6), input_neurons_label)';
    %w_evolution_7(:, i) = w(output_neurons_label(7), input_neurons_label)';
    w_evolution_8(:, i) = w(output_neurons_label(8), input_neurons_label)';
    %w_evolution_9(:, i) = w(output_neurons_label(9), input_neurons_label)';
    %w_evolution_10(:, i) = w(output_neurons_label(10), input_neurons_label)';
        
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
                
        mean_firing_rate(output_neurons_label, i) = mean(firing_rates(output_neurons_label, i));        
        mean_firing_rate(input_neurons_label, i) = mean(firing_rates(input_neurons_label, i));
        
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
        %tau_f = tau_f + 0.2 * (1 + ((repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N )).^2) .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );        
        
        % STP rule for synaptic strength
        %w = w + learning_rate * ( 1 ./ (abs(tau_d_squared)) ) .* ( (repmat(error(:, i), 1, N) ) .* repmat( neurons_currently_firing_logical, 1, N ) ./ (freq_max^2) );  
        
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
        
        % store STP parameters
        temp = tau_f(output_neurons_label, :)';
        temp(logical(temp==0)) = [];
        tau_f_evolution_out(:, i) = mean(reshape(temp, N-1, N_outputs))';
        tau_f_evolution_out_mean(i) = mean(tau_f_evolution_out(:, i));
        tau_f_evolution_out_sd(i) = std(tau_f_evolution_out(:, i));

        temp = tau_d(output_neurons_label, :)';
        temp(logical(temp==0)) = [];
        tau_d_evolution_out(:, i) = mean(reshape(temp, N-1, N_outputs))';
        tau_d_evolution_out_mean(i) = mean(tau_d_evolution_out(:, i));
        tau_d_evolution_out_sd(i) = std(tau_d_evolution_out(:, i));

        temp = U(output_neurons_label, :)';
        temp(logical(temp==0)) = [];
        U_evolution_out(:, i) = mean(reshape(temp, N-1, N_outputs))';
        U_evolution_out_mean(i) = mean(U_evolution_out(:, i));
        U_evolution_out_sd(i) = std(U_evolution_out(:, i));
                 
    end    
               
    display(i)
    
end

s = sym_measure(w(output_neurons_label, output_neurons_label));

syms u
theoretical_mean_unif_noprune = 0.6137;
theoretical_variance_unif_noprune = 0.0017;
f = (1 / sqrt(2*pi*theoretical_variance_unif_noprune)) * exp( - (u - theoretical_mean_unif_noprune)^2 / (2*theoretical_variance_unif_noprune) );

p_s_out{1} = 2 * ( double(int(f, 0, s_out{1})) * (s_out{1} < theoretical_mean_unif_noprune) + double(int(f, s_out{1}, 1)) * (s_out{1} >= theoretical_mean_unif_noprune) );
p_s_out{2} = 2 * ( double(int(f, 0, s_out{2})) * (s_out{2} < theoretical_mean_unif_noprune) + double(int(f, s_out{2}, 1)) * (s_out{2} >= theoretical_mean_unif_noprune) );
p_s_out{3} = 2 * ( double(int(f, 0, s_out{3})) * (s_out{3} < theoretical_mean_unif_noprune) + double(int(f, s_out{3}, 1)) * (s_out{3} >= theoretical_mean_unif_noprune) );
p_s_out{4} = 2 * ( double(int(f, 0, s_out{4})) * (s_out{4} < theoretical_mean_unif_noprune) + double(int(f, s_out{4}, 1)) * (s_out{4} >= theoretical_mean_unif_noprune) );


%% TM model traces production
for ind = 1:length(iter_end_phases)
    TM_single_synapse(tau_f_evolution_out_mean(iter_end_phases(ind)), tau_d_evolution_out_mean(iter_end_phases(ind)), U_evolution_out_mean(iter_end_phases(ind)), freq_end_phases(ind), time_simul_traces(ind), time_end_train(ind), time_extra_spike(ind), figure_name_post(ind, find(figure_name_post(ind,:) ~= ' ')), figure_name_pre(ind, find(figure_name_pre(ind,:) ~= ' ')))
end

%% Save, Print and plot

sprintf('Connectivity data:\n')
sprintf('Symmetry measure over the subnet of %d output neurons at different times:\nt=%f s=%f\nt=%f s=%f\nt=%f s=%f\nt=%f s=%f\n', N_outputs, time_simul_single_regime, s_out{1}, 2*time_simul_single_regime, s_out{2}, 3*time_simul_single_regime, s_out{3}, 4*time_simul_single_regime, s_out{4})
sprintf('Corresponding p-value: %f\n%f\n%f\n%f\n', p_s_out{1}, p_s_out{2}, p_s_out{3}, p_s_out{4})

figure(1);
plot(t(timesteps_for_firing_rate+1:end), mean(abs(error(output_neurons_label,timesteps_for_firing_rate+1:end))), 'LineWidth', lineThickness, 'Color', 'k');
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Mean error (Hz)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Error');


figure(2);
confplot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output_neurons_label, timesteps_for_firing_rate+1:end-1), 1), std(firing_rates(output_neurons_label, timesteps_for_firing_rate+1:end-1), 1) );
hold on
plot(t(timesteps_for_firing_rate+1:end-1), mean(firing_rates(output_neurons_label, timesteps_for_firing_rate+1:end-1), 1), 'LineWidth', lineThickness-1, 'Color', 'k')
plot(0:5:400, high, 'LineWidth', lineThickness, 'LineStyle', '-.', 'Color', [ 1 1 1 ] .* .5)
plot(0:5:400, low, 'LineWidth', lineThickness, 'LineStyle', '-.', 'Color', [ 1 1 1 ] .* .5)
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Mean firing rate (Hz)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Mean_firing_rate');


% figure(3);
% plot(t, firing_rates(output_neurons_label, :), 'LineWidth', lineThickness)
% xlab = xlabel('t (s)','fontsize',axesFontSize);
% ylab = ylabel('Output firing rates (Hz)','fontsize',axesFontSize);
% set(gca,'fontsize',numericFontSize);
% 
% writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Firing_rates');


figure(4);
confplot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out_mean(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out_sd(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out_sd(timesteps_for_firing_rate+1:end-1));
hold on
p = plot(t(timesteps_for_firing_rate+1:end-1), tau_f_evolution_out_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p, 'Color', [1 1 1] * 0.);
axis([0, 400, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Facilitation constant tau_f (s)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_f');


figure(5);
confplot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out_mean(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out_sd(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out_sd(timesteps_for_firing_rate+1:end-1));
hold on
p = plot(t(timesteps_for_firing_rate+1:end-1), tau_d_evolution_out_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p, 'Color', [1 1 1] * 0.);
axis([0, 400, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Recovery constant tau_d (s)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d');


figure(6);
confplot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out_mean(timesteps_for_firing_rate+1:end-1), U_evolution_out_sd(timesteps_for_firing_rate+1:end-1), U_evolution_out_sd(timesteps_for_firing_rate+1:end-1));
hold on
p = plot(t(timesteps_for_firing_rate+1:end-1), U_evolution_out_mean(timesteps_for_firing_rate+1:end-1),'-','LineWidth',lineThickness);
set(p, 'Color', [1 1 1] * 0.);
axis([0, 400, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Synaptic utilization U','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U');


% figure(7);
% plot(t, w_evolution_1(:, :), 'LineWidth', lineThickness, 'Color', 'k');
% xlabel('t(s)','fontsize',axesFontSize)
% ylabel('w from inputs to output 1','fontsize',axesFontSize)
% set(gca,'fontsize',numericFontSize);
% 
% print(gcf, '-depsc2', '-loose', 'Full_w_in_out1'); % Print the figure in eps (first option) and uncropped (second object)
% writeFig300ppi(gcf, 'Full_w_in_out1');
%
% 
% figure(8);
% plot(t, w_evolution_8(:, :), 'LineWidth', lineThickness, 'Color', 'k');
% xlabel('t(s)','fontsize',axesFontSize)
% ylabel('w from inputs to output 8','fontsize',axesFontSize)
% set(gca,'fontsize',numericFontSize);
% 
% print(gcf, '-depsc2', '-loose', 'Full_w_in_out8'); % Print the figure in eps (first option) and uncropped (second object)
% writeFig300ppi(gcf, 'Full_w_in_out8');


figure(9);
plot(t(timesteps_for_firing_rate+1:end-1), s_evolution(timesteps_for_firing_rate+1:end-1), 'LineWidth', lineThickness, 'Color', 'k');
axis([0, 400, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('s for entire network','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'S_tot');


figure(10);
plot(t(timesteps_for_firing_rate+1:end-1), s_out_evolution(timesteps_for_firing_rate+1:end-1), 'LineWidth', lineThickness, 'Color', 'k');
axis([0, 400, 0, 1]);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Symmetry index s','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'S_output');


figure(11);
imagesc(w_out{1})
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'End_regime_1');


figure(12);
imagesc(w_out{2})
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'End_regime_2');


figure(13);
imagesc(w_out{3})
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);
set(gca,'fontsize', numericFontSize);
axis square
colormap(gray)
colorbar

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'End_regime_3');


figure(14);
imagesc(w_out{4})
axis square
colormap(gray)
colorbar
set(gca,'fontsize', numericFontSize);
xlab = xlabel('Output neurons','fontsize',axesFontSize);
ylab = ylabel('Output neurons','fontsize',axesFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'End_regime_4');


x = 0.02:0.04:1;
figure(15)
[n, x] = hist(tau_d_output_phase_end{1}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('tau_r_e_c (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out_init');


figure(16)
[n, x] = hist(tau_d_output_phase_end{2}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('tau_r_e_c (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out1');


figure(17)
[n, x] = hist(tau_d_output_phase_end{3}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('tau_r_e_c (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out2');


figure(18)
[n, x] = hist(tau_d_output_phase_end{4}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('tau_r_e_c (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out3');


figure(19)
[n, x] = hist(tau_d_output_phase_end{5}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('tau_r_e_c (s)','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'Tau_d_out4');


x = 0.02:0.04:1;
figure(20)
[n, x] = hist(U_output_phase_end{1}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
ylim([0 30])

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out_init');


figure(21)
[n, x] = hist(U_output_phase_end{2}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out1');


figure(22)
[n, x] = hist(U_output_phase_end{3}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out2');


figure(23)
[n, x] = hist(U_output_phase_end{4}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out3');


figure(24)
[n, x] = hist(U_output_phase_end{5}, x);
h = bar(x, n);
set(h,'Facecolor', [1 1 1] .* .0)
xlab = xlabel('U','fontsize',axesFontSize);
ylab = ylabel('Counts','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
ylim([0 250])

writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, 'U_out4');

%save Learning_STP_one_out_pop
