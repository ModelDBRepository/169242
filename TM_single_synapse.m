
function TM_single_synapse(tau_f, tau_d, U, freq_end_phases, time_simul, time_end_train, time_extra_spike, figure_name_post, figure_name_pre)

%% This function calls the script Euler_integration_IF_with_external_current for integration of the membrane potential

%% STP parameters
A = 1.54e-10;   %Amp    %maximum synaptic efficacy

%% Plotting parameters
numericFontSize = 25;
axesFontSize = 30;
lineThickness = 2;
markLine = 1;
markSize = 12;

%% Neuron parameters 
V_rest = 0e-3;  %V      %resting potential
Res = 1e9;      %Ohm    %membrane resistance
tau_m = 60e-3;  %s      %membrane time constant

tau_s = 25e-5;   %s      %synaptic current time costant

%% Simulation parameters
t_start = 50e-3;        %s
dt_simul = 1e-3;        %s
n_sample = round(time_simul / dt_simul) + 1;

%% Input spike train
dt_spike_train = 1 / freq_end_phases;
t_spike_pattern = [t_start : dt_spike_train : time_end_train, time_extra_spike, time_simul+1];    %the last spike is not relevant. It is needed to have the simulation running till the end without having an error for j reaching a value outside the size of t_spike_pattern
t_spike_pattern = round(1000.*t_spike_pattern)./1000; %this approximates the time of spike to the order of ms (the sampling time is 1 ms)
n_spikes = size(t_spike_pattern, 2);

%% Variables
t = 0 : dt_simul : time_simul;
t = round(1000.*t)./1000; %this approximates the time steps to the order of ms (the sampling time is 1 ms)
r = zeros(1, n_spikes); 
u = zeros(1, n_spikes);
a = zeros(1, n_spikes);
I = zeros(1, n_sample);
V = zeros(1, n_sample);
V_pre = zeros(1, n_sample);

%% Initial state
u(1) = U;
r(1) = 1;
a(1) = A * u(1) * r(1);
I(1) = 0;
V(1) = V_rest;

%% Dynamics
j = 1;
for i = 2 : n_sample
    
    I(i) = I(i-1) * exp(-dt_simul / tau_s);
    V(i) = Euler_integration_IF_with_external_current( V(i-1), V_rest, tau_m, Res, I(i-1), tau_s, dt_simul);        %exponential decay of the membrane potential
    V_pre(i) = 0;
    
    if t_spike_pattern(j) == t(i)
        V_pre(i) = 1;
        
        if j == 1            
            a(1) = A * u(1) * r(1);                                         %amplitude of the post synaptic current (EPSC) elicited by the incoming spike
        else
            dt_spike = t_spike_pattern(j) - t_spike_pattern(j-1);
            r(j) = 1 + (r(j-1) - r(j-1) * u(j-1) - 1) * exp(-dt_spike / tau_d);   %fraction of synaptic efficacy available immediately before the arrival of the spike
            u(j) = U + u(j-1) * (1 - U) * exp(-dt_spike / tau_f);                 %fraction of the available synaptic efficacy r that will be used by the arriving spike
            %u(j) = U;
            a(j) = A * u(j) * r(j);                                         %amplitude of the post synaptic current (EPSC) elicited by the incoming spike            
        end
        
        I(i) = I(i) + a(j);
        
        j = j + 1;
        
    end
       
end

%% Plots
figure(1);
plot(t, V, 'k', 'LineWidth', lineThickness-1);
xlab = xlabel('','fontsize',axesFontSize);
ylab = ylabel('Postsynaptic voltage (V)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
box off
set(gca, 'XTick', [])
set(gca,'XColor','w')
writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, figure_name_post);

figure(2);
plot(t, V_pre, 'k', 'LineWidth', lineThickness-1);
xlab = xlabel('t (s)','fontsize',axesFontSize);
ylab = ylabel('Postsynaptic voltage (V)','fontsize',axesFontSize);
set(gca,'fontsize',numericFontSize);
ylim([0 10])
axis off
writePDF1000ppi(gcf, numericFontSize, axesFontSize, xlab, ylab, figure_name_pre);
