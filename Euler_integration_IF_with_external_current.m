
%% This function uses the Euler method to evaluate the membrane potential Vf after the time dt
%   1. for a given initial potential Vi
%   2. for a membrane with time constant tau_m and resting potential V_rest
%   3. with an external time-dependent current with initial value Ii and time constant decay tau_s

function Vf = Euler_integration_IF_with_external_current( Vi, V_rest, tau_m, Res, Ii, tau_s, dt)

h = 100;        %number of integration steps

dt_h = dt / h;  %corrsponding value of time

Vh = Vi;        %initialization of the membrane potential
Ih = Ii;        %initialization of the external current

i = 1;
while (i <= h)
    
    Vh = Vh + dt_h * ( ( -(Vh-V_rest) + Res * Ih ) / tau_m );
    Ih = Ih * exp( -dt_h / tau_s);
    
    i = i + 1;
end

Vf = Vh;