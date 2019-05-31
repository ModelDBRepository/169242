
%% This function uses the Euler method to evaluate the membrane potential Vf of N neurons after the time dt and for a given vector of initial potential Vi

function Vf = Euler_integration_conductance_based_IF_multi_synapses( Vi, E, g, tau_g, g_L, dt, N)

h = 100;        %number of integration steps

dt_h = dt / h;  %corrsponding value of time

Vh = Vi;        %initialization of the membrane potential (column vector)
gh = g;         %initialization of the conductances (squared matrix)

i = 1;
while (i <= h)
            
    Vh = Vh + dt_h * ( -g_L * Vh + diag ( gh * (repmat( E - Vh', N, 1 ) )) ) ;
    gh = gh .* exp( -dt_h / tau_g);
            
    i = i + 1;    
end

Vf = Vh;