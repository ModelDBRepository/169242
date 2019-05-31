
This is the readme for the model associated with the paper

Esposito U, Giugliano M and Vasilaki E (2015)
Adaptation of short-term plasticity parameters via error-driven learning
may explain the correlation between activity-dependent synaptic properties,
connectivity motifs and target specificity. 
Front. Comput. Neurosci. 8:175. doi: 10.3389/fncom.2014.00175

This matlab code reproduces Figures 1B, 2, 3, 4B, 5 and 6 from the paper.
To run it you need to use Matlab with Symbolic Math Toolbox and Statistics Toolbox.

-----

The code is organized in 8 .m files:

1. Learning_STP_one_out_pop.m reproduces the graphs of Fig. 1A, 2 and 3 (single population).

2. Learning_STP_two_out_pop.m reproduces the graphs of Fig. 4B, 5 and 6 (double population).

3. Both scripts call: Euler_integration_conductance_based_IF_multi_synapses.m for neuron integration, 
   sym_measure.m for evaluation of the symmetry measure, 
   TM_single_synapse.m to evaluate the STP traces (figures 2 and 5),
   confplot.m and writePDF1000ppi.m for plotting.
4. TM_single_synapse.m calls Euler_integration_IF_with_external_current.m.

-----

How to use:

1. Single population case: to switch between the various models presented in the papaer,
   corresponding to different combination of the paramenters that has to be learnt, 
   comment or uncomment the lines 437,438,439,442 (defining the 4 learning rules).
   Example: to obtain Figures 1B and 2 (learning model with U and tau_d) comment out lines 
   439 (learning rule for tau_f) and 442 (learning rule for w).
   To obtain Figure 3 uncomment line 442 from the bove configuration.

2. Double population case: to switch between the various models presented in the papaer,
   corresponding to different combination of the paramenters that has to be learnt, 
   comment or uncomment the lines 924,925,926,929 (defining the 4 learning rules).
   Example: to obtain Figures 4B and 5 (full learning model) uncomment all the four lines.
   To obtain Figure 6 comment lines 924 and 926 from the bove configuration.