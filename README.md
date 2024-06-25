# multistrain-clustering
Source code for the paper: On the Interplay of Clustering and Evolution in the Emergence of Epidemic Outbreaks

compatible with python3

### simEvolution_clustered_network.py
Simulation code for simulating multi-strain spreading on a random clustered network described by the configuration model. Generate a social network with parameters according to the parameters given below and initiate simulation of multi-strain spreading with two strains with given transmissibility and mutation probabilities.

Run `python simEvolution_clustered_network.py -[parameter] value` to run the simulation.

### Available parameters to change:
- n - number of nodes 
- e - number of experiments
- t1 - transmission rate of strain 1
- t2 - transmission rate of strain 2
- m1 - the probability that a host infection with strain-1 does not lead to a mutation, i.e., \mu_{11}
- m2 - the probability that a host infection with strain-2 does not lead to a mutation, i.e., \mu_{22}
- start_strain - the strain that the seed node is infected with
- meanDegSingle - mean degree for single-edges
- meanDegTriangle - mean degree for triangle-edges
- thrVal - threshold value to define a giant component for finite node simulations
- numCores - number of cores to use
- logName - name of the log file

### Steps to reproduce the results of Fig. 3:

Run `python simEvolution_clustered_network.py -m 0`

Run `python simEvolution_clustered_network.py -m 0.5`

...

...

...

Run `python simEvolution_clustered_network.py -m 7`

This will yield the probabilities of emergence and the epidemic size of each strain for each value of the \lambda parameter. 
This same code can be used to reproduce the simulations in Figures 5 and 6 by setting the values (t1,t2,m1,m2) appropriately.

### Steps to reproduce the results of Fig. 4:

