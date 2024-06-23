# multistrain-clustering
Source code for the paper: On the Interplay of Clustering and Evolution in the Emergence of Epidemic Outbreaks

compatible with python3

### simEvolution_generated_network.py
Simulation code with self-generated network. Generate a social network with parameters according to the input, with 2 strains.

Run `python simEvolution_generated_network.py -[parameter] value` to run the simulation.

### Available parameters:

- m - mean degree of the network 
- n - number of nodes 
- e - number of experiments
- t1 - transmission rate of strain 1
- t2 - transmission rate of strain 2
- m1 - mutation rate of strain 1
- m2 - mutation rate of strain 2
- thrVal - threshold value to consider a component giant
- numCores - number of cores to use
- logName - name of the log file

### Steps to reproduce the results of Table 1 (Fig. 1):

Run `python simEvolution_generated_network.py -m 0`

Run `python simEvolution_generated_network.py -m 0.5`

...

...

...

Run `python simEvolution_generated_network.py -m 7`

_You'll get the probabilities of emergence and the epidemic size of each strain for each value of m._
