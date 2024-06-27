# multistrain-clustering
Source code for the paper: On the Interplay of Clustering and Evolution in the Emergence of Epidemic Outbreaks

# Installation Instructions
*The code is written in Python 3.9*

In terminal, run `python3 -m venv .venv` to create a virtual environment.

Run `source .venv/bin/activate` to activate the virtual environment.

Run `pip install -r requirements.txt` to install the required packages.

### simEvolution_clustered_network.py
Simulation code for simulating multi-strain spreading on a random clustered network described by the configuration model. Generate a social network with parameters according to the parameters given below and initiate simulation of multi-strain spreading with two strains with given transmission rate and mutation probabilities.

Run `python simEvolution_clustered_network.py -[parameter] value` to run the simulation.

### Available parameters to change
- n - number of nodes 
- e - number of experiments
- t1 - transmission rate of strain 1
- t2 - transmission rate of strain 2
- m1 - the probability that a host infection with strain-1 does not lead to a mutation, i.e., μ11
- m2 - the probability that a host infection with strain-2 does not lead to a mutation, i.e., μ22
- i - the strain that the seed node is infected with
- thrVal - threshold value to define a giant component for finite node simulations
- numCores - number of cores to use
- logName - name of the log file
---
### Additional parameters for 
simEvolution_clustered_network_with_c_lambda.py

*The code to explore the effect of clustering coefficient and λ value on epidemic outbreaks.*
- c - clustering coefficient
- l - λ value

### Steps to reproduce the results of Fig. 3:

Run `python simEvolution_clustered_network.py -m 0`

Run `python simEvolution_clustered_network.py -m 0.5`

...

...

...

Run `python simEvolution_clustered_network.py -m 7`

This will yield the probabilities of emergence and the epidemic size of each strain for each value of the λ parameter. 
This same code can be used to reproduce the simulations in Figures 4 and 5 by setting the values (t1,t2,m1,m2) appropriately.

### Steps to reproduce the results of Fig. 6:
Run `python simEvolution_clustered_network_with_c_lambda.py -c 0.01 -l 0.3`

Run `python simEvolution_clustered_network_with_c_lambda.py -c 0.01 -l 0.35`

...

...

...

Run `python simEvolution_clustered_network_with_c_lambda.py -c 0.01 -l 5.0`

...

Run `python simEvolution_clustered_network_with_c_lambda.py -c 3.99 -l 5.0`
