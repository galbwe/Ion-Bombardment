# ion-bombardment
Code from my master's project meant to numerically approximate the solutions of a coupled system of partial differential equations which model the response of a binary solid to ion bombardment at normal incidence.
## Physics Background
The system of PDE's that we are interested in solving is a variant of the Bradley-Shipman equations:
|[BS_equations](https://latex.codecogs.com/gif.latex?%5Cbegin%7Baligned%7D%20u_t%20%26%3D%20%5Cphi%20-%20a%5Cnabla%5E2u%20-%20%5Cnabla%5E2%5Cnabla%5E2%20u%20&plus;%20%5Clambda%5Cvert%5Cnabla%20u%5Cvert%5E2%5C%5C%20%5Cphi_t%20%26%3D%20-%5Cphi%20&plus;%20b%5Cnabla%5E2%20u%20-%20c%5Cnabla%5E2%5Cphi%20-%20d%5Cnabla%5E2%5Cnabla%5E2%5Cphi%20&plus;%20%5Cnu%5Cphi%5E2%20&plus;%20%28%5Ceta%20&plus;%20%5Czeta%5Cgrad%5E2%29%5Cphi%5E3.%20%5Cend%7Baligned%7D)
