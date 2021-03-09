# Monkey-Mind-Reader
Recursive Bayesian Filter Based Neural Decoding Algorithm Estimating 2D Monkey Hand Trajectories Amira El Fekih, Gauri Gupta, Iani Gayo and Joanna-Svilena Haralampieva (The Spike Girls Team) Department of Bioengineering, Imperial College London, BMI 2020

The aim of the group project was to implement a neural
decoding algorithm estimating precise hand trajectories from
recorded neural data of a monkey as it reaches for eight different
targets on a fronto-parallel screen. The data, provided by the
laboratory of Prof. Krishna Shenoy at Stanford University [1],
consists of time-discretized spike trains of 98 neural units
recorded over 100 trials for each of the reaching angles.
Additionally, the simultaneous 3D hand trajectory data for each
of those trials is given.
The continuous estimation algorithm accepts a set of training
data (consisting of hand trajectories and neural data for all
angles) for training the model in a supervised manner, and a set
of testing data (consisting of neural data only) for model
performance evaluation purposes as input. By the help of the
trained model, the neural decoding algorithm classifies the
specific target angle and causally estimates the monkey’s x- and
y- hand positions over time from the spike trains of the testing
data. Hereby, consecutive 20ms increments of recorded data are
being fed into the model.
The implemented algorithm draws upon concepts from
general recursive Bayesian filtering applied to recorded firing
rates, neural tuning curves and population vector analysis as well
as nearest neighbor (NN) classifications for estimating specific
target angles and predicting 2D hand trajectories.
![Confusion_matrix](https://user-images.githubusercontent.com/44643180/110509290-dabd8e80-80f9-11eb-84fa-6a4f9fd73398.png)
![Method_2](https://user-images.githubusercontent.com/44643180/110509356-ed37c800-80f9-11eb-81ce-b6dfdbe793b2.png)



[1] https://shenoy.people.stanford.edu/overview
(Acessed: 20 th April 2020)
