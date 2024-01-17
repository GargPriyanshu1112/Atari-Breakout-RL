# Atari-Breakout-RL

While training the Q-learning model, we use two networks.
- **Online Network:** This network is used to *predict the action value functions (Q-values)*. Once the training process is completed, this network is used to form the greedy policy. This is the model that we constantly update during the training process. 

- **Target Network:** This network is *used to compute the target outputs* for training the network. In other words, *predictions by this network as well as obtained rewards are ued to form the outputs for the trainng proces.

However, the parameters of this model are updates less frequently than the online network. In the code, the target network's parameters we set to be updated after every (**) steps. After this many time steps, we copy the parameters from the online network to target network.