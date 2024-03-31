## Reinforcement Learning
Reinforcement Learning (RL) is a type of machine learning wherein an agent learns to make decisions through interaction with its environment. The agent receives feedback in the form of rewards or penalties based on its actions and its goal is to maximize the cumulative reward over time.

This learning paradigm mirrors human learning, where individuals explore and refine their actions through trial and error. Through exploration, the agent learns to estimate the optimal actions to take in specific states.


## Deep Q-learning
Deep Q-Learning is a reinforcement learning algorithm that combines Q-Learning, a classical reinforcement learning method, with deep neural networks. The goal of DQL is to approximate the optimal action-value function, Q(s, a), where 's' represents the state of the environment and 'a' is the action taken by the agent. The Q-value represents the expected cumulative reward the agent will receive when taking action 'a' in state 's' and following the optimal policy thereafter.

Here's a brief overview of the key components of DQL:
1. **Q-Network:** In DQL, the Q-value function is parameterized by a deep neural network. The input to the network is the state of the environment, and the output is a Q-value for each possible action.

2. **Replay Buffer:** The replay buffer is populated with episodes of completely random actions. During training, random batches of experiences are sampled from this buffer to update the Q-network. By taking random samples from past experience, we get a better representation of the true data distribution. With the default Q-update, we will always learn from beginning to end (not desirable).

3. **Target Network:** To further stabilize training, DQL uses two Q-networks: the target network and the online network. The target network's parameters are updated less frequently and are used to compute the target Q-values during training. This helps prevent the network from chasing a moving target during training.

4. **Epsilon-Greedy Policy:** An epsilon-greedy policy balances exploration and exploitation ensuring that the agent explores the environment initially but gradually shifts towards exploitation as training progresses.


## Asynchronous Advantage Actor-Critic (A3C)
A3C (Asynchronous Advantage Actor-Critic) is an advanced reinforcement learning technique utilizing parallel agents. By combining asynchronous updates, it efficiently trains deep neural networks in high-dimensional environments. Gradients computed by each worker are aggregated and used to update a shared global network, facilitating efficient learning across multiple agents.

Here's a brief overview of the key components of A3C:
1. **Asynchronous Updates:** A3C utilizes asynchronous updates, meaning multiple copies of the environment and agents run in parallel. Each agent interacts with its own copy of the environment, collects experiences, and updates the global network asynchronously. This parallelism significantly accelerates learning.

2. **Advantage Estimation:** A3C utilizes an advantage function, Q(s,a) − V(s), to assess the benefit of selecting a specific action 'a' in state 's' relative to the average action value. This comparison aids the agent in distinguishing between favorable and unfavorable actions, enabling it to prioritize actions that yield superior outcomes.

3. **Actor-Critic Architecture:** A3C employs an actor-critic architecture:
   - **Actor**: The policy network that learns to select actions based on the observed states.
   - **Critic**: The value network that evaluates the goodness of the actions taken by the actor by estimating the expected cumulative reward.
