# Import dependencies
import gym
import itertools
import tensorflow as tf

from config import Config
from workers import Worker
from utils import get_networks

if __name__ == "__main__":
    # Initialize the Breakout environment
    env = gym.make(
        id="ALE/Breakout-v5",
        full_action_space=False,
        repeat_action_probability=0.1,
        obs_type="rgb",
    )
    env.reset()

    # define policy and value network
    policy_network, value_network = get_networks()

    total_steps_counter = itertools.count()
    returns_list = []

    threads_coordinator = tf.train.Coordinator()

    workers = []
    for worker_id in range(Config.NUM_WORKERS):
        worker = Worker(
            f"worker_#{worker_id+1}",
            env,
            policy_network,
            value_network,
            total_steps_counter,
            returns_list,
            Config.DISCOUNT_FACTOR,
            Config.MAX_STEPS,
        )
        workers.append(worker)

    worker_threads = []
    for worker in workers:
        worker_func = lambda: worker.run()
