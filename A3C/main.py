import itertools
import threading
import multiprocessing
import tensorflow as tf
from keras.optimizers import RMSprop

from networks import ActorCritic
from workers import Worker
from utils import plot_results
from config import (
    INP_SHAPE,
    NUM_ACTIONS,
    DISCOUNT_FACTOR,
    MAX_STEPS,
    STEPS_BEFORE_UPDATE,
)


if __name__ == "__main__":
    global_network = ActorCritic(INP_SHAPE, NUM_ACTIONS)

    # Instantiate global optimizers (shared among workers)
    actor_opt = RMSprop(0.00025, 0.99, 0.0, 1e-6)
    actor_opt.build(global_network.actor.trainable_weights)
    critic_opt = RMSprop(0.00025, 0.99, 0.0, 1e-6)
    critic_opt.build(global_network.critic.trainable_weights)

    rewards_list = []  # stores rewards achieved by each individual worker
    global_step_counter = itertools.count()
    coordinator = tf.train.Coordinator()  # threads coordinator

    # Create workers
    workers = []
    for worker_id in range(multiprocessing.cpu_count()):
        worker = Worker(
            f"worker_#{worker_id+1}",
            global_network,
            actor_opt,
            critic_opt,
            global_step_counter,
            rewards_list,
            DISCOUNT_FACTOR,
            MAX_STEPS,
        )
        workers.append(worker)

    # Start worker threads
    worker_threads = []
    for worker in workers:
        worker_func = lambda: worker.run(coordinator, STEPS_BEFORE_UPDATE)
        t = threading.Thread(target=worker_func)
        t.start()
        worker_threads.append(t)

    # Wait for all workers to finish
    coordinator.join(
        worker_threads, stop_grace_period_secs=300
    )  # raises error if threads don't terminate within 5 minutes

    # Save model for inference
    global_network.actor.save("model.keras")
    # Plot results
    plot_results(rewards_list)
