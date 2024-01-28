import multiprocessing


class Config:
    MAX_STEPS = 5e6
    MIN_STEPS_BEFORE_UPDATE = 5  # no. of steps each worker has to perform before calculating the gradient and sending it back to the global network
    NUM_ACTIONS = 4
    NUM_WORKERS = multiprocessing.cpu_count()
