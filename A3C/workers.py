from image_transformer import ImageTransformer
from utils import repeat_frame, get_next_state, Step
from nets import ValueNetwork, PolicyNetwork
from config import env


class Worker:
    def __init__(
        self,
        name,
        global_counter,
        global_value_network,
        global_policy_network,
        returns_list,
        discount_factor=0.99,
        max_steps=5e6,
    ):
        self.name = name
        self.global_counter = global_counter
        self.global_value_network = global_value_network
        self.global_policy_network = global_policy_network

        # self.discount_factor = discount_factor

        # self.max_steps = max_steps

        self.current_state = None  # to keep track of the current state
        self.img_transformer = ImageTransformer()
        self.worker_value_network = ValueNetwork()
        self.worker_policy_network = PolicyNetwork()

    def copy_weights(self):
        """Copies weights from global networks to local networks."""
        self.worker_value_network.model.set_weights(
            self.global_value_network.model.get_weights()
        )
        self.worker_policy_network.model.set_weights(
            self.global_policy_network.model.get_weights()
        )

    def update_weights(self):
        """Use gradient from local network to update the weights of global network."""
        """For complete model."""
        # Get global networks...
        global_policy_network = None
        global_value_network = None
        pass

    def run(self, coordinator):
        # Get initial state
        self.current_state = repeat_frame(
            self.img_transformer.transform(self.env.reset()[0])
        )
        assert self.current_state.ndims == 3  # .shape == (h, w, NUM_STACKED_FRAMES)

        #####
        ###
        #####
        while not coordinator.should_stop():
            # Copy weights from global networks to local networks so that the
            # workers always work with relatively fresh copy of maser networks weithgs.
            self.copy_weights()

            step_count = None  # run_n_steps()  # rough, needs change...

            if step_count >= self.max_steps:
                coordinator.request_stop()
                return
            self.update()  # write...

        pass

    def sample_action(self, state):
        self.worker_policy_network.get_logits(state)

    def run_n_steps(self, n):
        steps = []
        for _ in range(n):
            # Take a step
            action = None
            obs, reward, terminated, truncated, _ = env.step(action)
            next_state = get_next_state(self.state, self.img_transformer.transform(obs))

            # Save the step
            step = Step(
                self.current_state, action, reward, next_state, terminated or truncated
            )
            steps.append(step)

            # Update state
            self.current_state = next_state

            # next(self.global_counter)

            # if done_flag:  # initialize total_reward
            #     print(f"Total Reward: {self.total_reward} - {self.name}")
