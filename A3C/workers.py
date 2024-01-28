from image_transformer import ImageTransformer
from utils import repeat_frame


class Worker:
    def __init__(
        self,
        name,
        env,
        p_network,
        v_network,
        counter,
        returns_list,
        discount_factor=0.99,
        max_steps=5e6,  # why none ?
    ):
        self.name = name
        self.env = env
        self.global_policy_network = p_network
        self.global_value_network = v_network
        self.total_steps_counter = counter
        self.discount_factor = discount_factor
        self.img_transformer = ImageTransformer()

        self.worker_policy_network, self.worker_value_network = (
            None,
            None,
        )  # get_networks()

    def copy_weights(self):
        """Copies weights from the global networks to local networks."""
        # model.set_weights(base_model.get_weights())
        pass

    def update_weights():
        """
        Updates...
        """
        pass

    def run(self):
        # Get initial state
        obs, _ = self.env.reset()
        frame = self.img_transformer.transform(obs)
        state = repeat_frame(frame)
        assert state.shape == 3  # state.shape == (h, w, NUM_STACKED_FRAMES)

        pass
