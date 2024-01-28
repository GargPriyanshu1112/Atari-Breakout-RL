from keras.models import Sequential
from keras.layers import Input, Rescaling, Conv2D, Flatten, Dense


class BaseModel:
    def __init__(self, inp_shape):
        self.base_model = Sequential(
            [
                Input(shape=inp_shape),
                Rescaling(scale=1.0 / 255),
                Conv2D(16, kernel_size=(8, 8), strides=(4, 4), activation="relu"),
                Conv2D(32, kernel_size=(4, 4), strides=(2, 2), activation="relu"),
                Flatten(),
                Dense(units=256),
            ]
        )


class PolicyNetwork(BaseModel):
    def __init__(self, num_actions, reg_const=0.01):
        super().__init__()
        self.model = None


class ValueNetwork(BaseModel):
    def __init__(self):
        super().__init__()
        self.model = None


class PolicyNetwork:
    """
    base model -|
    final_dense with softmax. (prob for each action...)
    sample action from logits (result before softmax...)

    entrop-first vid...
    """

    pass
