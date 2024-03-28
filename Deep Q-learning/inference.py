# Import dependencies
import gym
import imageio
import numpy as np
from keras.models import load_model

from episode import get_next_state
from image_transformer import ImageTransformer


def play(env, model, filename="./video/test.mp4"):
    img_transformer = ImageTransformer()

    with imageio.get_writer(filename, fps=30) as video:
        # Initial state
        obs, _ = env.reset()
        frame = img_transformer.transform(obs)
        state = np.stack([frame] * 4, axis=-1)

        try:
            reward_sum = 0
            step_counter = 0

            done = False
            while not done:
                video.append_data(obs)

                action = np.argmax(
                    model.predict(np.expand_dims(state, axis=0), verbose=False)
                )
                obs, reward, terminated, truncated, _ = env.step(action)
                frame = img_transformer.transform(obs)
                state = get_next_state(state, frame)

                reward_sum += reward
                step_counter += 1
                done = terminated or truncated
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            video.close()
            env.close()
    print(f"\nSteps: {step_counter} | Reward: {reward_sum}\n")


if __name__ == "__main__":
    env = gym.make("Breakout-v0", render_mode="human")
    model = load_model("./model/model_final.keras")

    play(env, model)
