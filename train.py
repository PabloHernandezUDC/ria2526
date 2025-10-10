import random, time, math
import numpy as np
import gymnasium as gym
from robobopy.Robobo import Robobo
from robobopy.utils.IR import IR
from robobosim.RoboboSim import RoboboSim
from robobopy.utils.BlobColor import BlobColor
from math import dist
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt



seed = 42


class CustomCallback(BaseCallback):
    """
    A custom callback that derives from ``BaseCallback``.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        self.rewards = list()
        self.ep_lengths = list()
        self.current_episode_rewards = list()
        self.positions = list()  # List of episodes, each episode is a list of (x, z) positions
        self.current_episode_positions = list()


    def _on_step(self) -> bool:
        # Access the info dict from the last step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]  # For single environment
            if "step_reward" in info:
                self.current_episode_rewards.append(info["step_reward"])
            if "robot_pos" in info:
                self.current_episode_positions.append((info["robot_pos"]["x"], info["robot_pos"]["z"]))
        
        # Check if episode ended
        if len(self.locals.get("dones", [])) > 0:
            if self.locals["dones"][0]:  # Episode finished
                if self.current_episode_rewards:
                    self.rewards.append(sum(self.current_episode_rewards))
                    self.current_episode_rewards = list()
                if self.current_episode_positions:
                    self.positions.append(self.current_episode_positions.copy())
                    self.current_episode_positions = list()
        
        return True


    def _on_training_end(self) -> None:
        # Plot the rewards with moving average
        if self.rewards:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot raw rewards
            episodes = np.arange(len(self.rewards))
            ax.plot(episodes, self.rewards, alpha=0.3, label='Episode Reward', color='blue')
            
            # Calculate and plot moving average
            window_size = 10
            if len(self.rewards) >= window_size:
                moving_avg = []
                for i in range(len(self.rewards)):
                    if i < window_size - 1:
                        # For early episodes, use average of all episodes so far
                        moving_avg.append(np.mean(self.rewards[:i+1]))
                    else:
                        # Use last 10 episodes
                        moving_avg.append(np.mean(self.rewards[i-window_size+1:i+1]))
                
                ax.plot(episodes, moving_avg, linewidth=2, label=f'Moving Average (last {window_size} episodes)', color='red')
            
            ax.set_xlabel("Episode")
            ax.set_ylabel("Total Reward")
            ax.set_title("Rewards per Episode with Moving Average")
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig("plots/episode_rewards.jpg")
            plt.close()
            
            print(f"\nTotal episodes: {len(self.rewards)}")
            print(f"Mean reward: {np.mean(self.rewards):.2f}")
            print(f"Std reward: {np.std(self.rewards):.2f}")
        
        # Plot the robot trajectories
        if self.positions:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 10))
            
            num_episodes = len(self.positions)
            
            # Create color gradient from red to green
            for i, episode_positions in enumerate(self.positions):
                if len(episode_positions) > 0:
                    # Calculate color: red (1, 0, 0) -> green (0, 1, 0)
                    ratio = i / max(1, num_episodes - 1)  # Avoid division by zero
                    color = (1 - ratio, ratio, 0)  # RGB: red to green
                    
                    # Extract x and z coordinates
                    xs = [pos[0] for pos in episode_positions]
                    zs = [pos[1] for pos in episode_positions]
                    
                    # Plot the trail
                    ax.plot(xs, zs, '-', color=color, alpha=0.6, linewidth=1)
                    # Mark start position
                    ax.plot(xs[0], zs[0], 'o', color="black", markersize=4, alpha=1)
                    # Mark end position
                    ax.plot(xs[-1], zs[-1], 'o', color="black", markersize=4, alpha=1)
            
            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Z Position")
            ax.set_title(f"Robot Trajectories (Red=Early Episodes, Green=Late Episodes)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            
            plt.savefig("plots/robot_trajectories.jpg", dpi=150)
            plt.close()
            
            print(f"Saved trajectory plot with {num_episodes} episodes")

class RoboboEnv(gym.Env):

    def __init__(self):
        self.observation_space = gym.spaces.Dict(
            {
                # "agent": gym.spaces.Box(-1000, 1000, shape=(3,), dtype=int),   # [x, z] coordinates and [y] rotation
                # "target": gym.spaces.Box(-1000, 1000, shape=(2,), dtype=int),  # [x, z] coordinates
                # "x_offset": gym.spaces.Box(-1, 100, shape=(1,), dtype=int),
                "sector": gym.spaces.Discrete(6)
            }
        )

        self.action_space = gym.spaces.Discrete(3)

        speed = 10
        self._action_to_direction = {
            0: np.array([speed*3, speed*3]),   # Forward
            1: np.array([0, speed*2]),         # Turn Left
            2: np.array([speed*2, 0]),         # Turn Right
            # 3: np.array([-speed, -speed]),   # Go Backwards
        }
        
        ip = "localhost"
        self.robobo = Robobo(ip)
        self.sim = RoboboSim(ip)
        self.robobo.connect()
        self.sim.connect()
                
        self.target_pos = get_cylinder_pos(self.sim)
        self.target_color = BlobColor.RED
        
        self.steps_without_target = 0
            
    def step(self, action):
        l_speed, r_speed = self._action_to_direction[action]
        
        duration = 0.5 # habría que adaptarlo si se acelera el simulador
        self.robobo.moveWheelsByTime(r_speed, l_speed, duration=duration, wait=True)
        time.sleep(.01)

        observation = self._get_obs()
        
        if observation["sector"] == 5:
            self.steps_without_target += 1
        else:
            self.steps_without_target = 0
        

        distance = get_distance_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )
        
        angle = get_angle_to_target(
            get_robot_pos(self.sim),
            self.target_pos            
        )

        reward = get_reward(distance, angle, alpha=0.4)

        print(f"Action: {parse_action(action)} | Reward: {(reward):.3f} | Distance: {(distance):.3f} | Obs: {observation}")

        terminated = False
        if distance <= 100:
            print(f"Target reached!")
            terminated = True
            # reward += 200
                
        info = self._get_info()
        info["step_reward"] = reward  # Store the reward in info
        info["robot_pos"] = get_robot_pos(self.sim)  # Store robot position
        
        truncated = False
        if self.steps_without_target >= 35:
            print(f"Too many steps without seeing target!")
            truncated = True
            self.steps_without_target = 0
            reward -= 100
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        print("Resetting env...")
        
        self.sim.resetSimulation()
        self.robobo.stopMotors()
        time.sleep(.1) # sin esto no funciona (???????????)
        self.robobo.moveTiltTo(115, speed=20, wait=True)
        # time.sleep(2)
        
        observation = self._get_obs()
        info = self._get_info()
        
        self.target_pos = get_cylinder_pos(self.sim)
        
        return observation, info
    
    def render(self):
        # TODO
        ...
    
    def close(self):
        self.robobo.disconnect()
        self.sim.disconnect()
    
    def _get_obs(self):
        red_x = np.array([self.robobo.readColorBlob(self.target_color).posx])
        if red_x == 0:
            sector = 5
        elif red_x == 100:
            sector = 4 # para corregir y que el 100 no se quede solo en el sector 5
        else:
            sector = red_x // 20
        
        return {
            "sector": np.array([sector], dtype=int).flatten()
            }
    
    def _get_info(self):
        return {}
        

def parse_action(action: int):
    if action == 0:
        return "↑"
    if action == 1:
        return "←"
    if action == 2:
        return "→"
    if action == 3:
        return "↓"
    else:
        return "unknown"


def get_robot_pos(sim: RoboboSim):
    data = sim.getRobotLocation(0)
    
    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]    
    rot_y = data["rotation"]["y"]
    
    return {"x": pos_x, "z": pos_z, "y": rot_y}


def get_cylinder_pos(sim: RoboboSim):
    # IMPORTANTE: la posición del cilindro no se actualiza en tiempo real,
    # solo podemos saber la posición en la que empieza
    data = sim.getObjectLocation("CYLINDERMIDBALL")

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]
    
    return {"x": pos_x, "z": pos_z}


def get_distance_to_target(robot_pos: dict, target_pos: dict):
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]
    return dist((rx, rz), (tx, tz))


def get_angle_to_target(robot_pos: dict, target_pos: dict):
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]

    dx = tx - rx
    dz = tz - rz
    
    target_angle = math.degrees(math.atan2(dx, dz))
    
    robot_angle = robot_pos["y"]
    
    angle_diff = target_angle - robot_angle
    
    # Normalize to [-180, 180]
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360
        
    return angle_diff


def get_reward(distance: float, angle: float, alpha: float = 0.5):
    r1 = 1000 / distance
    r2 = -(abs(angle) / 90)

    return (alpha)*r1 + (1-alpha)*r2


def plot_evaluation_results():
    data = np.load("eval_results/evaluations.npz")

    timesteps = data["timesteps"]
    rewards = data["results"]
    ep_lengths = data["ep_lengths"]

    mean_rewards = np.mean(rewards, axis=1)
    std_rewards = np.std(rewards, axis=1)

    mean_ep_lengths = np.mean(ep_lengths, axis=1)
    std_ep_lengths = np.std(ep_lengths, axis=1)

    for type, means, stds in (("reward", mean_rewards, std_rewards), ("episode_length", mean_ep_lengths, std_ep_lengths)):
        plt.clf()
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.errorbar(timesteps, means, yerr=stds, fmt="o-", capsize=4, color="black", ecolor="blue")
        ax.set_xticks(timesteps)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(f"Mean {type}")
        fig.suptitle(f"Mean and std. {type} over training")
        plt.tight_layout()
        plt.savefig(f"plots/eval_{type}s.jpg")
        plt.close()


def main():
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    train_env = Monitor(gym.make(id))
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        seed=seed,
        n_steps=512,
        )
    
    eval_env = Monitor(gym.make(id))
    eval_callback = EvalCallback(
        eval_env,
        log_path="eval_results/",
        eval_freq=512,
        n_eval_episodes=5
    )
    
    custom_callback = CustomCallback(verbose=1)
    callback_list = CallbackList([eval_callback, custom_callback])
    
    start = time.time()
    model.learn(total_timesteps=8192, callback=callback_list, progress_bar=True)
    learning_time = time.time() - start
    
    plot_evaluation_results()
    
    print(f"Training took {(learning_time):.2f} seconds.")
    
    model.save("checkpoint.zip")


if __name__ == "__main__":
    main()

'''
Eval num_timesteps=8192, episode_reward=-24.96 +/- 49.44
Episode length: 24.80 +/- 11.12
------------------------------------------
| eval/                   |              |
|    mean_ep_length       | 24.8         |
|    mean_reward          | -25          |
| time/                   |              |
|    total_timesteps      | 8192         |
| train/                  |              |
|    approx_kl            | 0.0062855026 |
|    clip_fraction        | 0.0854       |
|    clip_range           | 0.2          |
|    entropy_loss         | -0.9         |
|    explained_variance   | 0.0434       |
|    learning_rate        | 0.0003       |
|    loss                 | 546          |
|    n_updates            | 150          |
|    policy_gradient_loss | -0.00604     |
|    value_loss           | 1.04e+03     |
------------------------------------------
---------------------------------
| rollout/           |          |
|    ep_len_mean     | 50.8     |
|    ep_rew_mean     | -56.7    |
| time/              |          |
|    fps             | 2        |
|    iterations      | 16       |
|    time_elapsed    | 2745     |
|    total_timesteps | 8192     |
---------------------------------

Total episodes: 179
Mean reward: -0.47
Std reward: 17.55
Saved trajectory plot with 179 episodes
 100% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8,192/8,192  [ 0:45:45 < 0:00:00 , ? it/s ]
Training took 2746.97 seconds.

'''