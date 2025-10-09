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


    def _on_step(self) -> bool:
        # Access the info dict from the last step
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]  # For single environment
            if "step_reward" in info:
                self.current_episode_rewards.append(info["step_reward"])
        
        # Check if episode ended
        if len(self.locals.get("dones", [])) > 0:
            if self.locals["dones"][0]:  # Episode finished
                if self.current_episode_rewards:
                    self.rewards.append(sum(self.current_episode_rewards))
                    self.current_episode_rewards = list()
        
        return True


    def _on_training_end(self) -> None:
        # Plot the rewards
        if self.rewards:
            plt.clf()
            plt.plot(self.rewards)
            plt.xlabel("Episode")
            plt.ylabel("Total Reward")
            plt.title("Rewards per Episode")
            plt.savefig("episode_rewards.jpg")
            plt.close()
            
            print(f"\nTotal episodes: {len(self.rewards)}")
            print(f"Mean reward: {np.mean(self.rewards):.2f}")
            print(f"Std reward: {np.std(self.rewards):.2f}")

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
        plt.errorbar(timesteps, means, yerr=stds, fmt="o-", capsize=4, color="black", ecolor="blue")
        plt.xticks(timesteps)
        plt.xlabel("Timesteps")
        plt.ylabel(f"Mean {type}")
        plt.suptitle(f"Mean and std. {type} over training")
        plt.savefig(f"eval_{type}s.jpg")


def main():
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    train_env = Monitor(gym.make(id))
    model = PPO("MultiInputPolicy", train_env, verbose=1, seed=seed)
    
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

