import time
import math
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

# Semilla para reproducibilidad
seed = 42


class CustomCallback(BaseCallback):
    """
    Callback personalizado para recopilar métricas durante el entrenamiento.
    
    Registra recompensas, longitudes de episodios, trayectorias del robot
    y genera visualizaciones al finalizar el entrenamiento.
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)

    def _on_training_start(self) -> None:
        """Inicializar listas para almacenar métricas"""
        self.rewards = list()
        self.ep_lengths = list()
        self.current_episode_rewards = list()
        self.positions = list()
        self.current_episode_positions = list()
        self.episode_outcomes = list()

    def _on_step(self) -> bool:
        """Llamado después de cada paso del entorno"""
        # Recopilar recompensa y posición del paso actual
        if len(self.locals.get("infos", [])) > 0:
            info = self.locals["infos"][0]
            if "step_reward" in info:
                self.current_episode_rewards.append(info["step_reward"])
            if "robot_pos" in info:
                self.current_episode_positions.append(
                    (info["robot_pos"]["x"], info["robot_pos"]["z"]))

        # Procesar final de episodio
        dones = self.locals.get("dones", [])
        if len(dones) > 0 and dones[0]:
            if self.current_episode_rewards:
                self.rewards.append(sum(self.current_episode_rewards))
                self.current_episode_rewards = list()
            if self.current_episode_positions:
                self.positions.append(self.current_episode_positions.copy())
                self.current_episode_positions = list()

            # Determinar tipo de finalización del episodio
            infos = self.locals.get("infos", [])
            if len(infos) > 0:
                info = infos[0]
                if info.get("is_truncated", False):
                    self.episode_outcomes.append('truncated')
                elif info.get("is_terminated", False):
                    self.episode_outcomes.append('terminated')
                else:
                    self.episode_outcomes.append('other')
            else:
                self.episode_outcomes.append('other')

        return True

    def _on_training_end(self) -> None:
        """Generar visualizaciones al finalizar el entrenamiento"""
        # Gráfica de recompensas con media móvil
        if self.rewards:
            plt.clf()
            fig, ax = plt.subplots(figsize=(12, 6))

            # Plot raw rewards
            episodes = np.arange(len(self.rewards))
            ax.plot(episodes, self.rewards, alpha=0.3,
                    label='Episode Reward', color='blue')

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
                        moving_avg.append(
                            np.mean(self.rewards[i-window_size+1:i+1]))

                ax.plot(episodes, moving_avg, linewidth=2,
                        label=f'Moving Average (last {window_size} episodes)', color='red')

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

        # Gráfica de trayectorias del robot
        if self.positions:
            plt.clf()
            fig, ax = plt.subplots(figsize=(10, 10))

            num_episodes = len(self.positions)

            # Gradiente de color rojo a verde para indicar progreso temporal
            for i, episode_positions in enumerate(self.positions):
                if len(episode_positions) > 0:
                    # Color: rojo (1, 0, 0) -> verde (0, 1, 0)
                    ratio = i / max(1, num_episodes - 1)
                    color = (1 - ratio, ratio, 0)

                    # Extraer coordenadas x, z
                    xs = [pos[0] for pos in episode_positions]
                    zs = [pos[1] for pos in episode_positions]

                    # Dibujar trayectoria
                    ax.plot(xs, zs, '-', color=color, alpha=0.6, linewidth=1)
                    
                    # Marcar posición final según el tipo de finalización
                    if i < len(self.episode_outcomes):
                        outcome = self.episode_outcomes[i]
                        if outcome == 'terminated':
                            # Estrella para éxito (objetivo alcanzado)
                            ax.plot(xs[-1], zs[-1], '*', color="gold", markersize=12,
                                    alpha=1, markeredgecolor='black', markeredgewidth=0.5)
                        elif outcome == 'truncated':
                            # Cruz para episodio truncado
                            ax.plot(xs[-1], zs[-1], 'x', color="red",
                                    markersize=8, alpha=1, markeredgewidth=2)
                    else:
                        # Marcador por defecto
                        ax.plot(xs[-1], zs[-1], 'o', color="black",
                                markersize=4, alpha=1)

            ax.set_xlim(-1000, 1000)
            ax.set_ylim(-1000, 1000)
            ax.set_xlabel("X Position")
            ax.set_ylabel("Z Position")
            ax.set_title(
                f"Robot Trajectories (Red=Early Episodes, Green=Late Episodes)")
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal')

            # Leyenda para los marcadores
            from matplotlib.lines import Line2D
            legend_elements = [
                Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
                       markeredgecolor='black', markersize=10, label='Target Found'),
                Line2D([0], [0], marker='x', color='w', markerfacecolor='red',
                       markeredgecolor='red', markersize=8, label='Truncated', markeredgewidth=2)
            ]
            ax.legend(handles=legend_elements, loc='upper right')

            plt.savefig("plots/robot_trajectories.jpg", dpi=150)
            plt.close()

            print(f"Saved trajectory plot with {num_episodes} episodes")


class RoboboEnv(gym.Env):
    """
    Entorno Gymnasium personalizado para el robot Robobo.
    
    El robot debe navegar hacia un cilindro rojo visible usando únicamente
    información visual (sector del campo de visión donde aparece el objetivo).
    """

    def __init__(self):
        # Espacio de observaciones: sector visual (0-5)
        self.observation_space = gym.spaces.Dict(
            {
                "sector": gym.spaces.Discrete(6)
            }
        )

        # Espacio de acciones: 3 acciones discretas
        self.action_space = gym.spaces.Discrete(3)

        # Mapeo de acciones a velocidades de ruedas
        speed = 10
        self._action_to_direction = {
            0: np.array([speed*3, speed*3]),   # Forward
            1: np.array([0, speed*2]),         # Turn Left
            2: np.array([speed*2, 0]),         # Turn Right
        }

        # Conexión con RoboboSim
        ip = "localhost"
        self.robobo = Robobo(ip)
        self.sim = RoboboSim(ip)
        self.robobo.connect()
        self.sim.connect()

        self.target_pos = get_cylinder_pos(self.sim)
        self.target_color = BlobColor.RED
        self.steps_without_target = 0

    def step(self, action):
        """Ejecutar una acción y devolver el resultado"""
        l_speed, r_speed = self._action_to_direction[action]

        duration = 0.5
        self.robobo.moveWheelsByTime(
            r_speed, l_speed, duration=duration, wait=True)
        time.sleep(.01)

        observation = self._get_obs()

        # Contador de pasos sin ver el objetivo
        if observation["sector"] == 5:
            self.steps_without_target += 1
        else:
            self.steps_without_target = 0

        # Calcular distancia y ángulo al objetivo
        distance = get_distance_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )

        angle = get_angle_to_target(
            get_robot_pos(self.sim),
            self.target_pos
        )

        # Calcular recompensa
        reward = get_reward(distance, angle, alpha=0.4)

        print(
            f"Action: {parse_action(action)} | Reward: {(reward):.3f} | Distance: {(distance):.3f} | Obs: {observation}")

        # Verificar si alcanzó el objetivo
        terminated = False
        if distance <= 100:
            print(f"Target reached!")
            terminated = True

        # Almacenar información adicional
        info = self._get_info()
        info["step_reward"] = reward
        info["robot_pos"] = get_robot_pos(self.sim)

        # Verificar truncamiento (perdió el objetivo demasiado tiempo)
        truncated = False
        if self.steps_without_target >= 35:
            print(f"Too many steps without seeing target!")
            truncated = True
            self.steps_without_target = 0
            reward -= 100

        info["is_truncated"] = truncated
        info["is_terminated"] = terminated

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reiniciar el entorno para un nuevo episodio"""
        print("Resetting env...")

        self.sim.resetSimulation()
        self.robobo.stopMotors()
        time.sleep(.1)
        self.robobo.moveTiltTo(115, speed=20, wait=True)

        observation = self._get_obs()
        info = self._get_info()

        self.target_pos = get_cylinder_pos(self.sim)

        return observation, info

    def render(self):
        # TODO
        ...

    def close(self):
        """Cerrar conexiones con el simulador"""
        self.robobo.disconnect()
        self.sim.disconnect()

    def _get_obs(self):
        """Obtener observación actual del entorno"""
        red_x = np.array([self.robobo.readColorBlob(self.target_color).posx])
        if red_x == 0:
            sector = 5
        elif red_x == 100:
            sector = 4
        else:
            sector = red_x // 20

        return {
            "sector": np.array([sector], dtype=int).flatten()
        }

    def _get_info(self):
        """Obtener información adicional del paso"""
        return {}


# Funciones auxiliares

def parse_action(action: int):
    """Convertir número de acción a símbolo visual"""
    if action == 0:
        return "↑"
    if action == 1:
        return "←"
    if action == 2:
        return "→"
    else:
        return "unknown"


def get_robot_pos(sim: RoboboSim):
    """Obtener posición actual del robot"""
    data = sim.getRobotLocation(0)

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]
    rot_y = data["rotation"]["y"]

    return {"x": pos_x, "z": pos_z, "y": rot_y}


def get_cylinder_pos(sim: RoboboSim):
    """Obtener posición del cilindro objetivo"""
    data = sim.getObjectLocation("CYLINDERMIDBALL")

    pos_x = data["position"]["x"]
    pos_z = data["position"]["z"]

    return {"x": pos_x, "z": pos_z}


def get_distance_to_target(robot_pos: dict, target_pos: dict):
    """Calcular distancia euclidiana al objetivo"""
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]
    return dist((rx, rz), (tx, tz))


def get_angle_to_target(robot_pos: dict, target_pos: dict):
    """Calcular ángulo entre orientación del robot y dirección al objetivo"""
    rx, rz = robot_pos["x"], robot_pos["z"]
    tx, tz = target_pos["x"], target_pos["z"]

    dx = tx - rx
    dz = tz - rz

    target_angle = math.degrees(math.atan2(dx, dz))
    robot_angle = robot_pos["y"]
    angle_diff = target_angle - robot_angle

    # Normalizar a rango [-180, 180]
    while angle_diff > 180:
        angle_diff -= 360
    while angle_diff < -180:
        angle_diff += 360

    return angle_diff


def get_reward(distance: float, angle: float, alpha: float = 0.5):
    """
    Calcular recompensa multi-componente.
    
    Args:
        distance: Distancia al objetivo
        angle: Ángulo al objetivo
        alpha: Peso del componente de distancia (0-1)
    """
    r1 = 1000 / distance
    r2 = -(abs(angle) / 90)

    return (alpha)*r1 + (1-alpha)*r2


def plot_evaluation_results():
    """Generar gráficas de las evaluaciones periódicas durante entrenamiento"""
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
        ax.errorbar(timesteps, means, yerr=stds, fmt="o-",
                    capsize=4, color="black", ecolor="blue")
        ax.set_xticks(timesteps)
        ax.set_xlabel("Timesteps")
        ax.set_ylabel(f"Mean {type}")
        fig.suptitle(f"Mean and std. {type} over training")
        plt.tight_layout()
        plt.savefig(f"plots/eval_{type}s.jpg")
        plt.close()
def main():
    """Función principal de entrenamiento"""
    # Registrar el entorno personalizado
    id = "RoboboEnv"
    gym.register(
        id=id,
        entry_point=RoboboEnv,
    )

    # Crear entorno de entrenamiento
    train_env = Monitor(gym.make(id))
    
    # Crear modelo PPO
    model = PPO(
        "MultiInputPolicy",
        train_env,
        verbose=1,
        seed=seed,
        n_steps=512,
    )

    # Configurar callbacks para evaluación y monitoreo
    eval_env = Monitor(gym.make(id))
    eval_callback = EvalCallback(
        eval_env,
        log_path="eval_results/",
        eval_freq=512,
        n_eval_episodes=5
    )

    custom_callback = CustomCallback(verbose=1)
    callback_list = CallbackList([eval_callback, custom_callback])

    # Entrenar el modelo
    start = time.time()
    model.learn(
        total_timesteps=8192,
        callback=callback_list,
        progress_bar=True)
    learning_time = time.time() - start

    # Generar gráficas de evaluación
    plot_evaluation_results()

    print(f"Training took {(learning_time):.2f} seconds.")

    # Guardar modelo entrenado
    model.save("checkpoint.zip")

if __name__ == "__main__":
    main()