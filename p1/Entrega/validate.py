import random
import time
import math
import sys
import io
import os
import warnings
import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Suprimir warnings de Gymnasium
warnings.filterwarnings('ignore', category=UserWarning, module='gymnasium')

# Configurar codificación UTF-8 para la consola de Windows
if sys.platform == 'win32':
    import locale
    if sys.stdout.encoding != 'utf-8':
        sys.stdout.reconfigure(encoding='utf-8')

# Importar el entorno y funciones auxiliares del script de entrenamiento
from train import (
    RoboboEnv,
    get_robot_pos,
    get_cylinder_pos,
    get_distance_to_target,
    get_angle_to_target
)

# Función local para parsear acciones (flechas Unicode)
def parse_action(action: int):
    """Convierte número de acción a símbolo de flecha"""
    if action == 0:
        return "↑"
    elif action == 1:
        return "←"
    elif action == 2:
        return "→"
    elif action == 3:
        return "↓"
    else:
        return "?"

# Configuración de estilo para las gráficas
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class ValidationMetrics:
    """Clase para almacenar y gestionar métricas de validación"""
    
    def __init__(self):
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances = []  # Distancia final al objetivo
        self.episode_successes = []  # Si alcanzó el objetivo
        self.episode_trajectories = []  # Trayectorias (x, z) del robot
        self.episode_target_positions = []  # Posiciones del cilindro
        self.episode_step_rewards = []  # Recompensas por paso de cada episodio
        self.episode_actions = []  # Acciones tomadas en cada episodio
        
    def add_episode(self, total_reward, length, final_distance, success, 
                    trajectory, target_pos, step_rewards, actions):
        """Añadir datos de un episodio completo"""
        self.episode_rewards.append(total_reward)
        self.episode_lengths.append(length)
        self.episode_distances.append(final_distance)
        self.episode_successes.append(success)
        self.episode_trajectories.append(trajectory)
        self.episode_target_positions.append(target_pos)
        self.episode_step_rewards.append(step_rewards)
        self.episode_actions.append(actions)
        
    def get_statistics(self):
        """Calcular estadísticas descriptivas"""
        return {
            'num_episodes': len(self.episode_rewards),
            'mean_reward': np.mean(self.episode_rewards),
            'std_reward': np.std(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'max_reward': np.max(self.episode_rewards),
            'median_reward': np.median(self.episode_rewards),
            'mean_length': np.mean(self.episode_lengths),
            'std_length': np.std(self.episode_lengths),
            'mean_final_distance': np.mean(self.episode_distances),
            'std_final_distance': np.std(self.episode_distances),
            'success_rate': np.mean(self.episode_successes) * 100,
            'num_successes': np.sum(self.episode_successes),
        }
        
    def save_to_file(self, filepath):
        """Guardar métricas en archivo .npz"""
        np.savez(
            filepath,
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
            episode_distances=np.array(self.episode_distances),
            episode_successes=np.array(self.episode_successes),
            episode_trajectories=np.array(self.episode_trajectories, dtype=object),
            episode_target_positions=np.array(self.episode_target_positions, dtype=object),
            episode_step_rewards=np.array(self.episode_step_rewards, dtype=object),
            episode_actions=np.array(self.episode_actions, dtype=object)
        )


def run_validation_episode(model, env, episode_num, deterministic=True, max_steps=200):
    """
    Ejecutar un episodio de validación completo
    
    Args:
        model: Modelo entrenado de StableBaselines3
        env: Entorno de Gymnasium
        episode_num: Número del episodio actual
        deterministic: Si True, usa política determinista (sin exploración)
        max_steps: Número máximo de pasos por episodio
        
    Returns:
        Tupla con (recompensa_total, longitud, distancia_final, éxito, trayectoria, 
                   posición_objetivo, recompensas_paso, acciones)
    """
    obs, info = env.reset()
    
    episode_reward = 0
    episode_length = 0
    trajectory = []
    step_rewards = []
    actions = []
    
    # Obtener posición inicial del objetivo
    # Acceder al entorno real a través del wrapper
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    target_pos = actual_env.target_pos
    
    # Listas para guardar información de cada paso
    step_info = []
    
    for step in range(max_steps):
        # Obtener acción del modelo
        action, _states = model.predict(obs, deterministic=deterministic)
        
        # Guardar posición actual del robot
        actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
        robot_pos = get_robot_pos(actual_env.sim)
        trajectory.append((robot_pos['x'], robot_pos['z']))
        actions.append(int(action))
        
        # Calcular distancia y ángulo ANTES del step
        distance_before = get_distance_to_target(robot_pos, target_pos)
        angle_before = get_angle_to_target(robot_pos, target_pos)
        
        # Ejecutar acción (suprimiendo prints del environment)
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        try:
            obs, reward, terminated, truncated, info = env.step(int(action))
        finally:
            sys.stdout = old_stdout
        
        episode_reward += reward
        episode_length += 1
        step_rewards.append(reward)
        
        # Guardar información del paso
        step_info.append({
            'step': step + 1,
            'action': parse_action(int(action)),
            'reward': reward,
            'distance': distance_before,
            'angle': angle_before,
            'cyl_x': target_pos['x'],
            'cyl_z': target_pos['z']
        })
        
        done = terminated or truncated
        
        if done:
            # Guardar última posición
            actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
            robot_pos = get_robot_pos(actual_env.sim)
            trajectory.append((robot_pos['x'], robot_pos['z']))
            
            final_distance = get_distance_to_target(robot_pos, target_pos)
            success = terminated and final_distance <= 100
            
            # Mostrar título y tabla con todos los pasos del episodio
            print(f"\n{'='*90}")
            print(f"{'EPISODIO ' + str(episode_num):^90}")
            print(f"{'='*90}")
            print(f"{'Paso':>5} | {'Accion':>7} | {'Recompensa':>10} | {'Distancia':>10} | {'Angulo':>8} | {'Cil.X':>8} | {'Cil.Z':>8}")
            print("-"*90)
            for info_step in step_info:
                print(f"{info_step['step']:>5} | {info_step['action']:>7} | "
                      f"{info_step['reward']:>10.3f} | {info_step['distance']:>10.2f} | "
                      f"{info_step['angle']:>7.2f}° | {info_step['cyl_x']:>8.2f} | {info_step['cyl_z']:>8.2f}")
            print("-"*90)
            
            # Resultado del episodio
            if success:
                print(f"[EXITO] Robot alcanzo el objetivo")
            elif truncated:
                print(f"[TRUNCADO] Demasiados pasos sin ver objetivo")
            else:
                print(f"[TERMINADO] Episodio terminado")
                
            print(f"Recompensa total: {episode_reward:.2f}")
            print(f"Longitud: {episode_length} pasos")
            print(f"Distancia final: {final_distance:.2f}")
            
            return (episode_reward, episode_length, final_distance, success, 
                    trajectory, target_pos, step_rewards, actions)
    
    # Si llegamos al máximo de pasos
    actual_env = env.unwrapped if hasattr(env, 'unwrapped') else env
    robot_pos = get_robot_pos(actual_env.sim)
    final_distance = get_distance_to_target(robot_pos, target_pos)
    
    # Mostrar título y tabla con todos los pasos del episodio
    print(f"\n{'='*90}")
    print(f"{'EPISODIO ' + str(episode_num):^90}")
    print(f"{'='*90}")
    print(f"{'Paso':>5} | {'Accion':>7} | {'Recompensa':>10} | {'Distancia':>10} | {'Angulo':>8} | {'Cil.X':>8} | {'Cil.Z':>8}")
    print("-"*90)
    for info_step in step_info:
        print(f"{info_step['step']:>5} | {info_step['action']:>7} | "
              f"{info_step['reward']:>10.3f} | {info_step['distance']:>10.2f} | "
              f"{info_step['angle']:>7.2f}° | {info_step['cyl_x']:>8.2f} | {info_step['cyl_z']:>8.2f}")
    print("-"*90)
    
    print(f"[ADVERTENCIA] Maximo de pasos alcanzado")
    print(f"Recompensa total: {episode_reward:.2f}")
    print(f"Distancia final: {final_distance:.2f}")
    
    return (episode_reward, episode_length, final_distance, False, 
            trajectory, target_pos, step_rewards, actions)


def plot_validation_results(metrics, output_dir="plots", file_prefix="validation"):
    """
    Generar todas las visualizaciones de validación
    
    Args:
        metrics: Objeto ValidationMetrics con los datos
        output_dir: Directorio donde guardar las gráficas
        file_prefix: Prefijo para los nombres de archivo (por defecto: nombre del script)
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # 1. Gráfico de recompensas por episodio
    plt.figure(figsize=(14, 6))
    episodes = np.arange(1, len(metrics.episode_rewards) + 1)
    
    # Crear máscara de colores según éxito
    colors = ['green' if s else 'red' for s in metrics.episode_successes]
    
    plt.subplot(1, 2, 1)
    plt.bar(episodes, metrics.episode_rewards, color=colors, alpha=0.7, edgecolor='black')
    plt.axhline(y=np.mean(metrics.episode_rewards), color='blue', linestyle='--', 
                linewidth=2, label=f'Media: {np.mean(metrics.episode_rewards):.2f}')
    plt.xlabel('Episodio', fontsize=12)
    plt.ylabel('Recompensa Total', fontsize=12)
    plt.title('Recompensas por Episodio de Validación', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Box plot de recompensas
    plt.subplot(1, 2, 2)
    plt.boxplot(metrics.episode_rewards, vert=True, patch_artist=True,
                boxprops=dict(facecolor='lightblue', alpha=0.7),
                medianprops=dict(color='red', linewidth=2))
    plt.ylabel('Recompensa Total', fontsize=12)
    plt.title('Distribución de Recompensas', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_results.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {output_dir}/{file_prefix}_results.jpg")
    
    # 2. Trayectorias 2D del robot
    plt.figure(figsize=(12, 12))
    
    # Colormap para diferenciar episodios (forma actualizada para Matplotlib 3.7+)
    cmap = plt.colormaps.get_cmap('tab10')
    
    for i, trajectory in enumerate(metrics.episode_trajectories):
        if len(trajectory) > 0:
            xs = [pos[0] for pos in trajectory]
            zs = [pos[1] for pos in trajectory]
            
            # Color según éxito
            if metrics.episode_successes[i]:
                color = 'green'
                alpha = 0.8
                linewidth = 2
                label = f'Ep {i+1} (✓)'
            else:
                color = 'red'
                alpha = 0.4
                linewidth = 1
                label = f'Ep {i+1} (✗)'
            
            # Dibujar trayectoria
            plt.plot(xs, zs, '-', color=color, alpha=alpha, linewidth=linewidth, label=label)
            
            # Marcar inicio y fin
            plt.plot(xs[0], zs[0], 'o', color='blue', markersize=8, 
                    markeredgecolor='black', markeredgewidth=1)
            plt.plot(xs[-1], zs[-1], 's', color=color, markersize=10, 
                    markeredgecolor='black', markeredgewidth=1.5, alpha=1)
            
            # Marcar posición del objetivo
            target = metrics.episode_target_positions[i]
            plt.plot(target['x'], target['z'], '*', color='gold', markersize=20,
                    markeredgecolor='black', markeredgewidth=2, alpha=0.9)
    
    plt.xlim(-1000, 1000)
    plt.ylim(-1000, 1000)
    plt.xlabel('Posición X', fontsize=12)
    plt.ylabel('Posición Z', fontsize=12)
    plt.title('Trayectorias 2D del Robot en Validación', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axis('equal')
    
    # Leyenda personalizada
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', 
               markersize=8, label='Inicio', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='green', 
               markersize=8, label='Fin (Éxito)', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='red', 
               markersize=8, label='Fin (Fallo)', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w', markerfacecolor='gold', 
               markersize=12, label='Objetivo', markeredgecolor='black')
    ]
    plt.legend(handles=legend_elements, loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_trajectories_2d.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {output_dir}/{file_prefix}_trajectories_2d.jpg")
    
    # 3. Estadísticas comparativas con boxplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Recompensas
    axes[0, 0].boxplot(metrics.episode_rewards, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightcoral', alpha=0.7),
                       medianprops=dict(color='darkred', linewidth=2))
    axes[0, 0].set_ylabel('Recompensa Total')
    axes[0, 0].set_title('Distribución de Recompensas')
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Longitudes de episodio
    axes[0, 1].boxplot(metrics.episode_lengths, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightblue', alpha=0.7),
                       medianprops=dict(color='darkblue', linewidth=2))
    axes[0, 1].set_ylabel('Número de Pasos')
    axes[0, 1].set_title('Distribución de Longitudes de Episodio')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Distancias finales
    axes[1, 0].boxplot(metrics.episode_distances, vert=True, patch_artist=True,
                       boxprops=dict(facecolor='lightgreen', alpha=0.7),
                       medianprops=dict(color='darkgreen', linewidth=2))
    axes[1, 0].set_ylabel('Distancia (unidades)')
    axes[1, 0].set_title('Distribución de Distancias Finales')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Tasa de éxito
    success_count = np.sum(metrics.episode_successes)
    fail_count = len(metrics.episode_successes) - success_count
    axes[1, 1].bar(['Éxito', 'Fallo'], [success_count, fail_count], 
                   color=['green', 'red'], alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Número de Episodios')
    axes[1, 1].set_title(f'Tasa de Éxito: {np.mean(metrics.episode_successes)*100:.1f}%')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Añadir valores sobre las barras
    for i, (label, value) in enumerate(zip(['Éxito', 'Fallo'], [success_count, fail_count])):
        axes[1, 1].text(i, value + 0.1, str(value), ha='center', va='bottom', 
                       fontsize=12, fontweight='bold')
    
    plt.suptitle('Estadísticas de Validación', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_boxplots.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {output_dir}/{file_prefix}_boxplots.jpg")
    
    # 4. Histograma de acciones tomadas
    plt.figure(figsize=(10, 6))
    all_actions = []
    for episode_actions in metrics.episode_actions:
        all_actions.extend(episode_actions)
    
    action_labels = {0: 'Adelante ↑', 1: 'Izquierda ←', 2: 'Derecha →'}
    action_counts = [all_actions.count(i) for i in range(3)]
    
    bars = plt.bar(range(3), action_counts, color=['blue', 'orange', 'green'], 
                   alpha=0.7, edgecolor='black', linewidth=2)
    plt.xticks(range(3), [action_labels[i] for i in range(3)], fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    plt.title('Distribución de Acciones Durante la Validación', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Añadir porcentajes sobre las barras
    total_actions = sum(action_counts)
    for i, (bar, count) in enumerate(zip(bars, action_counts)):
        percentage = (count / total_actions) * 100
        plt.text(bar.get_x() + bar.get_width()/2, count + max(action_counts)*0.01, 
                f'{count}\n({percentage:.1f}%)', 
                ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{file_prefix}_actions.jpg", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {output_dir}/{file_prefix}_actions.jpg")


def print_statistics(stats):
    """Imprimir estadísticas de forma legible"""
    print("\n" + "="*70)
    print("ESTADISTICAS DE VALIDACION")
    print("="*70)
    print(f"Numero de episodios:           {stats['num_episodes']}")
    print(f"Episodios exitosos:            {stats['num_successes']} ({stats['success_rate']:.1f}%)")
    print(f"\nRecompensas:")
    print(f"  Media:                       {stats['mean_reward']:.2f} +/- {stats['std_reward']:.2f}")
    print(f"  Minima:                      {stats['min_reward']:.2f}")
    print(f"  Maxima:                      {stats['max_reward']:.2f}")
    print(f"  Mediana:                     {stats['median_reward']:.2f}")
    print(f"\nLongitud de episodios:")
    print(f"  Media:                       {stats['mean_length']:.1f} +/- {stats['std_length']:.1f} pasos")
    print(f"\nDistancia final al objetivo:")
    print(f"  Media:                       {stats['mean_final_distance']:.2f} +/- {stats['std_final_distance']:.2f}")
    print("="*70)


def save_statistics_to_file(stats, filepath):
    """Guardar estadísticas en archivo de texto"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("ESTADÍSTICAS DE VALIDACIÓN\n")
        f.write("="*70 + "\n")
        f.write(f"Número de episodios:           {stats['num_episodes']}\n")
        f.write(f"Episodios exitosos:            {stats['num_successes']} ({stats['success_rate']:.1f}%)\n")
        f.write(f"\nRecompensas:\n")
        f.write(f"  Media:                       {stats['mean_reward']:.2f} ± {stats['std_reward']:.2f}\n")
        f.write(f"  Mínima:                      {stats['min_reward']:.2f}\n")
        f.write(f"  Máxima:                      {stats['max_reward']:.2f}\n")
        f.write(f"  Mediana:                     {stats['median_reward']:.2f}\n")
        f.write(f"\nLongitud de episodios:\n")
        f.write(f"  Media:                       {stats['mean_length']:.1f} ± {stats['std_length']:.1f} pasos\n")
        f.write(f"\nDistancia final al objetivo:\n")
        f.write(f"  Media:                       {stats['mean_final_distance']:.2f} ± {stats['std_final_distance']:.2f}\n")
        f.write("="*70 + "\n")
    print(f"\nEstadisticas guardadas en: {filepath}")


def main():
    """Función principal de validación"""
    # Obtener nombre del script actual
    SCRIPT_NAME = Path(__file__).stem
    OUTPUT_DIR = "plots"
    
    print("\n" + "="*70)
    print("INICIANDO VALIDACIÓN DEL MODELO")
    print("="*70)
    
    # Configuración
    MODEL_PATH = "checkpoint.zip"
    NUM_EPISODES = 10  # Número de episodios de validación
    MAX_STEPS = 200     # Pasos máximos por episodio
    DETERMINISTIC = True  # Usar política determinista (sin exploración)
    
    # Registrar el entorno
    gym.register(
        id="RoboboEnv-v0",
        entry_point=RoboboEnv,
    )
    
    # Crear entorno
    print(f"\nCreando entorno de validación...")
    env = gym.make("RoboboEnv-v0")
    
    # Cargar modelo entrenado
    print(f"Cargando modelo desde: {MODEL_PATH}")
    try:
        model = PPO.load(MODEL_PATH, env=env)
        print("Modelo cargado exitosamente")
    except Exception as e:
        print(f"[ERROR] Error al cargar el modelo: {e}")
        return
    
    # Crear objeto para almacenar métricas
    metrics = ValidationMetrics()
    
    # Ejecutar episodios de validación
    print(f"\nEjecutando {NUM_EPISODES} episodios de validación...")
    print(f"Política determinista: {DETERMINISTIC}")
    print(f"Máximo de pasos por episodio: {MAX_STEPS}")
    
    start_time = time.time()
    
    for episode in range(1, NUM_EPISODES + 1):
        # Ejecutar episodio
        (reward, length, distance, success, trajectory, 
         target_pos, step_rewards, actions) = run_validation_episode(
            model, env, episode, deterministic=DETERMINISTIC, max_steps=MAX_STEPS
        )
        
        # Guardar métricas
        metrics.add_episode(reward, length, distance, success, trajectory, 
                          target_pos, step_rewards, actions)
    
    elapsed_time = time.time() - start_time
    
    # Cerrar entorno
    env.close()
    
    print(f"\n{'='*80}")
    print(f"VALIDACION COMPLETADA EN {elapsed_time:.2f} SEGUNDOS")
    print(f"Tiempo promedio por episodio: {elapsed_time/NUM_EPISODES:.2f} segundos")
    print(f"{'='*80}")
    
    # Mostrar tabla resumen de todos los episodios
    print(f"\n{'='*90}")
    print("RESUMEN DE TODOS LOS EPISODIOS")
    print(f"{'='*90}")
    print(f"{'Episodio':>8} | {'Recompensa':>11} | {'Pasos':>6} | {'Dist.Final':>11} | {'¿Exito?':>8}")
    print("-"*90)
    for i in range(len(metrics.episode_rewards)):
        resultado = "Exito" if metrics.episode_successes[i] else "Fracaso"
        print(f"{i+1:>8} | {metrics.episode_rewards[i]:>11.2f} | {metrics.episode_lengths[i]:>6} | "
              f"{metrics.episode_distances[i]:>11.2f} | {resultado:>8}")
    print(f"{'='*90}")
    
    # Estadísticas globales
    success_rate = (sum(metrics.episode_successes) / len(metrics.episode_successes)) * 100
    mean_reward = np.mean(metrics.episode_rewards)
    std_reward = np.std(metrics.episode_rewards)
    mean_length = np.mean(metrics.episode_lengths)
    mean_distance = np.mean(metrics.episode_distances)
    min_reward = np.min(metrics.episode_rewards)
    max_reward = np.max(metrics.episode_rewards)
    min_length = np.min(metrics.episode_lengths)
    max_length = np.max(metrics.episode_lengths)
    
    print(f"\n{'='*80}")
    print("ESTADISTICAS GLOBALES")
    print(f"{'='*80}")
    print(f"Episodios completados:        {len(metrics.episode_successes)}")
    print(f"Tasa de exito:                {success_rate:.1f}% ({sum(metrics.episode_successes)}/{len(metrics.episode_successes)})")
    print(f"\nRecompensas:")
    print(f"  Media:                      {mean_reward:.2f} +/- {std_reward:.2f}")
    print(f"  Rango:                      [{min_reward:.2f}, {max_reward:.2f}]")
    print(f"\nLongitud de episodios:")
    print(f"  Media:                      {mean_length:.1f} pasos")
    print(f"  Rango:                      [{min_length}, {max_length}] pasos")
    print(f"\nDistancia final al objetivo:")
    print(f"  Media:                      {mean_distance:.2f} unidades")
    print(f"{'='*80}")
    
    # Crear diccionario con estadísticas para guardar
    stats = {
        'num_episodes': len(metrics.episode_successes),
        'num_successes': sum(metrics.episode_successes),
        'success_rate': success_rate,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'min_reward': min_reward,
        'max_reward': max_reward,
        'median_reward': np.median(metrics.episode_rewards),
        'mean_length': mean_length,
        'std_length': np.std(metrics.episode_lengths),
        'mean_final_distance': mean_distance,
        'std_final_distance': np.std(metrics.episode_distances)
    }
    
    # Guardar estadísticas en archivo de texto
    stats_file = f"{OUTPUT_DIR}/{SCRIPT_NAME}_statistics.txt"
    save_statistics_to_file(stats, stats_file)
    
    # Guardar métricas en archivo
    print(f"\nGuardando datos de validacion...")
    data_file = f"{OUTPUT_DIR}/{SCRIPT_NAME}_data.npz"
    metrics.save_to_file(data_file)
    print(f"Datos guardados en: {data_file}")
    
    # Generar visualizaciones
    print(f"\nGenerando graficos...")
    plot_validation_results(metrics, output_dir=OUTPUT_DIR, file_prefix=SCRIPT_NAME)
    
    print("\n" + "="*70)
    print("VALIDACION FINALIZADA")
    print("="*70)
    print(f"\nResultados guardados en el directorio: {OUTPUT_DIR}/")
    print(f"  - {SCRIPT_NAME}_results.jpg          (Recompensas y distribucion)")
    print(f"  - {SCRIPT_NAME}_trajectories_2d.jpg  (Trayectorias 2D)")
    print(f"  - {SCRIPT_NAME}_boxplots.jpg         (Estadisticas comparativas)")
    print(f"  - {SCRIPT_NAME}_actions.jpg          (Distribucion de acciones)")
    print(f"  - {SCRIPT_NAME}_data.npz             (Datos completos)")
    print(f"  - {SCRIPT_NAME}_statistics.txt       (Estadisticas en texto)")
    print()


if __name__ == "__main__":
    main()
