import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- SIMULATION PARAMETERS ---
NUM_BOIDS = 60
WIDTH, HEIGHT = 800, 600

# Boid physical constraints
MAX_SPEED = 4.0
MAX_FORCE = 0.15

# Behavior radii
PERCEPTION_RADIUS = 60.0
SEPARATION_RADIUS = 25.0

# Dynamic target
TARGET = np.array([400.0, 300.0])
TARGET_ANGLE = 0.0

# Static obstacles (X, Y, Radius)
OBSTACLES = [
    np.array([250.0, 200.0, 40.0]),
    np.array([550.0, 450.0, 50.0]),
    np.array([400.0, 150.0, 30.0]),
    np.array([450.0, 400.0, 35.0])
]

# Weights for behaviors
W_SEPARATION = 1.5
W_ALIGNMENT = 1.0
W_COHESION = 1.0
W_TARGET = 0.8
W_OBSTACLE = 2.5

class SwarmSimulation:
    def __init__(self, num_boids, width, height):
        self.num_boids = num_boids
        self.width = width
        self.height = height
        
        # Initialize positions randomly within bounds
        self.positions = np.random.rand(num_boids, 2) * [width, height]
        
        # Initialize velocities using random angles
        angles = np.random.rand(num_boids) * 2 * np.pi
        self.velocities = np.column_stack((np.cos(angles), np.sin(angles))) * MAX_SPEED

    def limit(self, vector, max_val):
        """Truncate magnitude of a vector array to a maximum value."""
        norm = np.linalg.norm(vector, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1.0, norm)  # Avoid division by zero
        scale = np.minimum(norm, max_val) / norm
        return vector * scale

    def update(self):
        global TARGET, TARGET_ANGLE
        
        # 1. Update dynamic target position (moves in an infinity loop pattern)
        TARGET_ANGLE += 0.015
        TARGET[0] = 400.0 + 250.0 * np.cos(TARGET_ANGLE)
        TARGET[1] = 300.0 + 150.0 * np.sin(2 * TARGET_ANGLE)
        
        # 2. Compute pairwise distance matrices between all boids
        diff_pos = self.positions[:, np.newaxis, :] - self.positions[np.newaxis, :, :]
        distances = np.linalg.norm(diff_pos, axis=2)
        
        # Masks for neighborhood interactions (excluding self)
        in_perception = (distances > 0) & (distances < PERCEPTION_RADIUS)
        in_separation = (distances > 0) & (distances < SEPARATION_RADIUS)
        
        # Count neighbors per boid
        neighbor_counts = np.sum(in_perception, axis=1, keepdims=True)
        
        # --- BEHAVIOR 1: SEPARATION ---
        sep_forces = np.zeros((self.num_boids, 2))
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_dist = np.where(in_separation, 1.0 / distances, 0.0)
        repulsion_vectors = -diff_pos * inv_dist[:, :, np.newaxis]
        sep_forces = np.sum(repulsion_vectors, axis=1)
        
        # --- BEHAVIOR 2: ALIGNMENT ---
        align_forces = np.zeros((self.num_boids, 2))
        sum_vel = np.dot(in_perception, self.velocities)
        avg_vel = np.where(neighbor_counts > 0, sum_vel / np.maximum(neighbor_counts, 1), self.velocities)
        if np.any(neighbor_counts > 0):
            avg_vel_norm = np.linalg.norm(avg_vel, axis=1, keepdims=True)
            avg_vel_norm = np.where(avg_vel_norm == 0, 1.0, avg_vel_norm)
            desired_align = (avg_vel / avg_vel_norm) * MAX_SPEED
            align_forces = desired_align - self.velocities
            
        # --- BEHAVIOR 3: COHESION ---
        cohesion_forces = np.zeros((self.num_boids, 2))
        sum_pos = np.dot(in_perception, self.positions)
        avg_pos = np.where(neighbor_counts > 0, sum_pos / np.maximum(neighbor_counts, 1), self.positions)
        desired_cohesion = avg_pos - self.positions
        cohesion_norm = np.linalg.norm(desired_cohesion, axis=1, keepdims=True)
        cohesion_norm = np.where(cohesion_norm == 0, 1.0, cohesion_norm)
        desired_cohesion = (desired_cohesion / cohesion_norm) * MAX_SPEED
        cohesion_forces = np.where(neighbor_counts > 0, desired_cohesion - self.velocities, 0.0)

        # --- BEHAVIOR 4: TARGET TRACKING ---
        to_target = TARGET - self.positions
        target_norm = np.linalg.norm(to_target, axis=1, keepdims=True)
        target_norm = np.where(target_norm == 0, 1.0, target_norm)
        desired_target = (to_target / target_norm) * MAX_SPEED
        target_forces = desired_target - self.velocities

        # --- BEHAVIOR 5: OBSTACLE AVOIDANCE ---
        obstacle_forces = np.zeros((self.num_boids, 2))
        for obs in OBSTACLES:
            obs_pos = obs[:2]
            obs_rad = obs[2]
            
            to_obs = obs_pos - self.positions
            obs_dist = np.linalg.norm(to_obs, axis=1, keepdims=True)
            
            safety_buffer = 40.0
            is_near_obs = obs_dist < (obs_rad + safety_buffer)
            
            push_direction = -to_obs / np.where(obs_dist == 0, 1.0, obs_dist)
            force_magnitude = (obs_rad + safety_buffer - obs_dist) / (obs_rad + safety_buffer)
            force_magnitude = np.clip(force_magnitude, 0, 1)
            
            desired_avoid = push_direction * force_magnitude * MAX_SPEED
            obstacle_forces += np.where(is_near_obs, desired_avoid - self.velocities, 0.0)

        # --- COMBINE ACCELERATIONS ---
        sep_forces = self.limit(sep_forces, MAX_FORCE)
        align_forces = self.limit(align_forces, MAX_FORCE)
        cohesion_forces = self.limit(cohesion_forces, MAX_FORCE)
        target_forces = self.limit(target_forces, MAX_FORCE)
        obstacle_forces = self.limit(obstacle_forces, MAX_FORCE)
        
        acceleration = (W_SEPARATION * sep_forces + 
                        W_ALIGNMENT * align_forces + 
                        W_COHESION * cohesion_forces + 
                        W_TARGET * target_forces + 
                        W_OBSTACLE * obstacle_forces)
        
        acceleration = self.limit(acceleration, MAX_FORCE)
        
        self.velocities = self.limit(self.velocities + acceleration, MAX_SPEED)
        self.positions += self.velocities
        
        self.positions[:, 0] = np.mod(self.positions[:, 0], self.width)
        self.positions[:, 1] = np.mod(self.positions[:, 1], self.height)

# --- RUN ANIMATION RENDERING ENGINE ---
sim = SwarmSimulation(NUM_BOIDS, WIDTH, HEIGHT)

fig, ax = plt.subplots(figsize=(9, 7))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_aspect('equal')
ax.set_title("Autonomous Swarm Simulation: Obstacle Avoidance & Target Tracking", fontsize=12, pad=10)
ax.axis('off')

for obs in OBSTACLES:
    circle = plt.Circle((obs[0], obs[1]), obs[2], color='red', alpha=0.3)
    ax.add_patch(circle)
    ax.plot(obs[0], obs[1], 'rx', markersize=4)

target_dot, = ax.plot([], [], 'go', markersize=10, label='Moving Target Objective')
swarm_scatter = ax.scatter([], [], marker='^', c='blue', s=25, alpha=0.8, label='Autonomous Swarm Unit')
ax.legend(loc='upper right')

def init_anim():
    swarm_scatter.set_offsets(np.empty((0, 2)))
    target_dot.set_data([], [])
    return swarm_scatter, target_dot

def animate(frame):
    sim.update()
    swarm_scatter.set_offsets(sim.positions)
    target_dot.set_data([TARGET[0]], [TARGET[1]])
    return swarm_scatter, target_dot

anim = animation.FuncAnimation(fig, animate, init_func=init_anim, frames=200, interval=25, blit=True)
plt.show()
