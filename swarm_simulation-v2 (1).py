# -*- coding: utf-8 -*-
"""
Autonomous Swarm Simulation v3 - Attrition Logic Edition
Implements a 2D multi-agent flocking model (Boids) with target tracking,
obstacle avoidance, and dynamic attrition logic (drones are destroyed by threats).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION PARAMETERS ---
NUM_DRONES = 50
WIDTH, HEIGHT = 800, 600

# Kinematics
MAX_SPEED = 4.0
MAX_FORCE = 0.2

# Neighborhood Radii
SEPARATION_DIST = 25.0
PERCEPTION_DIST = 60.0

# Behavioral Weights
W_SEPARATION = 1.5
W_ALIGNMENT = 1.0
W_COHESION = 1.0
W_TARGET = 0.8
W_OBSTACLE = 3.0  # High priority to avoid danger

# Obstacles / Threats (x, y, radius)
THREATS = [
    np.array([250.0, 300.0, 50.0]),
    np.array([550.0, 250.0, 60.0]),
    np.array([400.0, 450.0, 45.0])
]

class SwarmSimulation:
    def __init__(self, num_drones):
        self.num_drones = num_drones
        # Random starting positions in a cluster to start as a cohesive group
        self.positions = np.random.uniform(50, 200, (num_drones, 2))
        
        # Random initial velocities
        angles = np.random.uniform(0, 2 * np.pi, num_drones)
        speeds = np.random.uniform(1, MAX_SPEED, num_drones)
        self.velocities = np.column_stack((np.cos(angles) * speeds, np.sin(angles) * speeds))
        
        # Active status tracking (True = alive, False = shot down)
        self.active_mask = np.ones(num_drones, dtype=bool)
        self.frame_count = 0

    def limit(self, vector, max_val):
        """Helper to cap vector magnitudes to kinematic limits."""
        mag = np.linalg.norm(vector, axis=-1, keepdims=True)
        # Avoid division by zero
        mag_mask = mag > 0
        scaled = np.zeros_like(vector)
        # Scale down elements exceeding maximum allowed constraint
        scale_factors = np.where(mag > max_val, max_val / (mag + 1e-6), 1.0)
        return vector * scale_factors

    def update(self, target_pos):
        self.frame_count += 1
        
        # Pull out slicing arrays containing only elements currently alive
        alive_idx = np.where(self.active_mask)[0]
        if len(alive_idx) == 0:
            return  # The entire swarm has been neutralized

        pos = self.positions[alive_idx]
        vel = self.velocities[alive_idx]
        n_alive = len(alive_idx)

        # 1. ATTRITION CHECK: Evaluate proximity to threat kill zones
        for threat in THREATS:
            threat_center = threat[:2]
            kill_radius = threat[2]
            
            # Compute distance vector to center of active air defenses
            diff = pos - threat_center
            dist = np.linalg.norm(diff, axis=1)
            
            # Identify which active drones fell within lethal radii
            killed_local_indices = np.where(dist < kill_radius)[0]
            if len(killed_local_indices) > 0:
                global_indices_killed = alive_idx[killed_local_indices]
                self.active_mask[global_indices_killed] = False

        # Re-evaluate live pool after processing structural losses
        alive_idx = np.where(self.active_mask)[0]
        if len(alive_idx) == 0:
            return
            
        pos = self.positions[alive_idx]
        vel = self.velocities[alive_idx]
        n_alive = len(alive_idx)

        # Compute full pair-wise distance matrix between surviving platforms
        diff_matrix = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff_matrix, axis=-1)

        # Core behavioral force components initializing matrices
        sep_forces = np.zeros((n_alive, 2))
        align_forces = np.zeros((n_alive, 2))
        coh_forces = np.zeros((n_alive, 2))

        # 2. CALCULATE BOIDS FLOCKING VECTORS
        for i in range(n_alive):
            # Mask out self coordinate indexes to parse neighbor fields
            neighbors = (dist_matrix[i] > 0) & (dist_matrix[i] < PERCEPTION_DIST)
            separators = (dist_matrix[i] > 0) & (dist_matrix[i] < SEPARATION_DIST)

            # Separation Force calculation
            if np.any(separators):
                # Push direction scales higher the closer objects get
                close_diffs = diff_matrix[i, separators]
                close_dists = dist_matrix[i, separators][:, np.newaxis]
                sep_forces[i] = np.sum(close_diffs / (close_dists ** 2), axis=0)

            # Alignment and Cohesion calculations
            if np.any(neighbors):
                # Alignment matching vector
                mean_vel = np.mean(vel[neighbors], axis=0)
                align_forces[i] = mean_vel - vel[i]

                # Cohesion grouping vector
                mean_pos = np.mean(pos[neighbors], axis=0)
                desired_coh = mean_pos - pos[i]
                coh_forces[i] = desired_coh - vel[i]

        # 3. CALCULATE TARGET SEEKING FORCES
        target_dir = target_pos - pos
        target_dist = np.linalg.norm(target_dir, axis=1, keepdims=True)
        # Prevent division warnings for precise coordinates
        target_dist = np.where(target_dist == 0, 1.0, target_dist)
        desired_target_vel = (target_dir / target_dist) * MAX_SPEED
        target_forces = desired_target_vel - vel

        # 4. CALCULATE OBSTACLE/THREAT AVOIDANCE FORCES
        obstacle_forces = np.zeros((n_alive, 2))
        for threat in THREATS:
            threat_center = threat[:2]
            threat_rad = threat[2]
            buffer_zone = threat_rad + 40.0  # Buffer alert envelope
            
            vec_to_threat = pos - threat_center
            dist_to_threat = np.linalg.norm(vec_to_threat, axis=1, keepdims=True)
            dist_to_threat = np.where(dist_to_threat == 0, 1.0, dist_to_threat)

            # Check if drone entered threat warning envelope
            inside_buffer = dist_to_threat < buffer_zone
            
            # Force scales exponentially higher as drone moves deeper into buffer zone
            push_magnitude = (buffer_zone - dist_to_threat) / buffer_zone
            push_dir = vec_to_threat / dist_to_threat
            
            # Apply force if inside buffer window
            obstacle_forces += np.where(inside_buffer, push_dir * push_magnitude * MAX_SPEED * 2, 0)

        # Apply kinematic limits on operational components
        sep_forces = self.limit(sep_forces, MAX_FORCE)
        align_forces = self.limit(align_forces, MAX_FORCE)
        coh_forces = self.limit(coh_forces, MAX_FORCE)
        target_forces = self.limit(target_forces, MAX_FORCE)
        obstacle_forces = self.limit(obstacle_forces, MAX_FORCE)

        # 5. INTEGRATE FORCE MATRIX INTO ACCELERATION
        acceleration = (W_SEPARATION * sep_forces + 
                        W_ALIGNMENT * align_forces + 
                        W_COHESION * coh_forces + 
                        W_TARGET * target_forces + 
                        W_OBSTACLE * obstacle_forces)

        # Update core motion dynamics state tracking matrices
        vel = self.limit(vel + acceleration, MAX_SPEED)
        pos += vel

        # Screen boundary handling - bounce back if hitting margins instead of wrapping
        for i in range(n_alive):
            if pos[i, 0] < 0 or pos[i, 0] > WIDTH:
                vel[i, 0] *= -1
                pos[i, 0] = np.clip(pos[i, 0], 0, WIDTH)
            if pos[i, 1] < 0 or pos[i, 1] > HEIGHT:
                vel[i, 1] *= -1
                pos[i, 1] = np.clip(pos[i, 1], 0, HEIGHT)

        # Commit internal math arrays back to master swarm index matrices
        self.positions[alive_idx] = pos
        self.velocities[alive_idx] = vel

# --- ANIMATION SETUP ---
sim = SwarmSimulation(NUM_DRONES)

fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_title("Autonomous Swarm Simulation v3: Attrition & Air Defense Kill Zones")

# Render Static Threat/Kill Zone Radii
for threat in THREATS:
    # Outer defensive perimeter buffer indicator
    ax.add_patch(plt.Circle((threat[0], threat[1]), threat[2] + 40, color='yellow', alpha=0.1, linestyle='--'))
    # Inner lethal hardware target neutralization boundary
    ax.add_patch(plt.Circle((threat[0], threat[1]), threat[2], color='red', alpha=0.4))
    ax.text(threat[0], threat[1], "KILL
ZONE", color='darkred', ha='center', va='center', fontsize=8, weight='bold')

# Scatter layer mappings
target_dot, = ax.plot([], [], 'go', ms=10, label="Dynamic Objective")
swarm_scatter = ax.scatter([], [], c='blue', marker='^', s=30, label="Active UAVs")

# Telemetry scoreboard HUD interface layout
hud_text = ax.text(20, HEIGHT - 30, "", color="black", weight="bold", fontsize=10)
ax.legend(loc="lower left")

def animate(frame):
    # Calculate a moving figure-eight target path coordinate frame
    t = frame * 0.015
    target_x = WIDTH / 2 + np.cos(t) * (WIDTH / 3)
    target_y = HEIGHT / 2 + np.sin(2 * t) * (HEIGHT / 4)
    target_pos = np.array([target_x, target_y])
    
    # Run loop iteration pass equations updating agent positional vectors
    sim.update(target_pos)
    
    # Filter matching indexes isolating surviving items
    alive_idx = np.where(sim.active_mask)[0]
    
    if len(alive_idx) > 0:
        # Update coordinate array indices mapping live nodes
        swarm_scatter.set_offsets(sim.positions[alive_idx])
        # Orient icons matching flight velocity vectors
        # Note: True orientation omitted here for standard quick scatter mapping
    else:
        swarm_scatter.set_offsets(np.empty((0, 2)))

    target_dot.set_data([target_x], [target_y])
    
    # Calculate current tactical fleet numbers
    losses = NUM_DRONES - len(alive_idx)
    survival_rate = (len(alive_idx) / NUM_DRONES) * 100
    
    hud_text.set_text(
        f"Fleet Telemetry Summary:\n"
        f"-------------------------\n"
        f"Initial Force Size : {NUM_DRONES} UAVs\n"
        f"Active Platforms   : {len(alive_idx)}\n"
        f"Combat Attrition   : {losses} Units Shot Down\n"
        f"Current Survival   : {survival_rate:.1f}%"
    )
    
    return swarm_scatter, target_dot, hud_text

ani = animation.FuncAnimation(fig, animate, frames=1000, interval=30, blit=True)
plt.show()
