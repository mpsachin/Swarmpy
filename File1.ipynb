import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# --- CONFIGURATION PARAMETERS ---
NUM_BOIDS = 50           # Initial number of drones
MAX_SPEED = 4.0          # Maximum speed of a drone
MAX_FORCE = 0.15         # Maximum steering force
WIDTH, HEIGHT = 800, 600 # Viewport size

# Rule Radii
SEPARATION_RADIUS = 25.0
PERCEPTION_RADIUS = 60.0

# Rule Weights
W_SEPARATION = 1.5
W_ALIGNMENT = 1.0
W_COHESION = 1.0
W_TARGET = 0.8
W_OBSTACLE = 2.5

# Threats (Air Defense Nodes)
# format: [x, y, avoidance_radius, kill_radius]
THREATS = np.array([
    [300.0, 200.0, 90.0, 50.0],
    [550.0, 450.0, 80.0, 40.0],
    [200.0, 450.0, 70.0, 35.0]
])

# Logistics & Wave parameters
WAVE_REINFORCEMENT_SIZE = 25
TRIGGER_THRESHOLD = 15
INITIAL_BATTERY_MIN = 300
INITIAL_BATTERY_MAX = 500

# --- SIMULATION STATE DATA ---
positions = np.random.rand(NUM_BOIDS, 2) * [WIDTH, HEIGHT]
angles = np.random.rand(NUM_BOIDS) * 2 * np.pi
velocities = np.column_stack((np.cos(angles), np.sin(angles))) * MAX_SPEED
batteries = np.random.randint(INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX, size=NUM_BOIDS).astype(float)

# Scoring Metrics
air_defense_losses = 0
battery_losses = 0
current_wave = 1

# --- CORE MATHEMATICAL FUNCTIONS ---
def limit(vector, max_val):
    """Truncates a vector matrix to a predefined maximum magnitude."""
    norms = np.linalg.norm(vector, axis=1, keepdims=True)
    norms[norms == 0] = 1.0  # Prevent division by zero
    factor = np.minimum(max_val / norms, 1.0)
    return vector * factor

def seek(pos, vel, target):
    """Calculates a steering force vector toward a specific target coordinate."""
    desired = target - pos
    norms = np.linalg.norm(desired, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    desired = (desired / norms) * MAX_SPEED
    steering = desired - vel
    return limit(steering, MAX_FORCE)

def calculate_swarm_forces(pos, vel):
    """Computes the primary Boids Reynolds vectors (Separation, Alignment, Cohesion)."""
    n = pos.shape[0]
    sep_forces = np.zeros((n, 2))
    align_forces = np.zeros((n, 2))
    coh_forces = np.zeros((n, 2))
    
    if n <= 1:
        return sep_forces, align_forces, coh_forces

    # Compute Euclidean distance matrices across all active drone combinations
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    dist_sq = np.sum(diff**2, axis=-1)
    dist = np.sqrt(dist_sq)
    
    # 1. Separation Core
    sep_mask = (dist > 0) & (dist < SEPARATION_RADIUS)
    for i in range(n):
        neighbors = sep_mask[i]
        if np.any(neighbors):
            # Repulsion strength is inversely proportional to proximity distance
            repulsion = diff[i, neighbors] / dist[i, neighbors, np.newaxis]**2
            sep_forces[i] = -np.sum(repulsion, axis=0)
            
    # 2. Alignment & 3. Cohesion Core
    percept_mask = (dist > 0) & (dist < PERCEPTION_RADIUS)
    for i in range(n):
        neighbors = percept_mask[i]
        if np.any(neighbors):
            # Alignment: Match average neighborhood heading vector
            align_forces[i] = np.mean(vel[neighbors], axis=0) - vel[i]
            # Cohesion: Steer toward the spatial neighborhood center of mass
            center_of_mass = np.mean(pos[neighbors], axis=0)
            desired_coh = center_of_mass - pos[i]
            coh_norms = np.linalg.norm(desired_coh)
            if coh_norms > 0:
                desired_coh = (desired_coh / coh_norms) * MAX_SPEED
            coh_forces[i] = desired_coh - vel[i]
            
    return limit(sep_forces, MAX_FORCE), limit(align_forces, MAX_FORCE), limit(coh_forces, MAX_FORCE)

def calculate_obstacle_avoidance(pos):
    """Calculates repulsive escape forces radiating away from air-defense barriers."""
    n = pos.shape[0]
    avoid_forces = np.zeros((n, 2))
    
    for threat in THREATS:
        t_pos = threat[:2]
        avoid_r = threat[2]
        
        diff = pos - t_pos
        dist = np.linalg.norm(diff, axis=1, keepdims=True)
        dist[dist == 0] = 1.0
        
        # Apply steering correction if inside the warning threshold perimeter
        inside_mask = (dist < avoid_r).flatten()
        if np.any(inside_mask):
            push_direction = diff / dist
            push_magnitude = (avoid_r - dist) / avoid_r
            avoid_forces[inside_mask] += (push_direction * push_magnitude * MAX_SPEED)[inside_mask]
            
    return limit(avoid_forces, MAX_FORCE * 2.0)

# --- MATPLOTLIB RENDERING ENGINE SETUP ---
fig, ax = plt.subplots(figsize=(10, 7))
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.set_facecolor('#111116')
fig.patch.set_facecolor('#111116')

# Render Visual Assets
scat = ax.scatter([], [], c='#3498db', s=25, marker='^', zorder=3, label='Drone Platform')
target_dot, = ax.plot([], [], 'go', ms=8, zorder=4, label='Mission Objective')

# Draw Threat Bubbles
for threat in THREATS:
    # Outer Predictive Avoidance Circle
    c_avoid = plt.Circle((threat[0], threat[1]), threat[2], color='#f39c12', fill=False, ls='--', alpha=0.4)
    # Inner Lethal System Boundary
    c_kill = plt.Circle((threat[0], threat[1]), threat[3], color='#e74c3c', fill=True, alpha=0.3)
    ax.add_patch(c_avoid)
    ax.add_patch(c_kill)

# Setup Live HUD Scoring Text Area
hud_text = ax.text(15, HEIGHT - 20, '', color='white', fontsize=10, 
                   fontfamily='monospace', bbox=dict(facecolor='black', alpha=0.6, boxstyle='round,pad=0.5'))

ax.legend(loc='lower left', facecolor='#222', edgecolor='#444', labelcolor='white')
plt.title("Decentralized Drone Swarm Framework - Attrition & Logistics Simulation", color='white', pad=15)
plt.tight_layout()

# --- MAIN RUNTIME ANIMATION LOOP ---
def update(frame):
    global positions, velocities, batteries, air_defense_losses, battery_losses, current_wave
    
    n = positions.shape[0]
    
    # 1. Evaluate Logistics/Reinforcement Waves
    if n < TRIGGER_THRESHOLD:
        current_wave += 1
        new_pos = np.random.rand(WAVE_REINFORCEMENT_SIZE, 2) * 50.0 + 50.0 # Spawn at (50, 50) staging base
        new_angles = np.random.rand(WAVE_REINFORCEMENT_SIZE) * 2 * np.pi
        new_vel = np.column_stack((np.cos(new_angles), np.sin(new_angles))) * MAX_SPEED
        new_batt = np.random.randint(INITIAL_BATTERY_MIN, INITIAL_BATTERY_MAX, size=WAVE_REINFORCEMENT_SIZE).astype(float)
        
        positions = np.vstack((positions, new_pos))
        velocities = np.vstack((velocities, new_vel))
        batteries = np.concatenate((batteries, new_batt))
        n = positions.shape[0]

    # 2. Update Moving Target Path Coordinate (Figure-Eight / Lemniscate)
    t = frame * 0.015
    target_x = WIDTH / 2 + (WIDTH * 0.35) * np.sin(t)
    target_y = HEIGHT / 2 + (HEIGHT * 0.35) * np.sin(2 * t) / 2
    target_dot.set_data([target_x], [target_y])
    target_coord = np.array([target_x, target_y])

    # 3. Calculate Steering System Behaviors
    f_sep, f_align, f_coh = calculate_swarm_forces(positions, velocities)
    f_target = seek(positions, velocities, target_coord)
    f_obstacle = calculate_obstacle_avoidance(positions)

    # 4. Integrate Force Vectors into Physics Engine Matrix
    acceleration = (W_SEPARATION * f_sep + 
                    W_ALIGNMENT * f_align + 
                    W_COHESION * f_coh + 
                    W_TARGET * f_target + 
                    W_OBSTACLE * f_obstacle)
    
    velocities = limit(velocities + acceleration, MAX_SPEED)
    positions += velocities

    # 5. Process Boundary Constraints (Screen Wrap-Around)
    positions[:, 0] = np.mod(positions[:, 0], WIDTH)
    positions[:, 1] = np.mod(positions[:, 1], HEIGHT)

    # 6. Apply Resource Drain (Battery Life Depreciation)
    batteries -= 1.0

    # 7. Combat Attrition Computations
    survival_mask = np.ones(n, dtype=bool)
    
    # Check Threat Intersections (Air Defense Lethal Kill-Zones)
    for threat in THREATS:
        t_pos = threat[:2]
        kill_r = threat[3]
        dist_to_threat = np.linalg.norm(positions - t_pos, axis=1)
        killed_by_defense = dist_to_threat < kill_r
        
        air_defense_losses += np.sum(killed_by_defense & survival_mask)
        survival_mask[killed_by_defense] = False

    # Check Fuel/Battery Depletion Limits
    dead_batteries = batteries <= 0
    battery_losses += np.sum(dead_batteries & survival_mask)
    survival_mask[dead_batteries] = False

    # Scrub All Casualties from Active Computing Tensors Instantly
    positions = positions[survival_mask]
    velocities = velocities[survival_mask]
    batteries = batteries[survival_mask]

    # 8. Refresh Render Canvas Visuals
    scat.set_offsets(positions)
    
    # Update Scores and Telemetry Data on HUD
    hud_text.set_text(
        f"--- SWARM TELEMETRY HUD ---\n"
        f"Active Drone Platforms : {positions.shape[0]}\n"
        f"Air Defense Losses     : {air_defense_losses}\n"
        f"Battery Expirations    : {battery_losses}\n"
        f"Deployment Wave Index  : {current_wave}"
    )

    return scat, target_dot, hud_text

# Run Framework
ani = animation.FuncAnimation(fig, update, frames=2000, interval=20, blit=True)
plt.show()
