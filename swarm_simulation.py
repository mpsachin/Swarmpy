import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Simulation Settings
N = 50               # Number of drones
WIDTH, HEIGHT = 800, 600
MAX_SPEED = 4
MAX_FORCE = 0.1
PERCEPTION = 50
SEPARATION = 25

class DroneSwarm:
    def __init__(self, n, w, h):
        self.n = n
        self.w = w
        self.h = h
        self.pos = np.random.rand(n, 2) * [w, h]
        angles = np.random.rand(n) * 2 * np.pi
        self.vel = np.column_stack((np.cos(angles), np.sin(angles))) * MAX_SPEED

    def update(self):
        new_vel = self.vel.copy()
        for i in range(self.n):
            diff = self.pos - self.pos[i]
            dist = np.linalg.norm(diff, axis=1)
            
            # Find close flockmates
            near = (dist > 0) & (dist < PERCEPTION)
            too_close = (dist > 0) & (dist < SEPARATION)
            
            sep = np.zeros(2)
            ali = np.zeros(2)
            coh = np.zeros(2)
            
            if np.any(too_close):
                sep = np.sum(-diff[too_close] / dist[too_close][:, np.newaxis], axis=0)
                sep = self.limit(sep, MAX_FORCE) * 1.5
                
            if np.any(near):
                ali = np.mean(self.vel[near], axis=0) - self.vel[i]
                ali = self.limit(ali, MAX_FORCE) * 1.0
                
                coh = (np.mean(self.pos[near], axis=0) - self.pos[i]) - self.vel[i]
                coh = self.limit(coh, MAX_FORCE) * 1.0
                
            new_vel[i] = self.limit(self.vel[i] + sep + ali + coh, MAX_SPEED)
            
        self.vel = new_vel
        self.pos = (self.pos + self.vel) % [self.w, self.h]

    @staticmethod
    def limit(v, max_val):
        norm = np.linalg.norm(v)
        return (v / norm) * max_val if norm > max_val and norm > 0 else v

# Initialize swarm object
swarm = DroneSwarm(N, WIDTH, HEIGHT)

# Setup plotting framework
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_facecolor('#111111')
fig.patch.set_facecolor('#111111')
scatter = ax.scatter(swarm.pos[:, 0], swarm.pos[:, 1], color='#00FFCC', s=25, edgecolors='white', alpha=0.8)
ax.set_xlim(0, WIDTH)
ax.set_ylim(0, HEIGHT)
ax.axis('off')
plt.title('Autonomous Drone Swarm Simulation (Boids Algorithm)', color='white', fontsize=12)

def animate(frame):
    swarm.update()
    scatter.set_offsets(swarm.pos)
    return scatter,

ani = animation.FuncAnimation(fig, animate, frames=200, interval=30, blit=True)
plt.show()
