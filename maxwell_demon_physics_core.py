"""
Maxwell's Demon Simulation - Core Physics Engine
================================================

A realistic 2D particle system with:
- Velocity Verlet integration for stable dynamics
- Elastic particle-particle and particle-wall collisions
- Spatial partitioning for O(n) collision detection
- Full mechanical property tracking (energy, momentum, forces)
- Ready for thermodynamic analysis and demon implementation

Author: Computational Physics
Last Modified: 2026
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from collections import defaultdict
import warnings

# ============================================================================
# PARTICLE CLASS
# ============================================================================

@dataclass
class Particle:
    """
    2D particle with position, velocity, acceleration, and physical properties.
    
    Attributes:
        x, y: Position in 2D space
        vx, vy: Velocity components
        ax, ay: Acceleration components
        mass: Particle mass (affects collision dynamics and inertia)
        radius: Particle radius (for collision detection and visualization)
        is_fixed: If True, particle doesn't move (for walls)
    """
    
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    ax: float = 0.0
    ay: float = 0.0
    mass: float = 1.0
    radius: float = 0.1
    is_fixed: bool = False
    
    # Optional: track collision history for debugging
    collision_count: int = 0
    
    @property
    def pos(self) -> np.ndarray:
        """Position vector."""
        return np.array([self.x, self.y])
    
    @pos.setter
    def pos(self, value: np.ndarray):
        """Set position from array."""
        self.x, self.y = value
    
    @property
    def vel(self) -> np.ndarray:
        """Velocity vector."""
        return np.array([self.vx, self.vy])
    
    @vel.setter
    def vel(self, value: np.ndarray):
        """Set velocity from array."""
        self.vx, self.vy = value
    
    @property
    def acc(self) -> np.ndarray:
        """Acceleration vector."""
        return np.array([self.ax, self.ay])
    
    @acc.setter
    def acc(self, value: np.ndarray):
        """Set acceleration from array."""
        self.ax, self.ay = value
    
    @property
    def speed(self) -> float:
        """Magnitude of velocity."""
        return np.sqrt(self.vx**2 + self.vy**2)
    
    @property
    def kinetic_energy(self) -> float:
        """Kinetic energy: KE = 0.5 * m * v²."""
        return 0.5 * self.mass * (self.vx**2 + self.vy**2)
    
    @property
    def momentum(self) -> np.ndarray:
        """Momentum vector: p = m * v."""
        return self.mass * self.vel
    
    def __repr__(self) -> str:
        return f"Particle(pos=[{self.x:.2f}, {self.y:.2f}], v={self.speed:.3f}, m={self.mass:.2f})"


# ============================================================================
# SPATIAL PARTITIONING FOR COLLISION DETECTION
# ============================================================================

class SpatialGrid:
    """
    Grid-based spatial partitioning for O(n) collision detection.
    
    Divides the simulation box into cells. Only particles in neighboring
    cells are checked for collisions, avoiding O(n²) brute force.
    """
    
    def __init__(self, box_size: float, cell_size: float):
        """
        Initialize spatial grid.
        
        Args:
            box_size: Size of simulation box (assumed square, 0 to box_size)
            cell_size: Size of each grid cell
        """
        self.box_size = box_size
        self.cell_size = cell_size
        self.cells = defaultdict(list)
    
    def clear(self):
        """Clear all cells."""
        self.cells.clear()
    
    def add_particle(self, particle: Particle):
        """Add particle to appropriate cell(s)."""
        cx = int(particle.x / self.cell_size)
        cy = int(particle.y / self.cell_size)
        
        # Clamp to grid
        cx = max(0, min(int(self.box_size / self.cell_size) - 1, cx))
        cy = max(0, min(int(self.box_size / self.cell_size) - 1, cy))
        
        self.cells[(cx, cy)].append(particle)
    
    def get_neighbors(self, particle: Particle) -> List[Particle]:
        """Get all particles in neighboring cells (and cell itself)."""
        cx = int(particle.x / self.cell_size)
        cy = int(particle.y / self.cell_size)
        
        # Clamp to grid
        cx = max(0, min(int(self.box_size / self.cell_size) - 1, cx))
        cy = max(0, min(int(self.box_size / self.cell_size) - 1, cy))
        
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                cell_key = (cx + dx, cy + dy)
                if cell_key in self.cells:
                    neighbors.extend(self.cells[cell_key])
        
        return neighbors


# ============================================================================
# COLLISION RESOLUTION
# ============================================================================

def resolve_elastic_collision(p1: Particle, p2: Particle,
                              restitution: float = 1.0,
                              friction: float = 0.0) -> bool:
    """
    Resolve elastic collision between two particles.
    
    Uses center-of-mass frame approach for realistic 2D collisions.
    
    Args:
        p1, p2: Particles to collide
        restitution: Coefficient of restitution (1.0 = perfectly elastic)
        friction: Tangential friction coefficient (0.0 = frictionless)
    
    Returns:
        True if collision occurred, False otherwise
    """
    
    # Vector from p1 to p2
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    dist_sq = dx**2 + dy**2
    dist = np.sqrt(dist_sq)
    
    # Check if particles overlap
    min_dist = p1.radius + p2.radius
    if dist >= min_dist or dist < 1e-10:
        return False
    
    # Normal vector (from p1 to p2)
    nx = dx / dist
    ny = dy / dist
    
    # Relative velocity
    dvx = p2.vx - p1.vx
    dvy = p2.vy - p1.vy
    
    # Relative velocity along collision normal
    dvn = dvx * nx + dvy * ny
    
    # Only collide if particles are approaching
    if dvn >= 0:
        return False
    
    # Fixed particles have infinite mass
    m1 = float('inf') if p1.is_fixed else p1.mass
    m2 = float('inf') if p2.is_fixed else p2.mass
    
    # Impulse magnitude (normal component)
    # j = -(1 + e) * (m2 * m1 / (m1 + m2)) * dvn
    if m1 == float('inf') and m2 == float('inf'):
        return False
    
    if m1 == float('inf'):
        impulse_magnitude = -(1 + restitution) * m2 * dvn / m2
    elif m2 == float('inf'):
        impulse_magnitude = -(1 + restitution) * m1 * dvn / m1
    else:
        impulse_magnitude = -(1 + restitution) * (m1 * m2 / (m1 + m2)) * dvn
    
    # Apply normal impulse
    if not p1.is_fixed:
        p1.vx -= impulse_magnitude * nx / p1.mass
        p1.vy -= impulse_magnitude * ny / p1.mass
    
    if not p2.is_fixed:
        p2.vx += impulse_magnitude * nx / p2.mass
        p2.vy += impulse_magnitude * ny / p2.mass
    
    # Tangential friction (optional)
    if friction > 0:
        # Tangential vector
        tx = -ny
        ty = nx
        
        # Relative velocity along tangent
        dvt = dvx * tx + dvy * ty
        
        # Friction impulse
        friction_magnitude = friction * impulse_magnitude
        friction_magnitude = min(friction_magnitude, abs(dvt) * 0.5)  # Limit to prevent instability
        
        if not p1.is_fixed:
            p1.vx += friction_magnitude * tx / p1.mass
            p1.vy += friction_magnitude * ty / p1.mass
        
        if not p2.is_fixed:
            p2.vx -= friction_magnitude * tx / p2.mass
            p2.vy -= friction_magnitude * ty / p2.mass
    
    # Separate particles to prevent overlap
    overlap = min_dist - dist
    separation = overlap / 2 + 0.001
    
    if not p1.is_fixed:
        p1.x -= separation * nx
        p1.y -= separation * ny
    
    if not p2.is_fixed:
        p2.x += separation * nx
        p2.y += separation * ny
    
    p1.collision_count += 1
    p2.collision_count += 1
    
    return True


def handle_wall_collision(particle: Particle, box_size: float,
                          restitution: float = 1.0,
                          friction: float = 0.0):
    """
    Handle collisions with rigid box walls (boundaries).
    
    Args:
        particle: Particle to check
        box_size: Size of simulation box
        restitution: Coefficient of restitution
        friction: Friction coefficient
    """
    
    # Left wall
    if particle.x - particle.radius < 0:
        particle.x = particle.radius
        particle.vx *= -restitution
        if friction > 0:
            particle.vy *= (1 - friction)
    
    # Right wall
    if particle.x + particle.radius > box_size:
        particle.x = box_size - particle.radius
        particle.vx *= -restitution
        if friction > 0:
            particle.vy *= (1 - friction)
    
    # Bottom wall
    if particle.y - particle.radius < 0:
        particle.y = particle.radius
        particle.vy *= -restitution
        if friction > 0:
            particle.vx *= (1 - friction)
    
    # Top wall
    if particle.y + particle.radius > box_size:
        particle.y = box_size - particle.radius
        particle.vy *= -restitution
        if friction > 0:
            particle.vx *= (1 - friction)


# ============================================================================
# SIMULATION ENGINE
# ============================================================================

class SimulationEngine:
    """
    2D particle simulation with realistic physics.
    
    Features:
    - Velocity Verlet integration for stability
    - Efficient collision detection via spatial grid
    - Full mechanical property tracking
    - Energy conservation verification
    - Ready for demon control logic
    """
    
    def __init__(self,
                 box_size: float = 10.0,
                 num_particles: int = 100,
                 dt: float = 0.001,
                 gravity: float = 0.0,
                 restitution: float = 0.99,
                 friction: float = 0.0,
                 particle_radius: float = 0.15,
                 temperature: float = 1.0):
        """
        Initialize simulation engine.
        
        Args:
            box_size: Size of square simulation box
            num_particles: Number of particles to create
            dt: Time step
            gravity: Gravitational acceleration (set to 0 for vibrating table analog)
            restitution: Coefficient of restitution in collisions
            friction: Friction coefficient
            particle_radius: Radius of particles
            temperature: Initial temperature (controls initial velocity distribution)
        """
        
        self.box_size = box_size
        self.dt = dt
        self.gravity = gravity
        self.restitution = restitution
        self.friction = friction
        self.particle_radius = particle_radius
        
        self.particles: List[Particle] = []
        self.time = 0.0
        self.step_count = 0
        
        # Spatial grid for collision detection
        # Cell size should be ~2-3x largest particle diameter
        cell_size = max(3 * particle_radius, 0.5)
        self.spatial_grid = SpatialGrid(box_size, cell_size)
        
        # Initialize particles with random positions and velocities
        self._initialize_particles(num_particles, temperature)
        
        # History for analysis
        self.history = {
            'time': [],
            'total_energy': [],
            'total_momentum': [],
            'temperature': [],
            'mean_speed': [],
            'collision_count': [],
        }
    
    def _initialize_particles(self, num_particles: int, temperature: float):
        """
        Create particles with random positions and Maxwell-Boltzmann velocities.
        
        Args:
            num_particles: Number of particles
            temperature: Controls width of velocity distribution
        """
        
        np.random.seed(42)  # For reproducibility
        
        margin = self.particle_radius + 0.1
        
        for i in range(num_particles):
            # Random position (avoid walls)
            x = np.random.uniform(margin, self.box_size - margin)
            y = np.random.uniform(margin, self.box_size - margin)
            
            # Maxwell-Boltzmann velocity distribution
            # sigma = sqrt(k_B * T / m) ≈ sqrt(T)
            sigma = np.sqrt(temperature)
            vx = np.random.normal(0, sigma)
            vy = np.random.normal(0, sigma)
            
            # Particle mass (can vary for multi-species simulation)
            mass = 1.0
            
            particle = Particle(
                x=x, y=y, vx=vx, vy=vy,
                mass=mass, radius=self.particle_radius
            )
            
            self.particles.append(particle)
    
    def _compute_forces(self) -> None:
        """
        Compute forces on all particles.
        
        Currently only gravity (which is usually 0 for vibrating table analog).
        Can be extended for external forces, thermostat, etc.
        """
        
        for particle in self.particles:
            # Gravity (downward)
            particle.ax = 0.0
            particle.ay = -self.gravity
    
    def _velocity_verlet_step(self) -> None:
        """
        Perform one integration step using Velocity Verlet method.
        
        This is more stable than Euler and conserves energy better.
        
        Algorithm:
            1. Compute acceleration at current position
            2. Update velocity (half step): v = v + 0.5 * a * dt
            3. Update position: x = x + v * dt
            4. Handle boundary collisions
            5. Detect and resolve particle-particle collisions
            6. Recompute acceleration
            7. Update velocity (half step): v = v + 0.5 * a * dt
        """
        
        # Step 1: Compute initial forces
        self._compute_forces()
        
        # Step 2-3: Half-step velocity update and position update
        for particle in self.particles:
            particle.vx += 0.5 * particle.ax * self.dt
            particle.vy += 0.5 * particle.ay * self.dt
            
            particle.x += particle.vx * self.dt
            particle.y += particle.vy * self.dt
        
        # Step 4: Handle wall collisions
        for particle in self.particles:
            handle_wall_collision(particle, self.box_size,
                                self.restitution, self.friction)
        
        # Step 5: Detect and resolve particle-particle collisions
        self._handle_particle_collisions()
        
        # Step 6: Recompute forces
        self._compute_forces()
        
        # Step 7: Final half-step velocity update
        for particle in self.particles:
            particle.vx += 0.5 * particle.ax * self.dt
            particle.vy += 0.5 * particle.ay * self.dt
    
    def _handle_particle_collisions(self) -> None:
        """
        Detect and resolve all particle-particle collisions using spatial grid.
        """
        
        # Rebuild spatial grid
        self.spatial_grid.clear()
        for particle in self.particles:
            self.spatial_grid.add_particle(particle)
        
        # Check collisions in neighboring cells
        checked_pairs = set()
        
        for particle in self.particles:
            neighbors = self.spatial_grid.get_neighbors(particle)
            
            for neighbor in neighbors:
                if particle is neighbor:
                    continue
                
                # Avoid duplicate checks
                pair = (id(particle), id(neighbor))
                if pair in checked_pairs:
                    continue
                checked_pairs.add(pair)
                
                # Attempt collision
                resolve_elastic_collision(particle, neighbor,
                                        self.restitution, self.friction)
    
    def step(self) -> None:
        """Execute one simulation step."""
        self._velocity_verlet_step()
        self.time += self.dt
        self.step_count += 1
    
    def run(self, num_steps: int, record_interval: int = 10) -> dict:
        """
        Run simulation for specified number of steps.
        
        Args:
            num_steps: Number of integration steps
            record_interval: Record data every N steps
        
        Returns:
            History dictionary with time series data
        """
        
        for i in range(num_steps):
            self.step()
            
            if (i + 1) % record_interval == 0:
                self._record_state()
        
        return self.history
    
    def _record_state(self) -> None:
        """Record current state to history."""
        
        self.history['time'].append(self.time)
        self.history['total_energy'].append(self.compute_total_energy())
        self.history['total_momentum'].append(
            np.linalg.norm(self.compute_total_momentum())
        )
        self.history['temperature'].append(self.compute_temperature())
        self.history['mean_speed'].append(self.compute_mean_speed())
        self.history['collision_count'].append(
            sum(p.collision_count for p in self.particles)
        )
    
    # ========================================================================
    # OBSERVABLE PROPERTIES
    # ========================================================================
    
    def compute_total_energy(self) -> float:
        """Total kinetic energy of all particles."""
        return sum(p.kinetic_energy for p in self.particles)
    
    def compute_total_momentum(self) -> np.ndarray:
        """Total momentum of all particles."""
        px = sum(p.mass * p.vx for p in self.particles)
        py = sum(p.mass * p.vy for p in self.particles)
        return np.array([px, py])
    
    def compute_temperature(self) -> float:
        """
        Temperature from equipartition theorem.
        
        In 2D: k_B * T = (1/2) * m * <v²>
        We set k_B = 1 for simplicity.
        """
        total_ke = self.compute_total_energy()
        n_dof = 2 * len(self.particles)  # 2D: x and y degrees of freedom
        return 2 * total_ke / n_dof if n_dof > 0 else 0.0
    
    def compute_mean_speed(self) -> float:
        """Mean speed of all particles."""
        if not self.particles:
            return 0.0
        return np.mean([p.speed for p in self.particles])
    
    def compute_velocity_distribution(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Histogram of particle speeds.
        
        Returns:
            (histogram, bin_edges)
        """
        speeds = np.array([p.speed for p in self.particles])
        return np.histogram(speeds, bins=bins)
    
    def compute_energy_distribution(self, bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """Histogram of kinetic energies."""
        energies = np.array([p.kinetic_energy for p in self.particles])
        return np.histogram(energies, bins=bins)
    
    def get_particle_positions(self) -> np.ndarray:
        """Get array of all particle positions."""
        return np.array([[p.x, p.y] for p in self.particles])
    
    def get_particle_velocities(self) -> np.ndarray:
        """Get array of all particle velocities."""
        return np.array([[p.vx, p.vy] for p in self.particles])
    
    def get_particle_masses(self) -> np.ndarray:
        """Get array of all particle masses."""
        return np.array([p.mass for p in self.particles])
    
    def get_particles_in_region(self, x_min: float, x_max: float,
                               y_min: float, y_max: float) -> List[Particle]:
        """Get particles within rectangular region."""
        return [p for p in self.particles
                if x_min <= p.x <= x_max and y_min <= p.y <= y_max]
    
    def get_particles_approaching_from_side(self, side: str = 'left',
                                           x_threshold: float = None) -> List[Particle]:
        """
        Get particles approaching from a side of the box.
        
        Args:
            side: 'left', 'right', 'bottom', or 'top'
            x_threshold: Optional custom threshold position
        
        Returns:
            List of particles moving toward that side
        """
        if x_threshold is None:
            if side == 'left':
                x_threshold = self.box_size * 0.25
            elif side == 'right':
                x_threshold = self.box_size * 0.75
        
        approaching = []
        
        if side == 'left':
            approaching = [p for p in self.particles
                         if p.x < x_threshold and p.vx < 0]
        elif side == 'right':
            approaching = [p for p in self.particles
                         if p.x > x_threshold and p.vx > 0]
        elif side == 'bottom':
            approaching = [p for p in self.particles
                         if p.y < x_threshold and p.vy < 0]
        elif side == 'top':
            approaching = [p for p in self.particles
                         if p.y > x_threshold and p.vy > 0]
        
        return approaching
    
    def __repr__(self) -> str:
        return (f"SimulationEngine(particles={len(self.particles)}, "
                f"box_size={self.box_size}, time={self.time:.4f})")
