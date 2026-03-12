"""
Maxwell's Demon Simulation - Demon Control Strategies
=====================================================

Multiple implementations of the Maxwell's Demon mechanism:

1. VelocityThresholdDemon: Simplest - opens gate if v > threshold
2. MeasurementDemon: Measures particle properties, then decides
3. EnergyAccountantDemon: Tracks measurement cost (advanced)
4. MixingDemon: Can mix particles or prevent mixing (experimental)
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional, Tuple
from dataclasses import dataclass


# ============================================================================
# BASE DEMON CLASS
# ============================================================================

class MaxwellDemon(ABC):
    """Abstract base class for demon strategies."""
    
    def __init__(self, simulation_engine):
        """
        Initialize demon with reference to simulation.
        
        Args:
            simulation_engine: SimulationEngine instance to control
        """
        self.engine = simulation_engine
        self.step_count = 0
        self.gate_open_count = 0
        self.history = {
            'gate_open': [],
            'particles_passed': [],
            'work_done': [],
            'separation_metric': [],
        }
    
    @abstractmethod
    def step(self) -> Tuple[bool, List]:
        """
        Execute one demon step.
        
        Returns:
            (gate_is_open, particles_that_passed_through)
        """
        pass
    
    @abstractmethod
    def reset(self):
        """Reset demon state."""
        pass
    
    def record_state(self, gate_open: bool, passed_particles: List,
                    work: float = 0, separation: float = 0):
        """Record step to history."""
        self.history['gate_open'].append(gate_open)
        self.history['particles_passed'].append(len(passed_particles))
        self.history['work_done'].append(work)
        self.history['separation_metric'].append(separation)
        self.step_count += 1
        if gate_open:
            self.gate_open_count += 1


# ============================================================================
# SIMPLE VELOCITY THRESHOLD DEMON
# ============================================================================

class VelocityThresholdDemon(MaxwellDemon):
    """
    Simplest demon: Opens gate if particle approaching from one side
    has speed > threshold.
    
    This is the "textbook" Maxwell's demon in computational form.
    Demonstrates the core principle without measurement overhead complexity.
    """
    
    def __init__(self, simulation_engine,
                 side: str = 'left',
                 velocity_threshold: float = 0.5,
                 gate_x: Optional[float] = None,
                 gate_width: float = 0.5):
        """
        Initialize velocity threshold demon.
        
        Args:
            simulation_engine: SimulationEngine instance
            side: Which side to separate ('left' or 'right')
            velocity_threshold: Gate opens if |v| > this value
            gate_x: x-position of gate (default: center)
            gate_width: Width of region to check for approaching particles
        """
        
        super().__init__(simulation_engine)
        
        self.side = side
        self.velocity_threshold = velocity_threshold
        self.gate_x = gate_x or (simulation_engine.box_size / 2)
        self.gate_width = gate_width
        
        self.gate_open = False
        self.last_frame_particles = set()
    
    def get_approaching_particles(self) -> List:
        """
        Get particles in gate region approaching from specified side.
        
        Returns:
            List of particles approaching from the configured side
        """
        
        gate_region = self.gate_width
        x_min = self.gate_x - gate_region / 2
        x_max = self.gate_x + gate_region / 2
        
        approaching = []
        
        if self.side == 'left':
            # Particles from left approaching gate moving right
            approaching = [p for p in self.engine.particles
                         if x_min <= p.x <= self.gate_x and p.vx > 0]
        
        elif self.side == 'right':
            # Particles from right approaching gate moving left
            approaching = [p for p in self.engine.particles
                         if self.gate_x <= p.x <= x_max and p.vx < 0]
        
        return approaching
    
    def step(self) -> Tuple[bool, List]:
        """
        Execute one demon step: check approaching particles and open/close gate.
        
        Returns:
            (gate_is_open, particles_that_moved_through_gate)
        """
        
        approaching = self.get_approaching_particles()
        
        # Decision logic: open gate if ANY particle has v > threshold
        fast_particles = [p for p in approaching
                         if p.speed > self.velocity_threshold]
        
        self.gate_open = len(fast_particles) > 0
        
        # Track which particles "passed through" this step
        passed = fast_particles
        
        # Record
        self.record_state(self.gate_open, passed)
        
        return self.gate_open, passed
    
    def reset(self):
        """Reset demon state."""
        self.gate_open = False
        self.step_count = 0
        self.gate_open_count = 0
        self.history = {
            'gate_open': [],
            'particles_passed': [],
            'work_done': [],
            'separation_metric': [],
        }
    
    def __repr__(self) -> str:
        return (f"VelocityThresholdDemon(side={self.side}, "
                f"v_threshold={self.velocity_threshold:.3f}, "
                f"gate_open={self.gate_open})")


# ============================================================================
# MEASUREMENT DEMON (AWARE OF ENERGY COST)
# ============================================================================

class MeasurementDemon(MaxwellDemon):
    """
    Demon that performs measurement, then admits based on result.
    
    Key insight: Measurement has energy cost (Szilard's version).
    This demon tracks that cost.
    """
    
    def __init__(self, simulation_engine,
                 side: str = 'left',
                 energy_threshold: float = 1.0,
                 gate_x: Optional[float] = None,
                 gate_width: float = 0.5,
                 measurement_cost: float = 0.1):
        """
        Initialize measurement demon.
        
        Args:
            simulation_engine: SimulationEngine instance
            side: Which side to separate
            energy_threshold: Gate opens if KE > this value
            gate_x: x-position of gate
            gate_width: Width of sensing region
            measurement_cost: Energy cost per measurement (in simulation units)
        """
        
        super().__init__(simulation_engine)
        
        self.side = side
        self.energy_threshold = energy_threshold
        self.gate_x = gate_x or (simulation_engine.box_size / 2)
        self.gate_width = gate_width
        self.measurement_cost = measurement_cost
        
        self.gate_open = False
        self.total_measurement_cost = 0.0
        self.total_work_extracted = 0.0
    
    def measure_approaching_particles(self) -> List[Tuple]:
        """
        Measure (sense) approaching particles' kinetic energies.
        
        Returns:
            List of (particle, kinetic_energy) tuples
        """
        
        gate_region = self.gate_width
        x_min = self.gate_x - gate_region / 2
        x_max = self.gate_x + gate_region / 2
        
        approaching = []
        
        if self.side == 'left':
            approaching = [p for p in self.engine.particles
                         if x_min <= p.x <= self.gate_x and p.vx > 0]
        elif self.side == 'right':
            approaching = [p for p in self.engine.particles
                         if self.gate_x <= p.x <= x_max and p.vx < 0]
        
        # "Measure" each particle
        measurements = [(p, p.kinetic_energy) for p in approaching]
        
        return measurements
    
    def step(self) -> Tuple[bool, List]:
        """
        Execute one demon step with energy-aware gating.
        
        Returns:
            (gate_is_open, particles_that_passed)
        """
        
        # Measure approaching particles
        measurements = self.measure_approaching_particles()
        
        # Count measurement cost
        measurement_cost_this_step = len(measurements) * self.measurement_cost
        self.total_measurement_cost += measurement_cost_this_step
        
        # Decision: admit if KE > threshold
        high_energy = [p for p, ke in measurements
                      if ke > self.energy_threshold]
        
        self.gate_open = len(high_energy) > 0
        
        # Record
        self.record_state(self.gate_open, high_energy,
                         work=-measurement_cost_this_step)
        
        return self.gate_open, high_energy
    
    def reset(self):
        """Reset demon state."""
        self.gate_open = False
        self.step_count = 0
        self.gate_open_count = 0
        self.total_measurement_cost = 0.0
        self.total_work_extracted = 0.0
        self.history = {
            'gate_open': [],
            'particles_passed': [],
            'work_done': [],
            'separation_metric': [],
        }
    
    def net_work(self) -> float:
        """
        Net work extracted by demon.
        
        = work from separation - measurement cost
        
        Returns:
            Net work (positive = useful work extracted)
        """
        return self.total_work_extracted - self.total_measurement_cost
    
    def __repr__(self) -> str:
        return (f"MeasurementDemon(side={self.side}, "
                f"E_threshold={self.energy_threshold:.3f}, "
                f"net_work={self.net_work():.3f})")


# ============================================================================
# DENSITY SEPARATION DEMON (Admit to balance pressure)
# ============================================================================

class DensityDemon(MaxwellDemon):
    """
    Demon that measures local density and maintains differential.
    
    Opens gate if local density on approach side is higher than on other side.
    """
    
    def __init__(self, simulation_engine,
                 side: str = 'left',
                 gate_x: Optional[float] = None,
                 gate_width: float = 0.5,
                 density_threshold: float = 0.5):
        """
        Initialize density-aware demon.
        
        Args:
            simulation_engine: SimulationEngine instance
            side: Which side to separate
            gate_x: x-position of gate
            gate_width: Width of sensing region on each side
            density_threshold: Density ratio needed to open gate
        """
        
        super().__init__(simulation_engine)
        
        self.side = side
        self.gate_x = gate_x or (simulation_engine.box_size / 2)
        self.gate_width = gate_width
        self.density_threshold = density_threshold
        
        self.gate_open = False
    
    def get_local_density(self, region_side: str) -> float:
        """
        Compute particle density in region on specified side.
        
        Args:
            region_side: 'left' or 'right'
        
        Returns:
            Particle density (particles per unit volume)
        """
        
        if region_side == 'left':
            x_min = 0
            x_max = self.gate_x - self.gate_width
        else:  # right
            x_min = self.gate_x + self.gate_width
            x_max = self.engine.box_size
        
        particles_in_region = self.engine.get_particles_in_region(
            x_min, x_max, 0, self.engine.box_size
        )
        
        area = (x_max - x_min) * self.engine.box_size
        density = len(particles_in_region) / area if area > 0 else 0
        
        return density
    
    def step(self) -> Tuple[bool, List]:
        """Execute one demon step based on density."""
        
        # Get densities on both sides
        density_left = self.get_local_density('left')
        density_right = self.get_local_density('right')
        
        # Decision: open gate if approaching side is denser
        if self.side == 'left':
            # Open if left is denser than right
            self.gate_open = (density_left > density_right * self.density_threshold)
        else:
            # Open if right is denser than left
            self.gate_open = (density_right > density_left * self.density_threshold)
        
        # Particles that can pass
        approaching = self.get_approaching_particles()
        passed = approaching if self.gate_open else []
        
        self.record_state(self.gate_open, passed)
        
        return self.gate_open, passed
    
    def get_approaching_particles(self) -> List:
        """Get particles in gate region approaching from specified side."""
        
        gate_region = self.gate_width
        x_min = self.gate_x - gate_region / 2
        x_max = self.gate_x + gate_region / 2
        
        if self.side == 'left':
            approaching = [p for p in self.engine.particles
                         if x_min <= p.x <= self.gate_x and p.vx > 0]
        else:
            approaching = [p for p in self.engine.particles
                         if self.gate_x <= p.x <= x_max and p.vx < 0]
        
        return approaching
    
    def reset(self):
        """Reset demon state."""
        self.gate_open = False
        self.step_count = 0
        self.gate_open_count = 0
        self.history = {
            'gate_open': [],
            'particles_passed': [],
            'work_done': [],
            'separation_metric': [],
        }
    
    def __repr__(self) -> str:
        return (f"DensityDemon(side={self.side}, "
                f"gate_x={self.gate_x:.2f})")


# ============================================================================
# HYSTERESIS DEMON (Adaptive threshold with memory)
# ============================================================================

class HysteresisDemon(MaxwellDemon):
    """
    Demon with hysteresis: higher threshold to open gate, lower to close.
    
    Reduces jitter from noise and makes demon more "decisive".
    """
    
    def __init__(self, simulation_engine,
                 side: str = 'left',
                 open_threshold: float = 0.6,
                 close_threshold: float = 0.4,
                 gate_x: Optional[float] = None,
                 gate_width: float = 0.5):
        """
        Initialize hysteresis demon.
        
        Args:
            simulation_engine: SimulationEngine instance
            side: Which side to separate
            open_threshold: Velocity needed to open gate
            close_threshold: Velocity below which gate closes
            gate_x: x-position of gate
            gate_width: Width of sensing region
        """
        
        super().__init__(simulation_engine)
        
        self.side = side
        self.open_threshold = open_threshold
        self.close_threshold = close_threshold
        self.gate_x = gate_x or (simulation_engine.box_size / 2)
        self.gate_width = gate_width
        
        self.gate_open = False
    
    def get_max_approaching_speed(self) -> float:
        """Maximum speed among approaching particles."""
        
        gate_region = self.gate_width
        x_min = self.gate_x - gate_region / 2
        x_max = self.gate_x + gate_region / 2
        
        approaching = []
        
        if self.side == 'left':
            approaching = [p for p in self.engine.particles
                         if x_min <= p.x <= self.gate_x and p.vx > 0]
        else:
            approaching = [p for p in self.engine.particles
                         if self.gate_x <= p.x <= x_max and p.vx < 0]
        
        if not approaching:
            return 0.0
        
        return max(p.speed for p in approaching)
    
    def step(self) -> Tuple[bool, List]:
        """Execute one demon step with hysteresis."""
        
        max_speed = self.get_max_approaching_speed()
        
        # Hysteresis logic
        if self.gate_open:
            # Gate is open: close if speed drops below close_threshold
            if max_speed < self.close_threshold:
                self.gate_open = False
        else:
            # Gate is closed: open if speed exceeds open_threshold
            if max_speed > self.open_threshold:
                self.gate_open = True
        
        # Particles that pass
        approaching = self.get_approaching_particles()
        passed = approaching if self.gate_open else []
        
        self.record_state(self.gate_open, passed)
        
        return self.gate_open, passed
    
    def get_approaching_particles(self) -> List:
        """Get approaching particles."""
        
        gate_region = self.gate_width
        x_min = self.gate_x - gate_region / 2
        x_max = self.gate_x + gate_region / 2
        
        if self.side == 'left':
            approaching = [p for p in self.engine.particles
                         if x_min <= p.x <= self.gate_x and p.vx > 0]
        else:
            approaching = [p for p in self.engine.particles
                         if self.gate_x <= p.x <= x_max and p.vx < 0]
        
        return approaching
    
    def reset(self):
        """Reset demon state."""
        self.gate_open = False
        self.step_count = 0
        self.gate_open_count = 0
        self.history = {
            'gate_open': [],
            'particles_passed': [],
            'work_done': [],
            'separation_metric': [],
        }
    
    def __repr__(self) -> str:
        return (f"HysteresisDemon(side={self.side}, "
                f"open={self.open_threshold:.3f}, "
                f"close={self.close_threshold:.3f}, "
                f"state={'OPEN' if self.gate_open else 'CLOSED'})")
