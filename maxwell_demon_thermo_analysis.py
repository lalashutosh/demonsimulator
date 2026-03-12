"""
Maxwell's Demon Simulation - Thermodynamic Analysis
=====================================================

Tools for:
- Computing thermodynamic observables (T, S, energy distributions)
- Tracking particle separation metrics
- Entropy calculations from phase-space distributions
- Statistical analysis of ensembles
"""

import numpy as np
from typing import Tuple, List, Dict, Optional
from scipy import stats


class ThermodynamicAnalyzer:
    """
    Compute and analyze thermodynamic properties of particle ensemble.
    """
    
    def __init__(self, simulation_engine):
        """
        Initialize analyzer with reference to simulation.
        
        Args:
            simulation_engine: SimulationEngine instance to analyze
        """
        self.engine = simulation_engine
    
    # ========================================================================
    # TEMPERATURE & ENERGY
    # ========================================================================
    
    def temperature(self) -> float:
        """
        Temperature from equipartition theorem.
        
        In 2D, with k_B = 1:
            k_B * T = (1/2) * m * <v²> = (total kinetic energy) / (# particles)
        
        Returns:
            Temperature (in simulation units where k_B = 1)
        """
        return self.engine.compute_temperature()
    
    def temperature_per_species(self, mass_groups: List[float]) -> Dict[float, float]:
        """
        Compute temperature for each mass species separately.
        
        Useful when you have different particle masses (e.g., light vs heavy).
        
        Args:
            mass_groups: List of masses to group by
        
        Returns:
            Dictionary mapping mass -> temperature
        """
        temps = {}
        
        for mass in mass_groups:
            particles = [p for p in self.engine.particles if p.mass == mass]
            
            if not particles:
                continue
            
            ke = sum(p.kinetic_energy for p in particles)
            n_dof = 2 * len(particles)
            temps[mass] = 2 * ke / n_dof if n_dof > 0 else 0.0
        
        return temps
    
    def temperature_regional(self, x_min: float, x_max: float,
                            y_min: float, y_max: float) -> float:
        """
        Compute temperature in a specific region of the box.
        
        Args:
            x_min, x_max, y_min, y_max: Region bounds
        
        Returns:
            Temperature in that region
        """
        particles = self.engine.get_particles_in_region(x_min, x_max, y_min, y_max)
        
        if not particles:
            return 0.0
        
        ke = sum(p.kinetic_energy for p in particles)
        n_dof = 2 * len(particles)
        
        return 2 * ke / n_dof if n_dof > 0 else 0.0
    
    def mean_kinetic_energy(self) -> float:
        """Mean kinetic energy per particle."""
        n = len(self.engine.particles)
        return self.engine.compute_total_energy() / n if n > 0 else 0.0
    
    # ========================================================================
    # DISTRIBUTIONS (Velocity, Energy, Speed)
    # ========================================================================
    
    def speed_distribution(self, bins: int = 50, weights: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Histogram of particle speeds.
        
        Args:
            bins: Number of bins
            weights: If True, normalize to probability density
        
        Returns:
            (histogram, bin_centers)
        """
        speeds = np.array([p.speed for p in self.engine.particles])
        hist, edges = np.histogram(speeds, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        if weights:
            # Normalize to probability density
            dv = edges[1] - edges[0]
            hist = hist / (hist.sum() * dv)
        
        return hist, bin_centers
    
    def velocity_distribution_2d(self, bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        2D histogram of velocity components (vx vs vy).
        
        Args:
            bins: Number of bins per dimension
        
        Returns:
            (histogram, x_edges, y_edges)
        """
        vx = np.array([p.vx for p in self.engine.particles])
        vy = np.array([p.vy for p in self.engine.particles])
        
        hist, xedges, yedges = np.histogram2d(vx, vy, bins=bins)
        
        return hist, xedges, yedges
    
    def kinetic_energy_distribution(self, bins: int = 50,
                                   weights: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Histogram of kinetic energies.
        
        Args:
            bins: Number of bins
            weights: If True, normalize to probability density
        
        Returns:
            (histogram, bin_centers)
        """
        energies = np.array([p.kinetic_energy for p in self.engine.particles])
        hist, edges = np.histogram(energies, bins=bins)
        bin_centers = (edges[:-1] + edges[1:]) / 2
        
        if weights:
            dE = edges[1] - edges[0]
            hist = hist / (hist.sum() * dE)
        
        return hist, bin_centers
    
    def maxwell_boltzmann_speed(self, T: float, m: float = 1.0,
                               speeds: np.ndarray = None) -> np.ndarray:
        """
        Maxwell-Boltzmann speed distribution for comparison with simulation.
        
        P(v) = 4π * (m / 2πk_B*T)^(3/2) * v² * exp(-m*v² / 2k_B*T)
        
        In 2D:
        P(v) = (m / πk_B*T) * v * exp(-m*v² / 2k_B*T)
        
        Args:
            T: Temperature
            m: Particle mass
            speeds: Speed values at which to evaluate PDF
        
        Returns:
            Probability density at given speeds
        """
        
        if speeds is None:
            speeds = np.linspace(0, 3*np.sqrt(2*T), 100)
        
        # 2D Maxwell-Boltzmann
        sigma = np.sqrt(T / m)
        A = m / (np.pi * T)
        pdf = A * speeds * np.exp(-m * speeds**2 / (2 * T))
        
        return pdf
    
    # ========================================================================
    # ENTROPY & INFORMATION
    # ========================================================================
    
    def entropy_from_speed_distribution(self, bins: int = 50) -> float:
        """
        Shannon entropy computed from speed distribution histogram.
        
        S = -Σ p_i * ln(p_i)
        
        Higher entropy = more uniform distribution
        Lower entropy = more concentrated distribution
        
        Args:
            bins: Number of bins for histogram
        
        Returns:
            Shannon entropy
        """
        hist, _ = self.speed_distribution(bins=bins, weights=False)
        
        # Normalize to probability
        p = hist / hist.sum()
        
        # Remove zero bins (log(0) undefined)
        p = p[p > 0]
        
        return -np.sum(p * np.log(p))
    
    def entropy_from_phase_space(self, n_bins_x: int = 10, n_bins_v: int = 10) -> float:
        """
        Entropy from phase space (position + velocity) distribution.
        
        More complete picture of disorder than speed distribution alone.
        
        Args:
            n_bins_x: Bins for position
            n_bins_v: Bins for velocity
        
        Returns:
            Phase-space entropy
        """
        # Position distribution
        positions = self.engine.get_particle_positions()
        pos_hist, _, _ = np.histogram2d(positions[:, 0], positions[:, 1],
                                       bins=n_bins_x)
        
        # Velocity distribution
        velocities = self.engine.get_particle_velocities()
        vel_hist, _, _ = np.histogram2d(velocities[:, 0], velocities[:, 1],
                                       bins=n_bins_v)
        
        # Combined phase space
        p_pos = pos_hist.flatten() / pos_hist.sum()
        p_vel = vel_hist.flatten() / vel_hist.sum()
        
        # Entropy from each
        p_pos = p_pos[p_pos > 0]
        p_vel = p_vel[p_vel > 0]
        
        S_pos = -np.sum(p_pos * np.log(p_pos)) if len(p_pos) > 0 else 0
        S_vel = -np.sum(p_vel * np.log(p_vel)) if len(p_vel) > 0 else 0
        
        return S_pos + S_vel
    
    # ========================================================================
    # PARTICLE SEPARATION METRICS
    # ========================================================================
    
    def separation_by_mass(self, left_region: float = None,
                          right_region: float = None) -> Dict[str, float]:
        """
        Measure how effectively particles have separated by mass.
        
        Useful for "different species" version of Maxwell's demon.
        
        Args:
            left_region: x-position defining left boundary (default: box_size/3)
            right_region: x-position defining right boundary (default: 2*box_size/3)
        
        Returns:
            Dictionary with separation metrics
        """
        
        if left_region is None:
            left_region = self.engine.box_size / 3
        if right_region is None:
            right_region = 2 * self.engine.box_size / 3
        
        # Get particles in each region
        left_particles = self.engine.get_particles_in_region(
            0, left_region, 0, self.engine.box_size
        )
        right_particles = self.engine.get_particles_in_region(
            right_region, self.engine.box_size, 0, self.engine.box_size
        )
        middle_particles = self.engine.get_particles_in_region(
            left_region, right_region, 0, self.engine.box_size
        )
        
        # Total mass in each region
        mass_left = sum(p.mass for p in left_particles)
        mass_right = sum(p.mass for p in right_particles)
        mass_middle = sum(p.mass for p in middle_particles)
        mass_total = mass_left + mass_right + mass_middle
        
        # Average mass in each region
        avg_mass_left = mass_left / len(left_particles) if left_particles else 0
        avg_mass_right = mass_right / len(right_particles) if right_particles else 0
        avg_mass_middle = mass_middle / len(middle_particles) if middle_particles else 0
        
        # Measure of separation: how far from equilibrium?
        # In equilibrium: same avg mass everywhere
        mean_mass = sum(p.mass for p in self.engine.particles) / len(self.engine.particles)
        
        separation_metric = (
            abs(avg_mass_left - mean_mass) +
            abs(avg_mass_right - mean_mass) +
            abs(avg_mass_middle - mean_mass)
        )
        
        return {
            'mass_left': mass_left,
            'mass_right': mass_right,
            'mass_middle': mass_middle,
            'mass_total': mass_total,
            'avg_mass_left': avg_mass_left,
            'avg_mass_right': avg_mass_right,
            'avg_mass_middle': avg_mass_middle,
            'separation_metric': separation_metric,
            'num_left': len(left_particles),
            'num_right': len(right_particles),
            'num_middle': len(middle_particles),
        }
    
    def separation_by_temperature(self, left_region: float = None,
                                 right_region: float = None) -> Dict[str, float]:
        """
        Measure temperature difference between regions (classic demon effect).
        
        Args:
            left_region: x-position defining left boundary
            right_region: x-position defining right boundary
        
        Returns:
            Dictionary with temperature metrics
        """
        
        if left_region is None:
            left_region = self.engine.box_size / 3
        if right_region is None:
            right_region = 2 * self.engine.box_size / 3
        
        T_left = self.temperature_regional(0, left_region, 0, self.engine.box_size)
        T_middle = self.temperature_regional(left_region, right_region, 0, self.engine.box_size)
        T_right = self.temperature_regional(right_region, self.engine.box_size, 0, self.engine.box_size)
        
        T_total = self.temperature()
        
        # Measure of separation
        delta_T = abs(T_left - T_right)
        
        return {
            'T_left': T_left,
            'T_middle': T_middle,
            'T_right': T_right,
            'T_total': T_total,
            'delta_T': delta_T,
            'ratio_T_left_right': T_left / T_right if T_right > 0 else float('inf'),
        }
    
    def separation_by_density(self, left_region: float = None,
                             right_region: float = None) -> Dict[str, float]:
        """
        Measure particle density difference between regions.
        
        Args:
            left_region: x-position defining left boundary
            right_region: x-position defining right boundary
        
        Returns:
            Dictionary with density metrics
        """
        
        if left_region is None:
            left_region = self.engine.box_size / 3
        if right_region is None:
            right_region = 2 * self.engine.box_size / 3
        
        n_left = len(self.engine.get_particles_in_region(
            0, left_region, 0, self.engine.box_size
        ))
        n_right = len(self.engine.get_particles_in_region(
            right_region, self.engine.box_size, 0, self.engine.box_size
        ))
        n_middle = len(self.engine.get_particles_in_region(
            left_region, right_region, 0, self.engine.box_size
        ))
        n_total = len(self.engine.particles)
        
        # Densities (particles per unit volume)
        # Areas: left and right each have area = left_region * box_size
        area_left = left_region * self.engine.box_size
        area_middle = (right_region - left_region) * self.engine.box_size
        area_right = (self.engine.box_size - right_region) * self.engine.box_size
        
        density_left = n_left / area_left if area_left > 0 else 0
        density_middle = n_middle / area_middle if area_middle > 0 else 0
        density_right = n_right / area_right if area_right > 0 else 0
        density_avg = n_total / (self.engine.box_size ** 2)
        
        return {
            'n_left': n_left,
            'n_middle': n_middle,
            'n_right': n_right,
            'density_left': density_left,
            'density_middle': density_middle,
            'density_right': density_right,
            'density_avg': density_avg,
            'delta_n': abs(n_left - n_right),
            'ratio_density_left_right': density_left / density_right if density_right > 0 else float('inf'),
        }
    
    # ========================================================================
    # STATISTICS & QUALITY CHECKS
    # ========================================================================
    
    def energy_conservation_error(self, initial_energy: float) -> float:
        """
        Fractional error in total energy relative to initial.
        
        Good check that numerical integration is stable.
        Should be << 1% for healthy simulation.
        
        Args:
            initial_energy: Total energy at t=0
        
        Returns:
            Fractional error
        """
        current_energy = self.engine.compute_total_energy()
        return abs(current_energy - initial_energy) / initial_energy if initial_energy > 0 else 0.0
    
    def momentum_conservation_error(self, initial_momentum: np.ndarray) -> float:
        """
        Magnitude of change in total momentum.
        
        Should be ~0 if no external forces (other than boundaries).
        
        Args:
            initial_momentum: Total momentum at t=0
        
        Returns:
            Magnitude of momentum change
        """
        current_momentum = self.engine.compute_total_momentum()
        return np.linalg.norm(current_momentum - initial_momentum)
    
    def collision_rate(self) -> float:
        """
        Average collision rate (collisions per particle per unit time).
        
        Useful for understanding how often particles are interacting.
        """
        total_collisions = sum(p.collision_count for p in self.engine.particles)
        time_elapsed = max(self.engine.time, 1e-10)
        n_particles = len(self.engine.particles)
        
        return total_collisions / (n_particles * time_elapsed)
    
    def pair_correlation_function(self, max_distance: float = None,
                                  bins: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Radial pair correlation function g(r).
        
        Measures clustering: g(r) = 1 for random distribution,
                           g(r) > 1 for clustering,
                           g(r) < 1 for repulsion.
        
        Args:
            max_distance: Maximum distance to compute (default: box_size/2)
            bins: Number of bins
        
        Returns:
            (g(r), r_values)
        """
        
        if max_distance is None:
            max_distance = self.engine.box_size / 2
        
        positions = self.engine.get_particle_positions()
        n = len(positions)
        
        # Compute all pairwise distances
        distances = []
        for i in range(n):
            for j in range(i+1, n):
                dx = positions[i, 0] - positions[j, 0]
                dy = positions[i, 1] - positions[j, 1]
                
                # Periodic boundaries (optional)
                dx = min(abs(dx), self.engine.box_size - abs(dx))
                dy = min(abs(dy), self.engine.box_size - abs(dy))
                
                dist = np.sqrt(dx**2 + dy**2)
                distances.append(dist)
        
        distances = np.array(distances)
        
        # Histogram
        hist, edges = np.histogram(distances, bins=bins, range=(0, max_distance))
        r = (edges[:-1] + edges[1:]) / 2
        dr = edges[1] - edges[0]
        
        # Normalize by random distribution density
        box_area = self.engine.box_size ** 2
        rho = n / box_area  # number density
        
        # For each shell of radius r and thickness dr:
        # Volume (area in 2D) = 2πr*dr
        # Expected count in random distribution = rho * 2πr*dr
        
        g = hist / (2 * np.pi * r * dr * rho + 1e-10)
        
        return g, r
    
    def __repr__(self) -> str:
        return f"ThermodynamicAnalyzer(T={self.temperature():.4f}, time={self.engine.time:.4f})"
