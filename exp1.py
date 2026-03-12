import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import random

# Initialize Pygame
pygame.init()
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Maxwell's Demon Simulation")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
RED = (255, 0, 0)

# Pymunk space
space = pymunk.Space()
space.gravity = (0.0, 0.0)
draw_options = pymunk.pygame_util.DrawOptions(screen)

# Configuration
num_particles = 100
particle_radius = 8
partition_x = screen_width // 2
demon_threshold = 10.0  # Energy threshold for demon logic

class Particle:
    def __init__(self, space, x, y, radius, mass):
        self.mass = mass
        self.radius = radius
        
        # Create body
        moment = pymunk.moment_for_circle(mass, 0, radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = x, y
        self.body.velocity = random.uniform(-200, 200), random.uniform(-200, 200)
        
        # Create shape
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.98
        self.shape.friction = 0.1
        
        space.add(self.body, self.shape)

def create_container(space, width, height):
    """Create container walls"""
    walls = [
        pymunk.Segment(space.static_body, (0, 0), (width, 0), 2),  # Bottom
        pymunk.Segment(space.static_body, (width, 0), (width, height), 2),  # Right
        pymunk.Segment(space.static_body, (width, height), (0, height), 2),  # Top
        pymunk.Segment(space.static_body, (0, height), (0, 0), 2),  # Left
    ]
    
    for wall in walls:
        wall.elasticity = 0.95
        wall.friction = 0.1
    
    space.add(*walls)

def demon_logic(space, partition_x, threshold):
    """Maxwell's Demon logic - sorts particles by energy"""
    for body in space.bodies:
        if isinstance(body, pymunk.Body) and body not in space.static_body.bodies:
            # Calculate kinetic energy
            energy = 0.5 * body.mass * body.velocity.length_squared
            
            # Demon decision
            if body.position.x < partition_x:  # Left side
                if energy > threshold:
                    # Move high-energy particle to right
                    body.position += (1, 0)
            else:  # Right side
                if energy < threshold:
                    # Move low-energy particle to left
                    body.position -= (1, 0)

def calculate_thermodynamic_properties(space, partition_x):
    """Calculate temperature, pressure, and entropy"""
    left_particles = []
    right_particles = []
    
    # Separate particles
    for body in space.bodies:
        if isinstance(body, pymunk.Body) and body not in space.static_body.bodies:
            if body.position.x < partition_x:
                left_particles.append(body)
            else:
                right_particles.append(body)
    
    def calculate_properties(particles):
        if not particles:
            return 0, 0, 0
        
        # Calculate average kinetic energy (temperature)
        total_energy = sum(0.5 * p.mass * p.velocity.length_squared for p in particles)
        avg_energy = total_energy / len(particles)
        
        # Simplified temperature (in arbitrary units)
        temperature = avg_energy
        
        # Pressure (simplified - based on momentum transfer)
        pressure = sum(p.mass * p.velocity.length for p in particles) / (2 * screen_width)
        
        # Entropy (based on energy distribution)
        energies = [0.5 * p.mass * p.velocity.length_squared for p in particles]
        energy_dist = np.array(energies) / sum(energies)
        entropy = -np.sum(energy_dist * np.log(energy_dist + 1.001))
        
        return temperature, pressure, entropy
    
    left_temp, left_press, left_ent = calculate_properties(left_particles)
    right_temp, right_press, right_ent = calculate_properties(right_particles)
    
    return {
        'left': {'temp': left_temp, 'press': left_press, 'ent': left_ent},
        'right': {'temp': right_temp, 'press': right_press, 'ent': right_ent}
    }

def display_text(screen, properties, font):
    """Display thermodynamic properties on screen"""
    text = font.render(f"Left: T={properties['left']['temp']:.2f}  P={properties['left']['press']:.2f}", 
                      True, BLACK)
    screen.blit(text, (10, 10))
    
    text = font.render(f"Right: T={properties['right']['temp']:.2f}  P={properties['right']['press']:.2f}", 
                      True, BLACK)
    screen.blit(text, (10, 40))
    
    text = font.render(f"Entropy: L={properties['left']['ent']:.2f}  R={properties['right']['ent']:.2f}", 
                      True, BLACK)
    screen.blit(text, (10, 70))

def main():
    # Create container
    create_container(space, screen_width, screen_height)
    
    # Create partition (demon's door)
    partition = pymunk.Segment(space.static_body, 
                              (partition_x, 0), 
                              (partition_x, screen_height), 
                              2)
    partition.friction = 0.0
    space.add(partition)
    
    # Create particles
    particles = []
    for _ in range(num_particles):
        x = random.randint(particle_radius, screen_width - particle_radius)
        y = random.randint(particle_radius, screen_height - particle_radius)
        mass = random.uniform(1, 3)
        particles.append(Particle(space, x, y, particle_radius, mass))
    
    # Font for display
    font = pygame.font.Font(None, 36)
    
    clock = pygame.time.Clock()
    running = True
    
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Clear screen
        screen.fill(WHITE)
        
        # Update physics
        space.step(1/60)
        
        # Apply Maxwell's Demon logic
        demon_logic(space, partition_x, demon_threshold)
        
        # Calculate properties
        properties = calculate_thermodynamic_properties(space, partition_x)
        
        # Draw partition
        pygame.draw.line(screen, BLACK, (partition_x, 0), (partition_x, screen_height), 3)
        
        # Draw all shapes
        space.debug_draw(draw_options)
        
        # Display text
        display_text(screen, properties, font)
        
        # Update display
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()