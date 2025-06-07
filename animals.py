
import brain
import numpy as np
import random


class Food:

    COLOR = np.array([0.0, 1.0])  # Blue color for food

    INTENSITY_SCALING = 0.1  # Scaling factor for food intensity

    def __init__(self, position, min_energy=50.0, max_energy=150.0, 
            random_walk_rate=0.1):
        self.energy = random.uniform(min_energy, max_energy)
        self.position = position
        self.random_walk_rate = random_walk_rate

    def random_walk(self):
        self.position += np.random.normal(0, self.random_walk_rate, size=2)

class Predator:

    COLOR = np.array([1.0, 0.0])  # Red color for predator

    INTENSITY_SCALING = 0.1  # Scaling factor for intensity

    def __init__(self, range=1.0, random_walk_rate=0.1):
        self.position = np.array([0.0, 0.0])
        self.range_sq = range * range
        self.random_walk_rate = random_walk_rate

    def random_walk(self):
        """Randomly walk the predator within its range."""
        self.position += np.random.normal(0, self.random_walk_rate, size=2)

class Eye:

    def __init__(self):
        self.color = np.array([0.0, 0.0])  # Red Blue color
        self.position = np.array([0.0, 0.0])

    def add(self, target: np.ndarray, color: np.ndarray, intensity: float = 1.0):
        """Add a direction vector to the position."""
        r = target - self.position
        d_sq = np.dot(r, r) + 1e-6  # Adding a small value to avoid division by zero
        self.color += color * intensity / d_sq  # Inverse square law for light intensity
        self.color = np.clip(self.color, 0, 1.0)  # Ensure color values are within [0, 1]


class Animal:

    MIN_SIZE_TO_REPRODUCE = 4.0

    FEEDING_RATE_PER_SIZE = 0.5  # Food consumed per step

    DRAG = 0.05  # Drag coefficient for velocity damping

    MAX_AGE = 1000  # Maximum age of the animal

    START_ENERGY = 100.0  # Starting energy of the animal

    MIN_ENERGY_TO_GROW = 50.0  # Minimum energy required to grow

    ENERGY_CONSUMPTION_RATE_PER_SIZE = 1.0  # Energy consumption rate per step

    START_SIZE = 1.0  # Starting size of the animal

    GROWTH_PER_CONSUMED_ENERGY = 0.01  # Growth rate of the animal size per step

    ENERGY_CONSUMED_FOR_GROWTH = 0.1  # Energy consumed for growth per step

    EATING_RANGE = 0.2  # Range within which the animal can eat food

    PUSH_COST = 0.5  # Cost of pushing in terms of energy

    def __init__(self):
        """Initialize the animal with given parameters."""
        
        self.position = np.array([0.0, 0.0])
        self.velocity = np.array([0.0, 0.0])

        self.energy = Animal.START_ENERGY
        self.size = Animal.START_SIZE

        self.range_sq = Animal.EATING_RANGE * Animal.EATING_RANGE

        self.smell = 0.0

        self.eye = Eye()

        self.brain = brain.Brain(input_size=7, hidden_size=10, output_size=3)

    def can_reproduce(self) -> bool:
        """Check if the animal can reproduce based on its size."""
        return self.size > Animal.MIN_SIZE_TO_REPRODUCE
    
    def is_dead(self) -> bool:
        """Check if the animal is dead based on its size or energy."""
        return self.size < 1e-6 or self.energy < 1e-6
    
    def reproduce(self) -> 'Animal':
        """Create a new animal with half the size and energy of the parent."""
        
        child = Animal()
        child.brain = self.brain  # Share the same brain for simplicity
        return child
    
    def eat(self, food: Food):
        """Consume food and gain energy."""
        print(f"Eating food at position {food.position} with energy {food.energy}")
        food_energy_eaten = min(self.size * Animal.FEEDING_RATE_PER_SIZE, food.energy)
        self.energy += food_energy_eaten
        food.energy -= food_energy_eaten

        self.brain.learn(food.energy, learning_rate=0.01)  # Learn from eating food

    def grow(self):
        """Grow the bacteria by consuming energy and increasing size."""
        consumed_energy_for_size = Animal.ENERGY_CONSUMPTION_RATE_PER_SIZE * self.size
        excess_energy = self.energy - consumed_energy_for_size - Animal.MIN_ENERGY_TO_GROW
        consumed_energy_for_growth  = Animal.ENERGY_CONSUMED_FOR_GROWTH * excess_energy
        total_consumed_energy = consumed_energy_for_size + consumed_energy_for_growth

        self.size += Animal.GROWTH_PER_CONSUMED_ENERGY * consumed_energy_for_growth
        self.energy -= total_consumed_energy

    def push(self, amount: float, direction: np.ndarray):
        """Update the velocity based on the direction vector."""
        print(f"Pushing with amount {amount} in direction {direction}")
        self.velocity += direction * amount
        self.energy -= Animal.PUSH_COST * self.size * amount

    def update_food(self, food_sources: list[Food]) -> list:
        """Update the smell based on the food sources."""
        self.smell = 0.0
        for food in food_sources:
            r = food.position - self.position
            d_sq = np.dot(r, r) + 1e-6  # Adding a small value to avoid division by zero (same as assume food has some radius)
            
            inv_d_sq = 1.0 / d_sq  # Inverse square law for smell intensity
            self.smell += food.energy * inv_d_sq  
            self.eye.color += Food.COLOR * Food.INTENSITY_SCALING * food.energy * inv_d_sq

            if d_sq < self.range_sq:
                # If within range, consume food
                self.eat(food)

        # Normalize smell to a reasonable range
        self.smell = np.clip(self.smell, 0, 100)

    def update_predators(self, predators: list[Predator]):
        """Update the eye color based on the predators."""
        for predator in predators:
            r = predator.position - self.position
            d_sq = np.dot(r, r) + 1e-6
            self.eye.color += Predator.COLOR * Predator.INTENSITY_SCALING / d_sq

            if d_sq < predator.range_sq:
                # Eaten by predator
                self.size = 0

    def update_body(self):
        """Update the brain with the current state."""
        input_data = np.array([self.smell, self.energy, self.size, self.eye.color[0],
                               self.eye.color[1], self.position[0], self.position[1]])
        
        output = self.brain.forward(input_data)
        
        # Update velocity based on brain output
        self.push(output[0].item(), np.array([output[1].item(), output[2].item()])) 

