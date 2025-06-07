
from animals import *

import json
import numpy as np
import sys

class Simulation:

    BASE_SCORE = 100.0

    SCORE_MULTIPLIER = 100.0

    FOOD_RANDOM_WALK_RATE = 0.5

    FOOD_RATE = 0.1

    FOOD_MAX_ENERGY = 10.0

    FOOD_MIN_ENERGY = 1.0

    SIZE = 1000.0

    def __init__(self):
        self.current_generation = Animal()
        self.food_sources = []
        self.predators = []
        self.food_reserve = 0
        self.generations = []

    def reset_with_new_generation(self):
        """Reset the simulation."""
        self.food_sources = []
        self.predators = []
        self.food_reserve = 0

    def create_new_food(self):
        """Create new food sources."""
        while self.food_reserve > 1:
            food = Food(position=np.random.uniform(-Simulation.SIZE, Simulation.SIZE, size=2),
                        min_energy=Simulation.FOOD_MIN_ENERGY,
                        max_energy=Simulation.FOOD_MAX_ENERGY,
                        random_walk_rate=Simulation.FOOD_RANDOM_WALK_RATE)
            self.food_sources.append(food)
            self.food_reserve -= 1.0
        
    def step(self):
        """Perform a step in the environment."""
        # NOTE: dt is assumed to be 1 for simplicity, as the simulation is discrete

        # Update positions of animal, food sources, and predators

        # Update position based on velocity
        self.current_generation.position += self.current_generation.velocity

        # Dampen the velocity with drag
        self.current_generation.velocity *= (1 - self.current_generation.DRAG)
        
        # Check boundaries
        self.current_generation.position = np.clip(self.current_generation.position, -Simulation.SIZE, Simulation.SIZE)
        
        for food in self.food_sources:
            # Random walk for food sources
            food.random_walk()
            # Check boundaries for food sources
            food.position = np.clip(food.position, -Simulation.SIZE, Simulation.SIZE)

        for predator in self.predators:
            # Random walk for predators
            predator.random_walk()
            # Check boundaries for predators
            predator.position = np.clip(predator.position, -Simulation.SIZE, Simulation.SIZE)

        # Update the animal senses with the current environment
        self.current_generation.update_food(self.food_sources)
        self.current_generation.update_predators(self.predators)

        # Update the animal's body based on the brain's output
        self.current_generation.update_body()

        for food in self.food_sources:
            if food.energy < 1e-6:
                # Remove food sources that have been consumed
                self.food_sources.remove(food)

        # Add food sources based on the food rate
        self.food_reserve += Simulation.FOOD_RATE
        self.create_new_food()

        self.current_generation.grow()

    def final_score(self, num_steps: int) -> float:
        """Calculate the score of the simulation based on the animal's energy."""
        
        if self.current_generation.size <= 0:
            return Simulation.BASE_SCORE* (num_steps / Animal.MAX_AGE)  # Penalize for dying
        
        return Simulation.BASE_SCORE + self.current_generation.size + (
            self.current_generation.can_reproduce() * Simulation.SCORE_MULTIPLIER / (num_steps + Animal.MAX_AGE))

    def run_generation(self):
        """Run the simulation for a given number of steps."""
        step = 0
        while step < Animal.MAX_AGE:
            self.step()
            # Optionally, you can print or log the state of the simulation here
            print(f"Animal position: {self.current_generation.position}, Energy: {self.current_generation.energy}, Size: {self.current_generation.size}")
            
            step += 1
            if self.current_generation.is_dead():
                print("Animal has been eaten or starved.")
                break

            if self.current_generation.can_reproduce():
                # Handle reproduction logic here if needed
                print("The animal could reproduce. New Generation Created. Learning from the current generation.")
                self.current_generation.brain.learn(self.current_generation.energy)
                self.generations.append(self.current_generation)
                break

        print(f"Simulation ended after {step} steps. Final score: {self.final_score(step)}")

def load_config(config_file: str = 'config.json') -> dict:
    """Load configuration from a JSON file."""
    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            return config
    except FileNotFoundError:
        print("Configuration file not found. Using default settings.")
        return {}
    

if __name__ == "__main__":

    # Load configuration
    if len(sys.argv) > 1:
        config_file = sys.argv[1]
    else:
        config_file = 'config.json'

    config = load_config(config_file)
    if config:
        print(f"Loaded configuration: {config}")

        # Apply configuration settings if available
       
        Simulation.BASE_SCORE = config.get('base_score', Simulation.BASE_SCORE)
        Simulation.SCORE_MULTIPLIER = config.get('score_multiplier', Simulation.SCORE_MULTIPLIER)
        Simulation.FOOD_RANDOM_WALK_RATE = config.get('food_random_walk_rate', Simulation.FOOD_RANDOM_WALK_RATE)
        Simulation.FOOD_RATE = config.get('food_rate', Simulation.FOOD_RATE)
        Simulation.FOOD_MAX_ENERGY = config.get('food_max_energy', Simulation.FOOD_MAX_ENERGY)
        Simulation.FOOD_MIN_ENERGY = config.get('food_min_energy', Simulation.FOOD_MIN_ENERGY)
        Simulation.SIZE = config.get('size', Simulation.SIZE)

    else:
        print("Using default configuration.")

    num_generations = 1
    if len(sys.argv) > 2:
        num_generations = int(sys.argv[2])

    sim = Simulation()

    for _ in range(num_generations):
        sim.reset_with_new_generation()
        sim.run_generation()

    print("Simulation complete.")
    # You can add more logic here to handle multiple generations or further analysis