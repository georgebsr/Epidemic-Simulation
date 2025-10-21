#Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import seaborn as sns
from dataclasses import dataclass
from typing import List, Tuple, Dict
from enum import Enum
import copy

class AgentState(Enum):
    """Πιθανες καταστασεις ενος agent"""
    SUSCEPTIBLE = "susceptible"
    INFECTED = "infected"
    INFECTIOUS = "infectious"
    RECOVERED = "recovered"
    QUARANTINED = "quarantined"

@dataclass
class SimulationParameters:
    """Παραμετροι προσομοίωσης"""
    grid_size: int = 50                      #size of simulation space
    population_size: int = 1000              #total population
    initial_infected: int = 10               #initially infected agents
    transmission_radius: float = 2.0         #radius of transmission
    infection_to_infectious_steps: int = 5   #steps from infected to being able to infect others
    transmission_probability: float = 0.3    #probability of transmission event
    movement_speed: float = 1.0              #movement speed
    quarantine_probability: float = 0.1      #probability for quarantine
    simulation_steps: int = 100              #total simulation steps

class Agent:
    """Class representing individuals in the population"""

    def __init__(self, agent_id: int, x: float, y: float, state: AgentState = AgentState.SUSCEPTIBLE):
        self.agent_id = agent_id
        self.x = x
        self.y = y
        self.state = state
        self.infection_time = 0
        self.is_quarantined = False

    def move(self, grid_size: int, speed: float = 1.0):
        """Move agent randomly using 9-neighborhood pattern"""
        if self.is_quarantined:
            return

        #9 possible movements (including staying still)
        movements = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, 1), (1, -1), (-1, 1), (-1, -1)
        ]

        dx, dy = random.choice(movements)
        self.x = max(0, min(grid_size - 1, self.x + dx * speed))
        self.y = max(0, min(grid_size - 1, self.y + dy * speed))

    def update_infection(self, params: SimulationParameters):
        """Update infection state based on disease progression"""
        if self.state == AgentState.INFECTED:
            self.infection_time += 1
            if self.infection_time >= params.infection_to_infectious_steps:
                self.state = AgentState.INFECTIOUS
                #check fr quarantine placement
                if random.random() < params.quarantine_probability:
                    self.is_quarantined = True
                    self.state = AgentState.QUARANTINED

class EpidemicSimulation:
    """Main simulation engine for epidemic modeling"""

    def __init__(self, params: SimulationParameters):
        self.params = params
        self.agents = []
        self.daily_stats = []
        self.initialize_population()

    def initialize_population(self):
        """Initialize population with susceptible and infected agents"""
        self.agents = []

        #create susceptible agents
        for i in range(self.params.population_size - self.params.initial_infected):
            x = random.uniform(0, self.params.grid_size - 1)
            y = random.uniform(0, self.params.grid_size - 1)
            agent = Agent(i, x, y, AgentState.SUSCEPTIBLE)
            self.agents.append(agent)

        #create initially infected agents
        for i in range(self.params.initial_infected):
            x = random.uniform(0, self.params.grid_size - 1)
            y = random.uniform(0, self.params.grid_size - 1)
            agent = Agent(self.params.population_size - self.params.initial_infected + i, 
                         x, y, AgentState.INFECTED)
            self.agents.append(agent)

    def calculate_distance(self, agent1: Agent, agent2: Agent) -> float:
        """Calculate Euclidean distance between two agents"""
        return np.sqrt((agent1.x - agent2.x)**2 + (agent1.y - agent2.y)**2)

    def transmission_step(self):
        """Handle disease transmission between agents"""
        new_infections = []

        infectious_agents = [a for a in self.agents 
                           if a.state == AgentState.INFECTIOUS and not a.is_quarantined]
        susceptible_agents = [a for a in self.agents if a.state == AgentState.SUSCEPTIBLE]

        for infectious_agent in infectious_agents:
            for susceptible_agent in susceptible_agents:
                distance = self.calculate_distance(infectious_agent, susceptible_agent)

                if distance <= self.params.transmission_radius:
                    if random.random() < self.params.transmission_probability:
                        new_infections.append(susceptible_agent)

        #apply neww infections
        for agent in new_infections:
            agent.state = AgentState.INFECTED
            agent.infection_time = 0

    def step(self):
        """Perform one simulation time step"""
        #agent movement with infection based speed mod
        for agent in self.agents:
            speed = self.params.movement_speed
            if agent.state == AgentState.INFECTIOUS:
                speed *= 0.7  #reduced mobility when infectious

            agent.move(self.params.grid_size, speed)

        #upd infection states
        for agent in self.agents:
            agent.update_infection(self.params)

        #handle disease transmission
        self.transmission_step()

        #collect daily statistics
        stats = self.collect_statistics()
        self.daily_stats.append(stats)

    def collect_statistics(self) -> Dict:
        """Collect daily statistics"""
        state_counts = {state: 0 for state in AgentState}

        for agent in self.agents:
            state_counts[agent.state] += 1

        return {
            'susceptible': state_counts[AgentState.SUSCEPTIBLE],
            'infected': state_counts[AgentState.INFECTED],
            'infectious': state_counts[AgentState.INFECTIOUS],
            'recovered': state_counts[AgentState.RECOVERED],
            'quarantined': state_counts[AgentState.QUARANTINED],
            'new_cases': state_counts[AgentState.INFECTED] + state_counts[AgentState.INFECTIOUS],
            'total_infected': (state_counts[AgentState.INFECTED] + 
                             state_counts[AgentState.INFECTIOUS] + 
                             state_counts[AgentState.RECOVERED] + 
                             state_counts[AgentState.QUARANTINED])
        }

    def run_simulation(self) -> pd.DataFrame:
        """Execute complete simulation and return result"""
        self.daily_stats = []

        for step in range(self.params.simulation_steps):
            self.step()

        df = pd.DataFrame(self.daily_stats)
        df['day'] = range(len(df))
        return df

def run_monte_carlo_experiment(params: SimulationParameters, num_runs: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run Monte Carlo experiment with multple simulation runs"""
    all_results = []

    for run in range(num_runs):
        sim = EpidemicSimulation(params)
        results = sim.run_simulation()
        results['run'] = run
        all_results.append(results)

    #combine the results and calculate avg
    combined_df = pd.concat(all_results, ignore_index=True)
    avg_results = combined_df.groupby('day').agg({
        'susceptible': 'mean',
        'infected': 'mean',
        'infectious': 'mean',
        'recovered': 'mean',
        'quarantined': 'mean',
        'new_cases': 'mean',
        'total_infected': 'mean'
    }).reset_index()

    return avg_results, combined_df

def parameter_sensitivity_analysis(baseline_params: SimulationParameters) -> Dict:
    """Comprehensive parameter sensitivity analysis"""
    results = {}

    #test transmission radius effects
    for radius in [1.0, 2.0, 3.0, 4.0]:
        params = copy.deepcopy(baseline_params)
        params.transmission_radius = radius
        avg_results, _ = run_monte_carlo_experiment(params, num_runs=10)
        results[f'radius_{radius}'] = avg_results

    #test population size effects
    for pop_size in [500, 1000, 2000]:
        params = copy.deepcopy(baseline_params)
        params.population_size = pop_size
        params.initial_infected = max(1, pop_size // 100)
        avg_results, _ = run_monte_carlo_experiment(params, num_runs=10)
        results[f'population_{pop_size}'] = avg_results

    #test transmission probability effects
    for prob in [0.1, 0.3, 0.5, 0.7]:
        params = copy.deepcopy(baseline_params)
        params.transmission_probability = prob
        avg_results, _ = run_monte_carlo_experiment(params, num_runs=10)
        results[f'transmission_prob_{prob}'] = avg_results

    #test quarantine probability effects
    for prob in [0.0, 0.1, 0.3, 0.5]:
        params = copy.deepcopy(baseline_params)
        params.quarantine_probability = prob
        avg_results, _ = run_monte_carlo_experiment(params, num_runs=10)
        results[f'quarantine_prob_{prob}'] = avg_results

    return results

#main execution example
if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    random.seed(42)

    #define baseline parameters
    baseline_params = SimulationParameters(
        grid_size=50,
        population_size=1000,
        initial_infected=10,
        transmission_radius=2.0,
        infection_to_infectious_steps=5,
        transmission_probability=0.3,
        movement_speed=1.0,
        quarantine_probability=0.1,
        simulation_steps=100
    )

    #run baseline experiment
    print("Running baseline Monte Carlo experiment")
    baseline_avg, baseline_all = run_monte_carlo_experiment(baseline_params)

    #run parameter sensitivity analysis
    print("Running parameter sensitivity analysis")
    sensitivity_results = parameter_sensitivity_analysis(baseline_params)

    #save results
    baseline_avg.to_csv('simulation_results_baseline.csv', index=False)
    print("Simulation completed successfully!")
