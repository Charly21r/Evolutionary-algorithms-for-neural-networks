import random
import math
from collections import defaultdict, deque

def sigmoid(x):
    return 1 / (1 + math.exp(-max(-500, min(500, x))))

def ReLU(x):
    return max(0, x)

def topological_sort(edges):
    """Topological sort for directed acyclic graph"""
    in_degree = defaultdict(int)
    nodes = set()
    
    for node in edges:
        nodes.add(node)
        for neighbor in edges[node]:
            nodes.add(neighbor)
            in_degree[neighbor] += 1
    
    queue = deque([node for node in nodes if in_degree[node] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in edges[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result

class InnovationTracker:
    """Enhanced innovation tracking with historical record"""
    def __init__(self):
        self.current_innovation = 0
        self.connection_innovations = {}  # (in_node_id, out_node_id) -> innovation_number
        self.node_innovations = {}        # connection_innovation -> (node_innovation, conn1_innovation, conn2_innovation)
        self.node_id_counter = 0
        
    def get_connection_innovation(self, in_node_id, out_node_id):
        """Get innovation number for a connection, creating new if needed"""
        key = (in_node_id, out_node_id)
        if key not in self.connection_innovations:
            self.connection_innovations[key] = self.current_innovation
            self.current_innovation += 1
        return self.connection_innovations[key]
    
    def get_node_innovation(self, connection_innovation):
        """Get innovation numbers for node insertion, creating new if needed"""
        if connection_innovation not in self.node_innovations:
            # Create new node and two connections
            node_id = self.node_id_counter
            self.node_id_counter += 1
            
            conn1_innovation = self.current_innovation
            self.current_innovation += 1
            
            conn2_innovation = self.current_innovation
            self.current_innovation += 1
            
            self.node_innovations[connection_innovation] = (node_id, conn1_innovation, conn2_innovation)
        
        return self.node_innovations[connection_innovation]

class NodeGene:
    def __init__(self, node_id, layer, activation, bias):
        self.node_id = node_id  # Unique identifier for the node
        self.layer = layer
        self.activation = activation
        self.bias = bias
    
    def __eq__(self, other):
        return isinstance(other, NodeGene) and self.node_id == other.node_id
    
    def __hash__(self):
        return hash(self.node_id)
    
    def copy(self):
        return NodeGene(self.node_id, self.layer, self.activation, self.bias)

class ConnectionGene:
    def __init__(self, in_node_id, out_node_id, weight, innovation, enabled=True):
        self.in_node_id = in_node_id
        self.out_node_id = out_node_id
        self.weight = weight
        self.enabled = enabled
        self.innovation = innovation
    
    def copy(self):
        return ConnectionGene(self.in_node_id, self.out_node_id, self.weight, self.innovation, self.enabled)

class Genome:
    def __init__(self, nodes, connections):
        self.nodes = {node.node_id: node for node in nodes}  # Dict for faster lookup
        self.connections = connections
        self.fitness = 0
    
    def get_node_by_id(self, node_id):
        return self.nodes.get(node_id)
    
    def mutate_add_connection(self, innovation_tracker):
        """Add a new connection between two nodes"""
        node_list = list(self.nodes.values())
        
        # Try multiple times to find valid connection
        for _ in range(10):
            node1, node2 = random.sample(node_list, 2)
            
            # Prevent self-connections and connections to input nodes
            if node1.node_id == node2.node_id or node2.layer == "input":
                continue
            
            # Prevent recurrent connections (simple check)
            if node1.layer == "output" and node2.layer != "output":
                continue
                
            # Check if connection already exists
            exists = any(c.in_node_id == node1.node_id and c.out_node_id == node2.node_id 
                        for c in self.connections)
            if exists:
                continue
            
            # Create new connection
            innovation = innovation_tracker.get_connection_innovation(node1.node_id, node2.node_id)
            new_conn = ConnectionGene(node1.node_id, node2.node_id, 
                                    random.uniform(-1, 1), innovation, True)
            self.connections.append(new_conn)
            break
    
    def mutate_add_node(self, innovation_tracker):
        """Add a new node by splitting an existing connection"""
        enabled_connections = [c for c in self.connections if c.enabled]
        if not enabled_connections:
            return
        
        connection = random.choice(enabled_connections)
        connection.enabled = False
        
        # Get innovation numbers for this node split
        node_id, conn1_innovation, conn2_innovation = innovation_tracker.get_node_innovation(connection.innovation)
        
        # Create new node
        new_node = NodeGene(node_id, "hidden", ReLU, random.uniform(-1, 1))
        self.nodes[node_id] = new_node
        
        # Create two new connections
        conn1 = ConnectionGene(connection.in_node_id, node_id, 1.0, conn1_innovation, True)
        conn2 = ConnectionGene(node_id, connection.out_node_id, connection.weight, conn2_innovation, True)
        
        self.connections.extend([conn1, conn2])
    
    def mutate_weights(self, rate=0.8, power=0.5):
        for conn in self.connections:
            if random.random() < rate:
                if random.random() < 0.1:  # Complete replacement
                    conn.weight = random.uniform(-1, 1)
                else:  # Perturbation
                    conn.weight += random.gauss(0, power)
                    conn.weight = max(-5, min(5, conn.weight))
    
    def mutate_bias(self, rate=0.7, power=0.5):
        for node in self.nodes.values():
            if node.layer != "input" and random.random() < rate:
                if random.random() < 0.1:
                    node.bias = random.uniform(-1, 1)
                else:
                    node.bias += random.gauss(0, power)
                    node.bias = max(-5, min(5, node.bias))
    
    def mutate_toggle_connection(self, rate=0.01):
        """Toggle enabled/disabled state of connections"""
        for conn in self.connections:
            if random.random() < rate:
                conn.enabled = not conn.enabled
    
    def mutate(self, innovation_tracker, conn_mutation_rate=0.05, node_mutation_rate=0.03, 
             weight_mutation_rate=0.8, bias_mutation_rate=0.7):
        self.mutate_weights(weight_mutation_rate)
        self.mutate_bias(bias_mutation_rate)
        self.mutate_toggle_connection()
        
        if random.random() < conn_mutation_rate:
            self.mutate_add_connection(innovation_tracker)
        
        if random.random() < node_mutation_rate:
            self.mutate_add_node(innovation_tracker)
    
    def evaluate(self, input_values):
        """Evaluate the network with given inputs"""
        node_values = {}
        node_inputs = defaultdict(list)
        
        input_nodes = [n for n in self.nodes.values() if n.layer == "input"]
        output_nodes = [n for n in self.nodes.values() if n.layer == "output"]
        
        if len(input_nodes) != len(input_values):
            raise ValueError("Number of inputs doesn't match number of input nodes")
        
        # Set input values
        for node, val in zip(input_nodes, input_values):
            node_values[node.node_id] = val
        
        # Build adjacency list and collect inputs for each node
        edges = defaultdict(list)
        for conn in self.connections:
            if conn.enabled:
                out_node = self.get_node_by_id(conn.out_node_id)
                in_node = self.get_node_by_id(conn.in_node_id)
                if out_node and in_node:
                    edges[in_node].append(out_node)
                    node_inputs[conn.out_node_id].append(conn)
        
        # Topological sort
        sorted_nodes = topological_sort(edges)
        
        # Evaluate nodes in topological order
        for node in sorted_nodes:
            if node.node_id in node_values:
                continue
            
            incoming = node_inputs[node.node_id]
            total_input = sum(
                node_values[conn.in_node_id] * conn.weight for conn in incoming
                if conn.in_node_id in node_values
            ) + node.bias
            
            node_values[node.node_id] = node.activation(total_input)
        
        return [node_values.get(n.node_id, 0) for n in output_nodes]

def crossover(parent1, parent2):
    """Crossover two genomes, assuming parent1 is fitter"""
    # Collect all nodes from both parents
    all_nodes = {}
    for node in parent1.nodes.values():
        all_nodes[node.node_id] = node.copy()
    for node in parent2.nodes.values():
        if node.node_id not in all_nodes:
            all_nodes[node.node_id] = node.copy()
    
    # Build maps of connections by innovation number
    genes1 = {g.innovation: g for g in parent1.connections}
    genes2 = {g.innovation: g for g in parent2.connections}
    
    offspring_connections = []
    all_innovations = set(genes1.keys()) | set(genes2.keys())
    
    for innov in sorted(all_innovations):
        gene1 = genes1.get(innov)
        gene2 = genes2.get(innov)
        
        if gene1 and gene2:  # Matching genes
            selected = random.choice([gene1, gene2])
            # If one is disabled, 75% chance the offspring is disabled
            if not gene1.enabled or not gene2.enabled:
                if random.random() < 0.75:
                    selected.enabled = False
            offspring_connections.append(selected.copy())
        elif gene1:  # Disjoint/excess from fitter parent
            offspring_connections.append(gene1.copy())
        # Don't include genes only in less fit parent
    
    # Only include nodes that are actually used in connections
    used_nodes = set()
    for conn in offspring_connections:
        used_nodes.add(conn.in_node_id)
        used_nodes.add(conn.out_node_id)
    
    offspring_nodes = [all_nodes[node_id] for node_id in used_nodes if node_id in all_nodes]
    
    return Genome(offspring_nodes, offspring_connections)

def distance(genome1, genome2, c1=1.0, c2=1.0, c3=0.4):
    """Calculate genetic distance between two genomes"""
    genes1 = {g.innovation: g for g in genome1.connections}
    genes2 = {g.innovation: g for g in genome2.connections}
    
    innovations1 = set(genes1.keys())
    innovations2 = set(genes2.keys())
    
    if not innovations1 and not innovations2:
        return 0
    
    matching = innovations1 & innovations2
    all_innovations = innovations1 | innovations2
    disjoint_excess = all_innovations - matching
    
    # Calculate average weight difference for matching genes
    if matching:
        weight_diff = sum(abs(genes1[i].weight - genes2[i].weight) for i in matching)
        avg_weight_diff = weight_diff / len(matching)
    else:
        avg_weight_diff = 0
    
    # Normalize by genome size
    N = max(len(genome1.connections), len(genome2.connections), 1)
    if N < 20:
        N = 1
    
    delta = (c1 * len(disjoint_excess)) / N + c3 * avg_weight_diff
    return delta

class Species:
    def __init__(self, representative):
        self.representative = representative
        self.members = []
        self.adjusted_fitness = 0
        self.max_fitness = 0
        self.stagnant_generations = 0
        
    def add_member(self, genome):
        self.members.append(genome)
        
    def clear_members(self):
        self.members = []
        
    def update_fitness_stats(self):
        if self.members:
            self.max_fitness = max(member.fitness for member in self.members)
            self.adjusted_fitness = sum(member.fitness for member in self.members) / len(self.members)

class Speciator:
    def __init__(self, compatibility_threshold=3.0):
        self.species = []
        self.compatibility_threshold = compatibility_threshold
        
    def speciate(self, population):
        # Clear existing species
        for species in self.species:
            species.clear_members()
        
        # Assign genomes to species
        for genome in population:
            found_species = False
            for species in self.species:
                if distance(genome, species.representative) < self.compatibility_threshold:
                    species.add_member(genome)
                    found_species = True
                    break
            
            if not found_species:
                new_species = Species(genome)
                new_species.add_member(genome)
                self.species.append(new_species)
        
        # Remove empty species and update representatives
        self.species = [s for s in self.species if s.members]
        
        for species in self.species:
            species.update_fitness_stats()
            # Update representative to be the best member
            species.representative = max(species.members, key=lambda g: g.fitness)

def create_initial_genome(num_inputs, num_outputs, innovation_tracker):
    """Create a minimal initial genome"""
    nodes = []
    connections = []
    
    # Create input nodes
    for i in range(num_inputs):
        node = NodeGene(i, "input", lambda x: x, 0)
        nodes.append(node)
    
    # Create output nodes
    for i in range(num_outputs):
        node_id = num_inputs + i
        node = NodeGene(node_id, "output", sigmoid, random.uniform(-1, 1))
        nodes.append(node)
    
    # Create connections from all inputs to all outputs
    for i in range(num_inputs):
        for j in range(num_outputs):
            in_node_id = i
            out_node_id = num_inputs + j
            weight = random.uniform(-1, 1)
            innovation = innovation_tracker.get_connection_innovation(in_node_id, out_node_id)
            conn = ConnectionGene(in_node_id, out_node_id, weight, innovation, True)
            connections.append(conn)
    
    return Genome(nodes, connections)

def create_initial_population(size, num_inputs, num_outputs, innovation_tracker):
    """Create initial population with identical topology but different weights"""
    population = []
    
    # Create the first genome
    first_genome = create_initial_genome(num_inputs, num_outputs, innovation_tracker)
    population.append(first_genome)
    
    # Create remaining genomes with same topology but different weights
    for _ in range(size - 1):
        genome = create_initial_genome(num_inputs, num_outputs, innovation_tracker)
        # Randomize weights
        for conn in genome.connections:
            conn.weight = random.uniform(-1, 1)
        population.append(genome)
    
    return population

# XOR Problem
X = [[0, 0], [0, 1], [1, 0], [1, 1]]
y = [0, 1, 1, 0]

def fitness_xor(genome):
    """Calculate fitness for XOR problem"""
    total_error = 0
    for i in range(len(X)):
        try:
            output = genome.evaluate(X[i])
            if output:
                error = abs(output[0] - y[i])
                total_error += error
        except:
            return 0
    
    fitness = 4 - total_error
    return max(0, fitness)

def evolution(population, speciator, innovation_tracker, target_offspring=None):
    """Evolve population for one generation"""
    if target_offspring is None:
        target_offspring = len(population)
    
    # Evaluate fitness
    for genome in population:
        genome.fitness = fitness_xor(genome)
    
    # Speciate
    speciator.speciate(population)
    
    new_population = []
    
    # Calculate total adjusted fitness
    total_adjusted_fitness = sum(s.adjusted_fitness for s in speciator.species)
    
    # Reproduce each species
    for species in speciator.species:
        # Calculate offspring allocation
        if total_adjusted_fitness > 0:
            offspring_count = max(1, int((species.adjusted_fitness / total_adjusted_fitness) * target_offspring))
        else:
            offspring_count = target_offspring // len(speciator.species)
        
        # Elite selection (keep best member)
        if species.members:
            best_member = max(species.members, key=lambda g: g.fitness)
            new_population.append(best_member)
            offspring_count -= 1
        
        # Generate offspring
        for _ in range(offspring_count):
            if len(species.members) == 1:
                # Asexual reproduction
                parent = species.members[0]
                child = Genome(list(parent.nodes.values()), [c.copy() for c in parent.connections])
            else:
                # Sexual reproduction
                parent1, parent2 = random.sample(species.members, 2)
                if parent1.fitness < parent2.fitness:
                    parent1, parent2 = parent2, parent1
                child = crossover(parent1, parent2)
            
            child.mutate(innovation_tracker)
            new_population.append(child)
    
    # Ensure we have the right population size
    while len(new_population) < target_offspring:
        if speciator.species:
            best_species = max(speciator.species, key=lambda s: s.adjusted_fitness)
            if best_species.members:
                parent = random.choice(best_species.members)
                child = Genome(list(parent.nodes.values()), [c.copy() for c in parent.connections])
                child.mutate(innovation_tracker)
                new_population.append(child)
        else:
            break
    
    return new_population[:target_offspring]

def run_neat_xor(generations=500, pop_size=150, target_fitness=3.9):
    """Run NEAT on XOR problem"""
    NUM_INPUTS = 2
    NUM_OUTPUTS = 1
    
    # Initialize components
    innovation_tracker = InnovationTracker()
    speciator = Speciator(compatibility_threshold=3.0)
    
    # Create initial population
    population = create_initial_population(pop_size, NUM_INPUTS, NUM_OUTPUTS, innovation_tracker)
    
    # Evolution loop
    for generation in range(generations):
        # Evaluate fitness
        fitness_scores = [fitness_xor(genome) for genome in population]
        best_fitness = max(fitness_scores)
        avg_fitness = sum(fitness_scores) / len(fitness_scores)
        
        print(f"Generation {generation}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}, "
              f"Species={len(speciator.species)}")
        
        # Check for solution
        if best_fitness >= target_fitness:
            print(f"Solution found in generation {generation}!")
            best_genome = population[fitness_scores.index(best_fitness)]
            return best_genome, generation
        
        # Evolve
        population = evolution(population, speciator, innovation_tracker, pop_size)
    
    print("No solution found within generation limit")
    return None, generations

# Run the algorithm
if __name__ == "__main__":
    best_genome, generations = run_neat_xor()
    
    if best_genome:
        print(f"\nSolution found! Testing on XOR:")
        for i, (x_val, y_val) in enumerate(zip(X, y)):
            output = best_genome.evaluate(x_val)
            print(f"Input: {x_val}, Expected: {y_val}, Got: {output[0]:.4f}")
