"""
Sports Tournament Scheduler - Genetic Algorithm Module

This module contains all the core GA logic including:
- Match class definition
- Population initialization
- Fitness function
- Selection methods (Tournament, Roulette Wheel)
- Crossover methods (One-point, Two-point, Order)
- Mutation methods (Swap, Inversion, Scramble)
- Main genetic algorithm function
"""

import random
import copy
from datetime import datetime, timedelta
from collections import defaultdict


# ==================== DATA CONFIGURATION ====================

TEAMS = [
    "Al Ahly", "Zamalek", "Pyramids", "Masry", "Future", "Ismaily",
    "Smouha", "ENPPI", "Ceramica", "National Bank", "Talaea El Gaish",
    "Alexandria Union", "El Dakhleya", "El Gouna", "Zed",
    "Modern Sport", "Pharco", "Wadi Degla"
]

VENUES = [
    "Cairo Stadium", "Borg El Arab", "Air Defense Stadium",
    "Suez Stadium", "Alexandria Stadium", "Petro Sport Stadium",
    "Military Academy Stadium", "Al Salam Stadium",
    "El Sekka El Hadeed Stadium", "Zed Club Stadium"
]

MATCH_TIMES = ["17:00", "20:00"]


def generate_dates(start_date=None, end_date=None):
    """Generate list of available dates for the tournament"""
    if start_date is None:
        start_date = datetime(2025, 5, 1)
    if end_date is None:
        end_date = datetime(2026, 1, 31)
    
    dates = []
    current = start_date
    while current <= end_date:
        dates.append(current)
        current += timedelta(days=1)
    return dates


# ==================== MATCH CLASS ====================

class Match:
    """Represents a single match in the tournament schedule"""
    
    def __init__(self, team1, team2, date, time, venue, leg):
        self.team1 = team1
        self.team2 = team2
        self.date = date
        self.time = time
        self.venue = venue
        self.leg = leg

    def __repr__(self):
        return f"{self.team1} vs {self.team2} (leg {self.leg}) on {self.date.strftime('%Y-%m-%d')} {self.time} at {self.venue}"
    
    def __eq__(self, other):
        if not isinstance(other, Match):
            return False
        return (self.team1 == other.team1 and 
                self.team2 == other.team2 and 
                self.leg == other.leg)
    
    def __hash__(self):
        return hash((self.team1, self.team2, self.leg))


# ==================== POPULATION INITIALIZATION ====================

def generate_all_matches(teams):
    """Generate all possible matches (home & away) for the tournament"""
    matches = []
    for i in range(len(teams)):
        for j in range(i + 1, len(teams)):
            matches.append((teams[i], teams[j], 1))  # Leg 1
            matches.append((teams[j], teams[i], 2))  # Leg 2
    return matches


def create_random_individual(teams, venues, dates, match_times):
    """Create a random chromosome (individual) representing a full schedule"""
    chromosome = []
    all_matches = generate_all_matches(teams)
    random.shuffle(all_matches)

    for team1, team2, leg in all_matches:
        chromosome.append(
            Match(
                team1,
                team2,
                random.choice(dates),
                random.choice(match_times),
                random.choice(venues),
                leg
            )
        )
    return chromosome


def create_initial_population(pop_size, teams, venues, dates, match_times):
    """Create initial population of random individuals"""
    population = []
    for _ in range(pop_size):
        population.append(
            create_random_individual(teams, venues, dates, match_times)
        )
    return population


# ==================== FITNESS FUNCTION ====================

def fitness(individual):
    """
    Calculate fitness of an individual (schedule).
    
    Fitness = 1 / (1 + total_penalty)
    
    Penalties:
    - Venue Conflicts (weight 5): Same venue + same date + same time
    - Rest Period Violations (weight 3): Team playing with less than 1 day rest
    - Match Time Balance (weight 1): Uneven distribution between 17:00 and 20:00
    """
    penalty = 0

    # 1. Venue Conflicts (weight 5)
    venue_time_dict = {}
    for match in individual:
        key = (match.date, match.time, match.venue)
        if key in venue_time_dict:
            penalty += 5
        else:
            venue_time_dict[key] = match

    # 2. Rest periods (weight 3)
    team_dates = {}
    for match in individual:
        for team in [match.team1, match.team2]:
            if team not in team_dates:
                team_dates[team] = []
            team_dates[team].append(match.date)

    for dates in team_dates.values():
        dates.sort()
        for i in range(1, len(dates)):
            if (dates[i] - dates[i-1]).days < 1:
                penalty += 3

    # 3. Balance game times (weight 1)
    team_time_count = {team: {"17:00": 0, "20:00": 0} for team in team_dates}
    for match in individual:
        team_time_count[match.team1][match.time] += 1
        team_time_count[match.team2][match.time] += 1

    for counts in team_time_count.values():
        penalty += abs(counts["17:00"] - counts["20:00"]) * 1

    # Final fitness (higher is better)
    return 1 / (1 + penalty)


def calculate_penalty_breakdown(individual):
    """
    Calculate detailed penalty breakdown for analysis.
    Returns dict with individual penalty components.
    """
    venue_conflicts = 0
    rest_violations = 0
    time_imbalance = 0
    
    # 1. Venue Conflicts
    venue_time_dict = {}
    for match in individual:
        key = (match.date, match.time, match.venue)
        if key in venue_time_dict:
            venue_conflicts += 1
        else:
            venue_time_dict[key] = match

    # 2. Rest periods
    team_dates = defaultdict(list)
    for match in individual:
        team_dates[match.team1].append(match.date)
        team_dates[match.team2].append(match.date)

    for dates_list in team_dates.values():
        dates_list.sort()
        for i in range(1, len(dates_list)):
            if (dates_list[i] - dates_list[i-1]).days < 1:
                rest_violations += 1

    # 3. Time balance
    team_time_count = {team: {"17:00": 0, "20:00": 0} for team in team_dates}
    for match in individual:
        team_time_count[match.team1][match.time] += 1
        team_time_count[match.team2][match.time] += 1

    for counts in team_time_count.values():
        time_imbalance += abs(counts["17:00"] - counts["20:00"])
    
    total_penalty = venue_conflicts * 5 + rest_violations * 3 + time_imbalance
    
    return {
        'venue_conflicts': venue_conflicts,
        'rest_violations': rest_violations,
        'time_imbalance': time_imbalance,
        'total_penalty': total_penalty,
        'fitness': 1 / (1 + total_penalty)
    }


# ==================== SELECTION METHODS ====================

def tournament_selection(population, fitness_scores, k=3):
    """
    Tournament selection: Select k individuals randomly and return the best.
    Balances exploration and exploitation.
    """
    selected = random.sample(list(zip(population, fitness_scores)), k)
    selected.sort(key=lambda x: x[1], reverse=True)
    return selected[0][0]


def roulette_wheel_selection(population, fitness_scores):
    """
    Roulette wheel selection: Select individual proportionally to fitness.
    Higher fitness = higher probability of selection.
    """
    total_fitness = sum(fitness_scores)
    pick = random.uniform(0, total_fitness)
    current = 0
    for ind, fit in zip(population, fitness_scores):
        current += fit
        if current > pick:
            return ind
    return population[-1]  # Fallback


# ==================== CROSSOVER METHODS ====================

def one_point_crossover(parent1, parent2):
    """
    One-point crossover: Exchange segments at a random point.
    """
    point = random.randint(1, len(parent1) - 2)
    child1 = parent1[:point] + parent2[point:]
    child2 = parent2[:point] + parent1[point:]
    return child1, child2


def two_point_crossover(parent1, parent2):
    """
    Two-point crossover: Swap a continuous block between two points.
    Allows larger structural changes in schedules.
    """
    p1 = random.randint(1, len(parent1) - 3)
    p2 = random.randint(p1 + 1, len(parent1) - 2)

    child1 = parent1[:p1] + parent2[p1:p2] + parent1[p2:]
    child2 = parent2[:p1] + parent1[p1:p2] + parent2[p2:]

    return child1, child2


def order_crossover(parent1, parent2):
    """
    Order crossover (OX): Preserves relative order of genes.
    Useful for permutation-based representations.
    """
    size = len(parent1)
    p1 = random.randint(0, size - 2)
    p2 = random.randint(p1 + 1, size - 1)

    child = [None] * size
    child[p1:p2] = parent1[p1:p2]

    p2_genes = [gene for gene in parent2 if gene not in child]

    idx = 0
    for i in range(size):
        if child[i] is None:
            child[i] = p2_genes[idx]
            idx += 1

    return child


# ==================== MUTATION METHODS ====================

def swap_mutation(individual):
    """
    Swap mutation: Exchange two randomly selected genes.
    Introduces small local changes while preserving feasibility.
    """
    i, j = random.sample(range(len(individual)), 2)
    individual[i], individual[j] = individual[j], individual[i]


def inversion_mutation(individual):
    """
    Inversion mutation: Reverse the order of a segment.
    Helps escape local optima by restructuring match order.
    """
    i, j = sorted(random.sample(range(len(individual)), 2))
    individual[i:j] = reversed(individual[i:j])


def scramble_mutation(individual):
    """
    Scramble mutation: Randomly shuffle a segment.
    Introduces more randomness than swap or inversion.
    """
    i, j = sorted(random.sample(range(len(individual)), 2))
    subset = individual[i:j]
    random.shuffle(subset)
    individual[i:j] = subset


def apply_mutation(individual, mutation_rate=0.1, method="swap"):
    """
    Apply mutation to individual based on mutation rate and method.
    """
    if random.random() > mutation_rate:
        return

    if method == "swap":
        swap_mutation(individual)
    elif method == "inversion":
        inversion_mutation(individual)
    elif method == "scramble":
        scramble_mutation(individual)


# ==================== MAIN GENETIC ALGORITHM ====================

def genetic_algorithm(teams, venues, dates, match_times,
                      pop_size=50, generations=100,
                      crossover_rate=0.8, mutation_rate=0.1,
                      elitism_count=2, tournament_size=3,
                      crossover_method="two_point",
                      mutation_method="swap",
                      selection_method="tournament",
                      callback=None):
    """
    Main Genetic Algorithm function.
    
    Parameters:
    -----------
    teams : list - List of team names
    venues : list - List of venue names
    dates : list - List of available dates
    match_times : list - List of match times (e.g., ["17:00", "20:00"])
    pop_size : int - Population size
    generations : int - Number of generations
    crossover_rate : float - Probability of crossover (0.0-1.0)
    mutation_rate : float - Probability of mutation (0.0-1.0)
    elitism_count : int - Number of elite individuals to preserve
    tournament_size : int - Size of tournament for selection
    crossover_method : str - "one_point", "two_point", or "order"
    mutation_method : str - "swap", "inversion", or "scramble"
    selection_method : str - "tournament" or "roulette"
    callback : function - Optional callback for progress updates
    
    Returns:
    --------
    dict with keys:
        - best_individual: Best schedule found
        - best_fitness: Fitness of best schedule
        - best_fitness_history: List of best fitness per generation
        - avg_fitness_history: List of average fitness per generation
        - worst_fitness_history: List of worst fitness per generation
    """
    print(f"\n{'='*60}")
    print("🧬 GENETIC ALGORITHM STARTED")
    print(f"{'='*60}")

    # Create initial population
    population = create_initial_population(pop_size, teams, venues, dates, match_times)

    # History tracking
    best_fitness_history = []
    avg_fitness_history = []
    worst_fitness_history = []
    best_individual = None
    best_fitness = 0

    # Evolution loop
    for gen in range(generations):
        # Calculate fitness for all individuals
        fitness_scores = [fitness(ind) for ind in population]

        gen_best = max(fitness_scores)
        gen_avg = sum(fitness_scores) / len(fitness_scores)
        gen_worst = min(fitness_scores)

        best_fitness_history.append(gen_best)
        avg_fitness_history.append(gen_avg)
        worst_fitness_history.append(gen_worst)

        # Track best individual
        if gen_best > best_fitness:
            best_fitness = gen_best
            best_individual = copy.deepcopy(
                population[fitness_scores.index(gen_best)]
            )

        # Progress output
        if gen % 10 == 0 or gen == generations - 1:
            print(f"Gen {gen:3d} | Best: {gen_best:.6f} | Avg: {gen_avg:.6f}")
            
        # Callback for GUI updates
        if callback:
            callback(gen, generations, gen_best, gen_avg)

        # ================= ELITISM =================
        elite_idx = sorted(
            range(len(fitness_scores)),
            key=lambda i: fitness_scores[i],
            reverse=True
        )[:elitism_count]

        new_population = [copy.deepcopy(population[i]) for i in elite_idx]

        # ================= CREATE OFFSPRING =================
        while len(new_population) < pop_size:

            # -------- Parent Selection --------
            if selection_method == "tournament":
                p1 = tournament_selection(population, fitness_scores, tournament_size)
                p2 = tournament_selection(population, fitness_scores, tournament_size)
            else:  # roulette
                p1 = roulette_wheel_selection(population, fitness_scores)
                p2 = roulette_wheel_selection(population, fitness_scores)

            # -------- Crossover --------
            if random.random() < crossover_rate:
                if crossover_method == "one_point":
                    c1, c2 = one_point_crossover(copy.deepcopy(p1), copy.deepcopy(p2))
                elif crossover_method == "two_point":
                    c1, c2 = two_point_crossover(copy.deepcopy(p1), copy.deepcopy(p2))
                else:  # order crossover
                    c1 = order_crossover(copy.deepcopy(p1), copy.deepcopy(p2))
                    c2 = order_crossover(copy.deepcopy(p2), copy.deepcopy(p1))
            else:
                c1, c2 = copy.deepcopy(p1), copy.deepcopy(p2)

            # -------- Mutation --------
            apply_mutation(c1, mutation_rate, mutation_method)
            apply_mutation(c2, mutation_rate, mutation_method)

            new_population.append(c1)
            if len(new_population) < pop_size:
                new_population.append(c2)

        population = new_population

    print(f"\n{'='*60}")
    print("✅ EVOLUTION COMPLETED")
    print(f"🏆 Best Fitness: {best_fitness:.6f}")
    print(f"{'='*60}\n")

    return {
        'best_individual': best_individual,
        'best_fitness': best_fitness,
        'best_fitness_history': best_fitness_history,
        'avg_fitness_history': avg_fitness_history,
        'worst_fitness_history': worst_fitness_history
    }


# Module test
if __name__ == "__main__":
    print("Testing Genetic Algorithm Module...")
    
    dates = generate_dates()
    
    # Quick test run
    results = genetic_algorithm(
        TEAMS, VENUES, dates, MATCH_TIMES,
        pop_size=10,
        generations=5
    )
    
    print(f"\nTest completed! Best fitness: {results['best_fitness']:.6f}")

