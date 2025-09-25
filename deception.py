import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from math import sqrt
from statsmodels.stats.proportion import proportion_confint  # Wilson CI for victory rate


class DefenseSimulation:
    """Core simulation class for infrastructure defense scenarios"""
    
    def __init__(self, n_defenders=10, n_decoys=3, infra_radius=1.0,
                 decoy_cost=0.01, decoy_efficacy_range=(0.2, 0.6), decoy_visibility=1.8,
                 random_state=None): 
        self.rng = np.random.default_rng(random_state)
        self.n_defenders = n_defenders
        self.n_decoys = n_decoys
        self.infra_radius = infra_radius
        self.center = np.array([0.0, 0.0])
        self.attacker_path = []

        # Create real defenders
        self.defender_positions = self._create_defense_ring()
        self.defender_strengths = np.full(n_defenders, 0.8)
        self.defender_status = np.ones(n_defenders, dtype=bool)
        self.defender_alert = np.zeros(n_defenders, dtype=bool)
        self.defender_costs = np.zeros(n_defenders)
        self.is_decoy = np.zeros(n_defenders, dtype=bool) # All real initially

        # Create decoys and add them to the arrays
        decoy_positions = self._create_decoy_ring(n_decoys)
        self.defender_positions = np.vstack([self.defender_positions, decoy_positions])
        self.defender_strengths = np.concatenate([self.defender_strengths, np.full(n_decoys, 0.1)]) # Weak
        self.defender_status = np.concatenate([self.defender_status, np.ones(n_decoys, dtype=bool)])
        self.defender_alert = np.concatenate([self.defender_alert, np.zeros(n_decoys, dtype=bool)])
        self.is_decoy = np.concatenate([self.is_decoy, np.ones(n_decoys, dtype=bool)]) # Mark them as decoys
        self.decoy_efficacy = np.zeros(n_defenders + n_decoys)
        if n_decoys > 0:
            decoy_indices = np.where(self.is_decoy)[0]
            self.decoy_efficacy[decoy_indices] = self.rng.uniform(
                decoy_efficacy_range[0], decoy_efficacy_range[1], size=n_decoys
            )
        
        self.attacker_position = self._place_attacker()
        self.attacker_strength = 0.3
        self.attack_power = 0.04
        self.attack_cost = 0.0
        self.attacker_detected = False
        self.attacker_steps_remaining = 0
        self.defender_costs = np.concatenate([self.defender_costs, np.full(n_decoys, decoy_cost)])
        self.decoy_visibility = decoy_visibility

        self.detection_radius = 0.35 * infra_radius
        self.engagement_radius = 0.12 * infra_radius
        self.defense_multiplier = 3.0
        self.attacker_penalty = 0.6
        self.time_step = 0


    def _create_decoy_ring(self, n_decoys):
        """Place decoys in a ring outside the main defense"""
        angles = np.linspace(0, 2*np.pi, n_decoys, endpoint=False)
        radius = 0.7 * self.infra_radius # Place decoys further out
        return np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])



    def execute_defensive_engagement(self):
        """Execute defensive engagement with decoy and real defender logic."""
        for i in range(len(self.defender_status)):  # Loop through all nodes
            if not self.defender_status[i]:
                continue  # Skip defeated nodes

            dist = np.linalg.norm(self.defender_positions[i] - self.attacker_position)
            if dist < self.engagement_radius:

                if self.is_decoy[i]:
                    # --- Decoy logic ---
                    reduction = self.decoy_efficacy[i]
                    self.attacker_strength = max(0, self.attacker_strength - reduction)

                    # 30% chance attacker is fully misdirected (chases ghost target for extra delay)
                    if self.rng.random() < 0.3:
                        self.attacker_steps_remaining += 3
                        misdirected = True
                    else:
                        self.attacker_steps_remaining += 1
                        misdirected = False

                    # Decoy is consumed/disabled
                    self.defender_status[i] = False

                    # Very cheap cost compared to real defenders
                    self.defender_costs[i] += 0.01

                    if misdirected:
                        print(f"Decoy {i} compromised! Attacker MISDIRECTED and strength reduced by {reduction:.2f}.")
                    else:
                        print(f"Decoy {i} compromised! Attacker delayed and strength reduced by {reduction:.2f}.")

                else:
                    # --- Real defender logic ---
                    defense_power = self._calculate_defense_power(i)
                    defense_probability = defense_power / (defense_power + self.attacker_strength + 1e-8)

                    if self.rng.random() < defense_probability:
                        # Defender wins engagement
                        self.attacker_strength = max(0.0, self.attacker_strength - defense_power * 0.25)
                        self.defender_costs[i] += 0.08
                    else:
                        # Defender defeated
                        self.defender_status[i] = False
                        self.defender_costs[i] += 0.35


    def _create_defense_ring(self):
        """Create defensive ring configuration"""
        angles = np.linspace(0, 2*np.pi, self.n_defenders, endpoint=False)
        radius = 0.4 * self.infra_radius
        return np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])

    def _place_attacker(self):
        """Place attacker at random position"""
        angle = self.rng.uniform(0, 2*np.pi)
        return np.array([0.85*self.infra_radius*np.cos(angle), 
                        0.85*self.infra_radius*np.sin(angle)])

    def _calculate_defense_power(self, defender_idx):
        """Calculate defensive power with coordination effects"""
        if not self.defender_status[defender_idx] or self.is_decoy[defender_idx]:
            return 0.0
        
        base_power = self.defender_strengths[defender_idx]
        
        # Coordination bonus - only count real defenders (non-decoys)
        real_defenders_mask = ~self.is_decoy & self.defender_status
        active_real_defenders = real_defenders_mask & (np.arange(len(self.defender_status)) != defender_idx)
        
        if np.any(active_real_defenders):
            active_positions = self.defender_positions[active_real_defenders]
            distances = cdist([self.defender_positions[defender_idx]], active_positions)
            nearby_allies = np.sum(distances < 0.4 * self.infra_radius)
            coordination_bonus = 1.0 + 0.15 * nearby_allies
        else:
            coordination_bonus = 1.0
        
        # Alert state advantage
        alert_multiplier = 2.0 if self.defender_alert[defender_idx] else 1.0
        
        return base_power * coordination_bonus * alert_multiplier * self.defense_multiplier

    def execute_detection(self):
        """Execute detection phase"""
        for i in range(self.n_defenders):
            if not self.defender_status[i]:
                continue
            # Decoys have larger detection radius to attract attackers
            current_detection_radius = self.detection_radius
            if self.is_decoy[i]:
                current_detection_radius *= self.decoy_visibility
               
            dist = np.linalg.norm(self.defender_positions[i] - self.attacker_position)
            if dist < current_detection_radius and not self.attacker_detected:
                self.attacker_detected = True
                self.defender_alert[i] = True
                self.defender_costs[i] += 0.03
                
                # If a decoy detects, give intelligence bonus
                if self.is_decoy[i]:
                    self._apply_deception_bonus(i)

    def _apply_deception_bonus(self, decoy_idx):
        """Apply strategic bonuses when attacker interacts with decoy"""
        # Reveal attacker strength to all defenders (intelligence gain)
        strength_reveal_bonus = 1.0 + (0.1 * self.attacker_strength)
        
        # Apply bonus to all real defenders
        for i in range(len(self.defender_status)):
            if not self.is_decoy[i] and self.defender_status[i]:
                self.defender_strengths[i] *= strength_reveal_bonus
                
        # Slow down attacker (they're busy investigating the decoy)
        self.attacker_strength *= 0.85  # Attacker wastes resources
        print(f"Decoy {decoy_idx} engaged! Attacker strength revealed and reduced.")
                

    def update_strategic_state(self):
        """Update game state - MODIFIED FOR DECEPTION AWARENESS"""
        # If attacker is currently delayed by a decoy, consume one delayed step and skip movement and growth
        if self.attacker_steps_remaining > 0:
            self.attacker_steps_remaining -= 1
            # attacker is distracted this timestep: no movement, no strength growth
            return

        movement_speed = 0.018 * self.infra_radius

        # Attacker prioritizes decoys if detected
        target_position = self.center  # Default target

        # Find nearest active decoy
        decoy_positions = self.defender_positions[self.is_decoy & self.defender_status]
        if len(decoy_positions) > 0:
            decoy_distances = cdist([self.attacker_position], decoy_positions)
            nearest_decoy_idx = np.argmin(decoy_distances)
            if decoy_distances[0, nearest_decoy_idx] < 0.6 * self.infra_radius:
                target_position = decoy_positions[nearest_decoy_idx]
                # Attacker moves faster toward decoys (curiosity/aggression)
                movement_speed *= 1.3

        direction = target_position - self.attacker_position
        norm = np.linalg.norm(direction)
        if norm > 0:
            self.attacker_position += movement_speed * direction / norm

        # Attacker regenerates/charges each timestep of action
        self.attacker_strength += self.attack_power
        self.attack_cost += self.attack_power * 0.06


    def check_termination(self):
        """Check termination conditions"""
        center_distance = np.linalg.norm(self.attacker_position)
        
        if center_distance < 0.15 * self.infra_radius:
            return 'infrastructure_breached'
        elif not np.any(self.defender_status):
            return 'defenses_destroyed'
        elif self.attacker_strength < 0.05:
            return 'attacker_defeated'
        elif self.time_step >= 180:
            return 'engagement_timeout'
        
        return 'continue'

    def run_simulation(self):
        """Run a single simulation instance"""
        results = {
            'status': 'continue',
            'time_steps': 0,
            'defense_cost': 0,
            'nodes_survived': self.n_defenders,
            'attacker_final_strength': self.attacker_strength
        }

        self.attacker_path = [self.attacker_position.copy()]

        while results['status'] == 'continue':
            self.time_step += 1

            self.attacker_path.append(self.attacker_position.copy())
            
            self.execute_detection()
            self.execute_defensive_engagement()
            self.update_strategic_state()
            
            results['status'] = self.check_termination()
            results['time_steps'] = self.time_step
            results['defense_cost'] = np.sum(self.defender_costs)
            results['nodes_survived'] = np.sum(self.defender_status)
            results['attacker_final_strength'] = self.attacker_strength
        
        return results

class DefenseAnalyzer:
    """Analysis class for defense configuration optimization"""

    def __init__(self, n_simulations=100, base_seed=1234):
        self.n_simulations = n_simulations
        self.base_seed = base_seed
        self.results = []
    


    def analyze_configurations(self, node_configs=[4, 6, 8, 10, 12, 16], decoy_configs=[0, 2, 4, 6],
                               decoy_costs=[0.01], decoy_efficacy_ranges=[(0.2, 0.6)],
                               decoy_visibilities=[1.8]
    ):

        self.results = []
        for n_nodes in node_configs:
            for n_decoys in decoy_configs:
                for cost in decoy_costs:
                    for efficacy_range in decoy_efficacy_ranges:
                        for visibility in decoy_visibilities:
                            victories = 0
                            defense_costs = []
                            for run_idx in range(self.n_simulations):
                                seed = self.base_seed + hash((n_nodes, n_decoys, cost, efficacy_range, visibility, run_idx)) % (2**32)
                                sim = DefenseSimulation(
                                    n_defenders=n_nodes,
                                    n_decoys=n_decoys,
                                    decoy_cost=cost,
                                    decoy_efficacy_range=efficacy_range,
                                    decoy_visibility=visibility,
                                    random_state=seed
                                )
                                result = sim.run_simulation()

                                if result['status'] in ['attacker_defeated', 'engagement_timeout']:
                                    victories += 1

                                defense_costs.append(result['defense_cost'])

                            # --- summary statistics ---
                            victory_rate = victories / self.n_simulations
                            mean_cost = np.mean(defense_costs)

                            # Efficiency (scale-invariant)
                            efficiency = victory_rate / (mean_cost + 1e-12)

                            # Confidence intervals
                            # Victory rate CI (Wilson interval)
                            ci_low, ci_high = proportion_confint(
                                victories, self.n_simulations, alpha=0.05, method='wilson'
                            )

                            # Mean cost CI (normal approx)
                            cost_std = np.std(defense_costs, ddof=1)
                            cost_se = cost_std / sqrt(self.n_simulations)
                            cost_ci_low = mean_cost - 1.96 * cost_se
                            cost_ci_high = mean_cost + 1.96 * cost_se

                            # Optional: marginal efficiency relative to baseline (0 decoys)
                            if n_decoys > 0:
                                baseline = next(
                                    (r for r in self.results if r['defenders'] == n_nodes and r['decoys'] == 0),
                                    None
                                )
                                if baseline:
                                    marginal_eff = (victory_rate - baseline['victory_rate']) / max(
                                        1e-12, (mean_cost - baseline['mean_cost'])
                                    )
                                else:
                                    marginal_eff = None
                            else:
                                marginal_eff = None

                            # --- store results ---
                            self.results.append({
                                'defenders': n_nodes,
                                'decoys': n_decoys,
                                'decoy_cost': cost,
                                'decoy_efficacy_range': efficacy_range,
                                'decoy_visibility': visibility,
                                'total_nodes': n_nodes + n_decoys,
                                'victory_rate': victory_rate,
                                'victory_ci': (ci_low, ci_high),
                                'mean_cost': mean_cost,
                                'cost_ci': (cost_ci_low, cost_ci_high),
                                'efficiency': efficiency,
                                'marginal_efficiency': marginal_eff
                            })

        return self.results





class DefenseVisualizer:
    """Visualization class for defense analysis results"""
    

    @staticmethod
    def plot_attack_trajectory(simulation, result, save_path='attack-trajectory.png'):
        """Plot the trajectory of an attacker during a simulation run and annotate the outcome."""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot infrastructure
        infra_circle = plt.Circle((0, 0), simulation.infra_radius, color='#F8F9FA', 
                                 ec='#6C757D', linewidth=3, alpha=0.7, zorder=1)
        ax.add_patch(infra_circle)
        ax.plot(0, 0, 'ko', markersize=12, zorder=2, label='Critical Infrastructure')
        
        # Plot initial positions of defenders (including lost ones)
        defender_positions = simulation.defender_positions[~simulation.is_decoy]
        defender_status = simulation.defender_status[~simulation.is_decoy]
        # Plot active defenders
        ax.plot(defender_positions[defender_status, 0], defender_positions[defender_status, 1], 'o', 
                color='#2E86AB', markersize=10, label='Real Defenders (Active)', zorder=3)
        # Plot disabled defenders
        ax.plot(defender_positions[~defender_status, 0], defender_positions[~defender_status, 1], 'o', 
                color='#6C757D', markersize=10, alpha=0.5, markeredgewidth=0, label='Real Defenders (Lost)', zorder=3)
        
        # Plot decoys
        decoy_positions = simulation.defender_positions[simulation.is_decoy]
        if len(decoy_positions) > 0:
            ax.plot(decoy_positions[:, 0], decoy_positions[:, 1], 's', 
                    color='#F18F01', markersize=10, label='Decoys', zorder=3)
        
        # Plot attacker path
        path = np.array(simulation.attacker_path)
        ax.plot(path[:, 0], path[:, 1], 'r-', linewidth=1.5, label='Attacker Path', alpha=0.9, zorder=4)
        ax.plot(path[:, 0], path[:, 1], 'r.', markersize=2, alpha=0.9, zorder=4)
        
        # Mark start and end points
        ax.plot(path[0, 0], path[0, 1], 'g>', markersize=12, label='Start', markeredgewidth=1, markeredgecolor='black', zorder=5)
        ax.plot(path[-1, 0], path[-1, 1], 'rx', markersize=14, label='End', linewidth=3, zorder=5)
        
        # --- NEW: ANNOTATE THE OUTCOME ON THE PLOT ITSELF ---
        outcome_text = f"Result: {result['status'].replace('_', ' ').title()}\n"
        outcome_text += f"Time Steps: {result['time_steps']}\n"
        outcome_text += f"Defenders Left: {result['nodes_survived']}/{simulation.n_defenders}\n"
        outcome_text += f"Total Cost: {result['defense_cost']:.2f}"
        
        # Place the text box in the plot
        ax.text(0.02, 0.98, outcome_text, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                zorder=6)
        
        # --- Also add configuration to title ---
        title_str = f"Attack Trajectory: {simulation.n_defenders} Defenders + {simulation.n_decoys} Decoys"
        ax.set_title(title_str, fontsize=14, fontweight='bold')
        
        
        # Formatting
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position', fontweight='bold')
        ax.set_ylabel('Y Position', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig, ax


    @staticmethod
    def plot_cost_analysis(results, save_path='cost-analysis.png'):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax1 = plt.subplots(figsize=(12, 6))

        # Create labels for x-axis
        labels = [f"{r['defenders']}+{r['decoys']}" for r in results]
        x = np.arange(len(labels))
        width = 0.35

        # Plot Victory Rate
        victory_rates = [r['victory_rate'] for r in results]
        bars1 = ax1.bar(x - width/2, victory_rates, width, label='Survival Probability', color='#2E86AB')
        ax1.set_ylabel('Survival Probability', fontsize=12, fontweight='bold')
        ax1.set_ylim(0, 1.1)

        # Create a second y-axis for Cost
        ax2 = ax1.twinx()
        mean_costs = [r['mean_cost'] for r in results]
        bars2 = ax2.bar(x + width/2, mean_costs, width, label='Mean Defense Cost', color='#A23B72')
        ax2.set_ylabel('Mean Defense Cost', fontsize=12, fontweight='bold')

        # Formatting
        ax1.set_xlabel('Configuration (Real Defenders + Decoys)', fontsize=12, fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(labels, rotation=45)
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.set_title('Survival Probability and Defense Cost by Configuration', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def plot_optimization_results(results, save_path='optimal-defense-analysis.png'):
        """Create publication-quality optimization plots"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Victory rates vs defense density
        defenders = [r['defenders'] for r in results]
        victory_rates = [r['victory_rate'] for r in results]
        
        ax1.plot(defenders, victory_rates, 'o-', linewidth=3, markersize=8, 
                color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
        ax1.set_xlabel('Number of Defender Nodes', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Infrastructure Survival Probability', fontsize=12, fontweight='bold')
        ax1.set_title('Security Effectiveness vs Defense Density', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)
        
        # Cost-effectiveness analysis
        efficiency = [r['efficiency'] for r in results]
        
        ax2.plot(defenders, efficiency, 's-', linewidth=3, markersize=8,
                color='#A23B72', markerfacecolor='white', markeredgewidth=2)
        ax2.set_xlabel('Number of Defender Nodes', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Cost-Effectiveness Ratio', fontsize=12, fontweight='bold')
        ax2.set_title('Optimal Resource Allocation', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight optimal point
        optimal_idx = np.argmax(efficiency)
        ax2.plot(defenders[optimal_idx], efficiency[optimal_idx], 'o', 
                markersize=12, markeredgewidth=3, markerfacecolor='none', color='#F18F01')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def plot_defense_configuration(n_defenders=10, infra_radius=1.0):
        """Visualize the defense configuration"""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot infrastructure
        circle = plt.Circle((0, 0), infra_radius, color='#F8F9FA', ec='#6C757D', linewidth=2)
        ax.add_patch(circle)
        ax.plot(0, 0, 'ko', markersize=8, label='Critical Infrastructure')
        
        # Plot defense ring
        angles = np.linspace(0, 2*np.pi, n_defenders, endpoint=False)
        radius = 0.4 * infra_radius
        positions = np.column_stack([radius*np.cos(angles), radius*np.sin(angles)])
        
        ax.plot(positions[:, 0], positions[:, 1], 'o', color='#2E86AB', 
                markersize=10, label='Defender Nodes')
        
        # Formatting
        ax.set_xlim(-1.1, 1.1)
        ax.set_ylim(-1.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X Position', fontweight='bold')
        ax.set_ylabel('Y Position', fontweight='bold')
        ax.set_title(f'Optimal Defense Configuration ({n_defenders} Nodes)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'defense-configuration-{n_defenders}-nodes.png', dpi=300, bbox_inches='tight')
        plt.show()


    @staticmethod
    def plot_deception_strategy(results, save_path='deception-strategy-analysis.png'):
        """Plot the effectiveness of different deception strategies"""
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Group by number of decoys
        decoy_levels = sorted(set(r['decoys'] for r in results))
        colors = plt.cm.viridis(np.linspace(0, 1, len(decoy_levels)))
        
        for decoy_count, color in zip(decoy_levels, colors):
            decoy_data = [r for r in results if r['decoys'] == decoy_count]
            defenders = [r['defenders'] for r in decoy_data]
            victory_rates = [r['victory_rate'] for r in decoy_data]
            
            ax.plot(defenders, victory_rates, 'o-', linewidth=2, markersize=6,
                    color=color, label=f'{decoy_count} Decoys')
        
        ax.set_xlabel('Number of Real Defender Nodes', fontsize=12, fontweight='bold')
        ax.set_ylabel('Infrastructure Survival Probability', fontsize=12, fontweight='bold')
        ax.set_title('Deception Strategy Effectiveness', fontsize=14, fontweight='bold')
        ax.legend(title='Decoy Count')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.0)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()



# Implementation and analysis
if __name__ == "__main__":
    print("=== CRITICAL INFRASTRUCTURE DEFENSE OPTIMIZATION WITH DECEPTION ===")
    
    # Analyze defense configurations with deception
    analyzer = DefenseAnalyzer(n_simulations=100)  # Reduced for faster testing
    results = analyzer.analyze_configurations()
    
    print(f"\nCOMPLETE RESULTS (Defenders | Decoys):")
    for result in results:
        print(f"   {result['defenders']:2d} + {result['decoys']:2d} nodes: "
              f"{result['victory_rate']:.3f} victory rate, "
              f"{result['mean_cost']:.3f} cost, {result['efficiency']:.3f} efficiency")

    # Find optimal configuration considering deception
    optimal_idx = np.argmax([r['efficiency'] for r in results])
    optimal_config = results[optimal_idx]

    print(f"\nOPTIMAL DEFENSE WITH DECEPTION:")
    print(f"   Real Defenders: {optimal_config['defenders']}")
    print(f"   Decoy Nodes: {optimal_config['decoys']}")
    print(f"   Total Nodes: {optimal_config['total_nodes']}")
    print(f"   Victory Rate: {optimal_config['victory_rate']:.3f}")
    print(f"   Mean Defense Cost: {optimal_config['mean_cost']:.3f}")
    print(f"   Cost-Effectiveness: {optimal_config['efficiency']:.3f}")    

    # Generate visualizations
    DefenseVisualizer.plot_optimization_results(results, 'defense-optimization-analysis.png')
    DefenseVisualizer.plot_deception_strategy(results, 'deception-strategy-analysis.png')
    DefenseVisualizer.plot_defense_configuration(optimal_config['defenders'])
    DefenseVisualizer.plot_cost_analysis(results, 'cost-analysis.png')


    print(f"\nGenerating trajectory plot for optimal configuration ({optimal_config['defenders']} + {optimal_config['decoys']})...")
    example_sim = DefenseSimulation(n_defenders=optimal_config['defenders'], 
                                   n_decoys=optimal_config['decoys'])
    example_result = example_sim.run_simulation()
    DefenseVisualizer.plot_attack_trajectory(example_sim, example_result, 'attack-trajectory.png')


    print(f"\nRECOMMENDATION:")
    print(f"   Deploy {optimal_config['defenders']} real defenders + {optimal_config['decoys']} decoys configuration")
    print(f"   Expected infrastructure survival: {optimal_config['victory_rate']:.1%}")
    print(f"   Cost-effective security investment")
