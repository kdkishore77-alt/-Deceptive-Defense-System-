''' Sensitivity analysis'''
import numpy as np
import pandas as pd
from mesa import Model
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm  # For progress bars, install with `pip install tqdm`
from scipy.spatial.distance import cdist
from deception import DefenseSimulation


class SensitivityAnalyzer:
    """
    Class to perform a comprehensive sensitivity analysis on key parameters
    of the DefenseSimulation model.
    """
    
    def __init__(self, n_simulations=100, base_config=None, base_seed=1234):
        """
        Initialize the sensitivity analyzer.
        
        Parameters:
        n_simulations (int): Number of simulations to run per parameter set.
        base_config (dict): Dictionary of base parameter values. If None, uses defaults.
        """
        self.n_simulations = n_simulations
        
        # Define the base configuration (default parameters from DefenseSimulation)
        if base_config is None:
            self.base_config = {
                'n_defenders': 6,     # Optimal number of real defenders
                'n_decoys': 2,        # Optimal number of decoys
                'infra_radius': 1.0,
                'attacker_strength': 0.3,
                'defense_multiplier': 3.0,
                'engagement_radius': 0.12,
                'decoy_cost': 0.01,
                'decoy_efficacy_range': (0.2, 0.6),
                'decoy_visibility': 1.8,
                'attack_power': 0.04
            }
        else:
            self.base_config = base_config
            
        self.results_df = pd.DataFrame()
    

    def create_parameter_grid(self):
        """
        Create a grid of parameters to test for sensitivity analysis.
        Each parameter is varied by ±20% and ±40% around its base value.
        """
        param_grid = {}
        
        # Define variations for each parameter (±20%, ±40%)
        variations = [-0.4, -0.2, 0.0, +0.2, +0.4]
        
        # Parameters to test
        test_params = [
            'attacker_strength',
            'defense_multiplier', 
            'engagement_radius',
            'decoy_efficacy_range',
            'n_decoys'
        ]
        
        for param in test_params:
            if param == 'decoy_efficacy_range':
                base_low, base_high = self.base_config['decoy_efficacy_range']
                param_values = [
                    (base_low * (1+v), base_high * (1+v)) for v in variations
                ]
            
            elif param == 'n_decoys':
                base_val = self.base_config['n_decoys']
                param_values = [int(max(0, base_val * (1 + v))) for v in variations]
                # Ensure unique integer values
                param_values = sorted(set(param_values))
            
            else:
                base_val = self.base_config[param]
                param_values = [base_val * (1 + v) for v in variations]
            
            param_grid[param] = param_values
        
        return param_grid


    
    def run_sensitivity_analysis(self):
        """
        Execute the full sensitivity analysis across the parameter grid.
        One-at-a-time (OAT) local sensitivity: varies each parameter independently.
        """
        param_grid = self.create_parameter_grid()
        all_results = []

        # Get list of parameters to test
        test_params = list(param_grid.keys())

        print("=== SENSITIVITY ANALYSIS ===")
        print(f"Running {self.n_simulations} simulations per parameter set...")
        print(f"Testing parameters: {test_params}")

        # Iterate through each parameter
        for param in tqdm(test_params, desc="Overall Progress"):
            param_values = param_grid[param]

            for value in tqdm(param_values, desc=f"Testing {param}", leave=False):
                # Create configuration for this test
                config = self.base_config.copy()
                config[param] = value

                # Run multiple simulations
                victory_flags = []
                defense_costs = []

                for run_idx in range(self.n_simulations):
                    # Unique but reproducible seed
                    seed = hash((param, str(value), run_idx)) % (2**32)

                    sim = DefenseSimulation(
                        n_defenders=config['n_defenders'],
                        n_decoys=config['n_decoys'],
                        infra_radius=config['infra_radius'],
                        decoy_cost=config['decoy_cost'],
                        decoy_efficacy_range=config['decoy_efficacy_range'],
                        decoy_visibility=config['decoy_visibility'],
                        random_state=seed
                    )

                    # Override continuous parameters directly
                    sim.attacker_strength = config['attacker_strength']
                    sim.defense_multiplier = config['defense_multiplier']
                    sim.engagement_radius = config['engagement_radius'] * config['infra_radius']
                    sim.attack_power = config['attack_power']

                    # Run simulation
                    result = sim.run_simulation()

                    # Collect results
                    if result['status'] in ['attacker_defeated', 'engagement_timeout']:
                        victory_flags.append(1)
                    else:
                        victory_flags.append(0)

                    defense_costs.append(result['defense_cost'])

                # --- summary statistics ---
                victory_mean = np.mean(victory_flags)
                victory_std = np.std(victory_flags, ddof=1)
                victory_se = victory_std / np.sqrt(self.n_simulations)

                cost_mean = np.mean(defense_costs)
                cost_std = np.std(defense_costs, ddof=1)
                cost_se = cost_std / np.sqrt(self.n_simulations)

                # Store results for this parameter value
                all_results.append({
                    'parameter': param,
                    'value': value,
                    'victory_rate': victory_mean,
                    'victory_se': victory_se,
                    'mean_cost': cost_mean,
                    'cost_se': cost_se,
                    'n_simulations': self.n_simulations
                })

        # Convert to DataFrame
        self.results_df = pd.DataFrame(all_results)
        return self.results_df
    


    def plot_sensitivity_analysis(self, save_path='sensitivity-analysis.png'):
        """
        Create comprehensive sensitivity analysis plots with error bars.
        """
        if self.results_df.empty:
            print("No results to plot. Run analysis first.")
            return
        
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # --- Plot 1: Victory Rate Sensitivity ---
        ax1 = axes[0]
        parameters = self.results_df['parameter'].unique()
        
        for param in parameters:
            param_data = self.results_df[self.results_df['parameter'] == param]
            
            # Handle ranges (e.g. decoy efficacy)
            if param == 'decoy_efficacy_range':
                values = param_data['value'].apply(np.mean)
                base_value = np.mean(self.base_config['decoy_efficacy_range'])
            else:
                values = param_data['value']
                base_value = self.base_config[param]
            
            normalized_values = values / base_value

            # Plot with error bars for victory rate
            ax1.errorbar(
                normalized_values,
                param_data['victory_rate'],
                yerr=param_data.get('victory_se', None),
                fmt='o-',
                linewidth=2.5,
                markersize=8,
                capsize=4,
                label=param.replace('_', ' ').title()

            )
        
        ax1.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1.0)')
        ax1.set_xlabel('Parameter Value (Normalized to Baseline)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Victory Rate ± SE', fontsize=12, fontweight='bold')
        ax1.set_title('Sensitivity of Victory Rate to Parameter Changes', fontsize=14, fontweight='bold')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, 1.0)

        # --- Plot 2: Defense Cost Sensitivity ---
        ax2 = axes[1]
        for param in parameters:
            param_data = self.results_df[self.results_df['parameter'] == param]
            
            if param == 'decoy_efficacy_range':
                values = param_data['value'].apply(np.mean)
                base_value = np.mean(self.base_config['decoy_efficacy_range'])
            else:
                values = param_data['value']
                base_value = self.base_config[param]
            
            normalized_values = values / base_value

            # Plot with error bars for mean cost
            ax2.errorbar(
                normalized_values,
                param_data['mean_cost'],
                yerr=param_data.get('cost_se', None),
                fmt='s-',
                linewidth=2.5,
                markersize=8,
                capsize=4,
                label=param.replace('_', ' ').title()

            )
        
        ax2.axvline(1.0, color='red', linestyle='--', alpha=0.7, label='Baseline (1.0)')
        ax2.set_xlabel('Parameter Value (Normalized to Baseline)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Mean Defense Cost ± SE', fontsize=12, fontweight='bold')
        ax2.set_title('Sensitivity of Defense Cost to Parameter Changes', fontsize=14, fontweight='bold')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig, axes


   
    def calculate_sensitivity_indices(self):
        """
        Calculate quantitative sensitivity indices for each parameter.
        Returns a DataFrame with sensitivity metrics.
        """
        if self.results_df.empty:
            print("No results to analyze. Run analysis first.")
            return None
        
        sensitivity_results = []
        
        for param in self.results_df['parameter'].unique():
            param_data = self.results_df[self.results_df['parameter'] == param].copy()
            
            if param == 'decoy_efficacy_range':
                # Use mean of (low, high) tuples
                values = np.array([np.mean(val) for val in param_data['value'].values])
                base_val = np.mean(self.base_config[param])
                # Find baseline row by comparing means
                baseline_mask = np.isclose(values, base_val)
                if not baseline_mask.any():
                    continue
                baseline_victory = param_data['victory_rate'].values[baseline_mask][0]
                baseline_cost = param_data['mean_cost'].values[baseline_mask][0]
            else:
                values = param_data['value'].values
                base_val = self.base_config[param]
                baseline_data = param_data[param_data['value'] == base_val]
                if len(baseline_data) == 0:
                    continue
                baseline_victory = baseline_data['victory_rate'].values[0]
                baseline_cost = baseline_data['mean_cost'].values[0]
            
            victory_rates = param_data['victory_rate'].values
            costs = param_data['mean_cost'].values

            # Sensitivity calculations (avoid divide-by-zero)
            if baseline_victory > 0:
                victory_sensitivity = np.gradient(victory_rates, values) * (base_val / baseline_victory)
            else:
                victory_sensitivity = np.gradient(victory_rates, values) * base_val

            if baseline_cost > 0:
                cost_sensitivity = np.gradient(costs, values) * (base_val / baseline_cost)
            else:
                cost_sensitivity = np.gradient(costs, values) * base_val

            sensitivity_results.append({
                'parameter': param,
                'victory_sensitivity': np.mean(np.abs(victory_sensitivity)),
                'cost_sensitivity': np.mean(np.abs(cost_sensitivity)),
                'most_sensitive_victory': np.argmax(np.abs(victory_sensitivity)),
                'most_sensitive_cost': np.argmax(np.abs(cost_sensitivity))
            })
        
        return pd.DataFrame(sensitivity_results).sort_values('victory_sensitivity', ascending=False)




if __name__ == "__main__":
    print("Running Sensitivity Analysis for Defense Simulation with Deception...")
    
    # Initialize analyzer
    analyzer = SensitivityAnalyzer(n_simulations=500, base_seed=42)  # ↑ Increase reps, fix seed for reproducibility
    
    # Run sensitivity analysis
    results = analyzer.run_sensitivity_analysis()
    
    # Display results with error bars
    print("\n=== SENSITIVITY ANALYSIS RESULTS ===")
    # Show rounded SE for readability
    display_cols = ['parameter', 'value', 'victory_rate', 'victory_se', 'mean_cost', 'cost_se']
    print(results[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Calculate and display sensitivity indices
    sensitivity_indices = analyzer.calculate_sensitivity_indices()
    if sensitivity_indices is not None:
        print("\n=== QUANTITATIVE SENSITIVITY INDICES ===")
        print("Parameters ranked by impact on Victory Rate:")
        print(sensitivity_indices[['parameter', 'victory_sensitivity', 'cost_sensitivity']]
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    
    # Create plots
    analyzer.plot_sensitivity_analysis('defense-sensitivity-analysis.png')
    
    print("\nAnalysis complete. Results saved to 'defense-sensitivity-analysis.png'")
