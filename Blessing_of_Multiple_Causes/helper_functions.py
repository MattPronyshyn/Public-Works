import numpy as np
import pandas as pd
import arviz as az
import scipy.stats as stats
import pymc as pm
import matplotlib.pyplot as plt

def create_offensive_defensive_datasets(entries_df, player_teams_df):
    """
    Create offensive and defensive datasets based on player-team alignment.

    This function processes a DataFrame containing game entries along with player-specific data
    and another DataFrame that maps players to their respective teams. It returns two DataFrames:
    one for offensive contributions and one for defensive contributions. In the offensive dataset,
    each player's data is retained only if they are on the same team as indicated in the entry.
    Conversely, in the defensive dataset, the player's data is retained only if they are on the opposing team.
    After processing, the 'Team' column is dropped from both datasets.

    Parameters
    ----------
    entries_df : pandas.DataFrame
        DataFrame containing game entries with a 'Team' column indicating the team associated with each entry.
        Additionally, it includes columns corresponding to individual players' contributions.
    
    player_teams_df : pandas.DataFrame
        DataFrame mapping players to teams. It must contain the columns 'player_name' for player identifiers
        and 'team_name' for the corresponding team.
    
    Returns
    -------
    offensive_df : pandas.DataFrame
        A modified version of entries_df for only players on the offensive side.
    
    defensive_df : pandas.DataFrame
        A modified version of entries_df for only players on the defensive side.
    """
    # Create empty DataFrames for offensive and defensive datasets with the same structure as entries_df
    offensive_df = entries_df.copy()
    defensive_df = entries_df.copy()
    
    # Go through each player in the player_teams_df and update the offensive and defensive datasets accordingly
    for _, row in player_teams_df.iterrows():
        player = row['player_name']
        team = row['team_name']
        
        # Check if the player is on the offensive team for each entry
        is_offensive = entries_df['Team'] == team
        is_defensive = ~is_offensive
        
        # Update the offensive_df by retaining player's presence only if they're on the offensive team
        offensive_df[player] = offensive_df[player].where(is_offensive, other=0)
        
        # Update the defensive_df by retaining player's presence only if they're on the defensive team
        defensive_df[player] = defensive_df[player].where(is_defensive, other=0)
    
    return offensive_df.drop("Team", axis=1), defensive_df.drop("Team", axis=1)

def discrepancy_function(zone_entry_factors,player_factors, data, mask, num_heldout):
    data = np.array(data)
    # Initialize the array to store the observed log probability
    observed_log_prob = np.zeros([zone_entry_factors.shape[1],zone_entry_factors.shape[0]]) # zone entries x num samples

    for i in range(zone_entry_factors.shape[0]):
        # Calculate the rate matrix
        rate = np.dot(zone_entry_factors[i], player_factors[i].T)
        
        # Calculate the log probability of the held-out data under the model
        held_out_rate = rate[~mask].reshape(rate.shape[0], num_heldout)
        held_out_data = data[~mask].reshape(data.shape[0], num_heldout)
        
        observed_log_prob[:,i] = stats.poisson.logpmf(held_out_data, held_out_rate).sum(axis=1)
        
    # Compute the discrepancy function for the actual held-out data
    discrepancy = np.mean(observed_log_prob, axis=1)
    
    return discrepancy

def predictive_score(offensive_dataset, mask, num_mask, num_iter=100, random_seed=2):
    """
    Fits a Poisson latent‑factor model to the masked offensive_dataset, generates posterior predictive 
    samples for the held‑out entries, computes a discrepancy statistic, and returns the overall predictive score.

    Parameters
    ----------
    offensive_dataset : array‑like, shape (n_zones, n_players)
        The observed count matrix.
    mask : ndarray of bool, same shape as offensive_dataset
        Boolean mask indicating which entries to include in fitting (True) vs held out (False).
    num_mask : int
        Number of held‑out entries per row (used by discrepancy_function).
    discrepancy_function : callable
        Function(discrete_factors, player_factors, data, mask, num_mask) -> array of shape (n_zones,)
        Computes a scalar discrepancy statistic for each row.
    num_iter : int, default=100
        Number of posterior predictive replicates.
    random_seed : int, default=2
        Random seed for reproducibility.

    Returns
    -------
    float
        Mean predictive score across all rows (higher is better).
    """
    np.random.seed(random_seed)

    with pm.Model() as model:
        zone_entry_factors = pm.Gamma('zone_entry_factors', alpha=1, beta=1,
                                      shape=(offensive_dataset.shape[0], 2))
        player_factors = pm.Gamma('player_factors', alpha=1, beta=1,
                                  shape=(offensive_dataset.shape[1], 2))
        rate = pm.math.dot(zone_entry_factors, player_factors.T)
        pm.Poisson('observed_counts', mu=rate[mask], observed=np.array(offensive_dataset)[mask])
        trace = pm.sample(return_inferencedata=True)

    # Extract and reshape posterior factors
    z = trace.posterior['zone_entry_factors'].values
    w = trace.posterior['player_factors'].values
    n_samples = z.shape[0] * z.shape[1]
    zone_entry_factors = z.reshape((n_samples, *z.shape[2:]))
    player_factors = w.reshape((n_samples, *w.shape[2:]))

    # Compute actual discrepancy
    actual = discrepancy_function(zone_entry_factors, player_factors, offensive_dataset, mask, num_mask)

    # Posterior predictive draws
    replicated = np.zeros((offensive_dataset.shape[0], num_iter))
    for i in range(num_iter):
        idx = np.random.randint(n_samples)
        pred_rate = zone_entry_factors[idx] @ player_factors[idx].T
        pred_counts = np.random.poisson(pred_rate)
        replicated[:, i] = discrepancy_function(zone_entry_factors, player_factors, pred_counts, mask, num_mask)

    score = np.mean(replicated < actual[:, None], axis=1)
    return float(score.mean())

# Calculate summary statistics for coefficients
players_positions = {
    'Blayre Turnbull': 'F',
    'Hilary Knight': 'F',
    'Sarah Fillier': 'F',
    'Hannah Bilka': 'F',
    'Brianne Jenner': 'F',
    'Alex Carpenter': 'F',
    'Natalie Buchbinder': 'D',
    'Jessie Eldridge': 'F',
    'Grace Zumwinkle': 'F',
    'Jamie Lee Rattray': 'F',
    'Emma Maltais': 'F',
    'Cayla Barnes': 'D',
    'Ashton Bell': 'D',
    'Emily Clark': 'F',
    'Haley Winn': 'D',
    'Abbey Murphy': 'F',
    'Sarah Nurse': 'F',
    'Megan Keller': 'D',
    'Lacey Eden': 'F',
    'Renata Fast': 'D',
    'Laura Stacey': 'F',
    'Julia Gosling': 'F',
    'Jamie Bourbonnais': 'D',
    'Tessa Janecke': 'F',
    'Abby Roque': 'F',
    'Taylor Heise': 'F',
    'Ella Shelton': 'D',
    'Savannah Harmon': 'D',
    'Marie-Philip Poulin': 'F',
    'Hayley Scamurra': 'F',
    'Gabbie Hughes': 'F',
    'Jessica DiGirolamo': 'D',
    "Kristin O'Neill": 'F',
    'Erin Ambrose': 'D',
    'Britta Curl': 'F',
    'Kelly Pannek': 'F',
    'Jocelyne Larocque': 'D',
    'Rory Guilday': 'D',
    'Sophie Jaques': 'D',
    'Laila Edwards': 'F',
    'Jennifer Gardiner': 'F',
    'Caroline Harvey': 'D',
    'Loren Gabel': 'F',
    'Kirsten Simms': 'F',
    'Danielle Serdachny': 'F',
    'Anna Wilgren': 'D',
    'Anne Cherkowski': 'F',
    'Allyson Simpson': 'D'
}

def plot_rankings(trace, dataset):
    summary_offensive = az.summary(trace, var_names=['offensive_coeff'], hdi_prob=0.94)
    summary_offensive = summary_offensive.assign(names=dataset.columns)

    summary_defensive = az.summary(trace, var_names=['defensive_coeff'], hdi_prob=0.94)
    summary_defensive = summary_defensive.assign(names=dataset.columns)

    # Sort the summary by mean values
    summary_sorted_offensive = summary_offensive.sort_values(by='mean')
    summary_sorted_defensive = summary_defensive.sort_values(by='mean')

    # Extract means and HDI intervals after sorting
    means_offensive, means_defensive = summary_sorted_offensive['mean'], summary_sorted_defensive['mean']
    hdi_lower_offensive, hdi_lower_defensive = summary_sorted_offensive['hdi_3%'], summary_sorted_defensive['hdi_3%']
    hdi_upper_offensive, hdi_upper_defensive = summary_sorted_offensive['hdi_97%'] , summary_sorted_defensive['hdi_97%']

    # Calculate errors from mean to lower and upper bounds
    errors_offensive = [means_offensive - hdi_lower_offensive, hdi_upper_offensive - means_offensive]
    errors_defensive = [means_defensive - hdi_lower_defensive, hdi_upper_defensive - means_defensive]


    plt.figure(figsize=(12, 8))
    for i, player in enumerate(summary_sorted_offensive['names']):
        color = 'blue' if players_positions.get(player, 'F') == 'F' else 'red'
        plt.errorbar(i, means_offensive[i], yerr=[[errors_offensive[0][i]], [errors_offensive[1][i]]], 
                     fmt='o', capsize=5, capthick=2, color=color)
    plt.xticks(range(len(summary_sorted_offensive['names'])), summary_sorted_offensive['names'], rotation='vertical')
    plt.xlabel('Player')
    plt.ylabel('Offensive Skill Estimate')
    plt.title('Offensive Skill Estimates with 94% HDI, Sorted by Mean')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 8))
    for i, player in enumerate(summary_sorted_defensive['names']):
        color = 'blue' if players_positions.get(player, 'F') == 'F' else 'red'
        plt.errorbar(i, means_defensive[i], yerr=[[errors_defensive[0][i]], [errors_defensive[1][i]]], 
                     fmt='o', capsize=5, capthick=2, color=color)
    plt.xticks(range(len(summary_sorted_defensive['names'])), summary_sorted_defensive['names'], rotation='vertical')
    plt.xlabel('Player')
    plt.ylabel('Defensive Skill Estimate')
    plt.title('Defensive Skill Estimates with 94% HDI, Sorted by Mean')
    plt.tight_layout()
    plt.show()