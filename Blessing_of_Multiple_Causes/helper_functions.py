import numpy as np
import pandas as pd
import pymc as pm
from scipy import stats

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

