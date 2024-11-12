import numpy as np

# Constants and settings
NUM_COUNTIES = 5
#BETAS = [0.1, 0.2, 0.3, 0.5, 0.8, 1]  # List of beta values for fairness constraints
# Define the range of beta values
start_beta = 0.01
end_beta = 0.2
step_beta = 0.01

# Generate the list of beta values
BETAS = np.arange(start_beta, end_beta + step_beta, step_beta)
C_UI = np.array([5, 4, 3, 2, 1])  # Cost per unit of unmet demand for each county

# Additional supplies and their probabilities
A_realizations = np.array([100, 150, 200, 250])
A_probabilities = np.array([0.25, 0.25, 0.25, 0.25])

# The maximum supply should be the initial supply plus the maximum possible additional supply
initial_supply = 10
MAX_SUPPLY = initial_supply + max(A_realizations)  # Update this to ensure it covers all scenarios

# Demand realizations and probabilities for each of the five counties
D_realizations = [
    np.array([10, 15, 20, 25, 30]),
    np.array([30, 45, 60, 75, 80]),
    np.array([80, 95, 100, 105, 110]),
    np.array([115, 120, 125, 130, 135]),
    np.array([125, 130, 135, 140, 145])
]
D_probabilities = [
    np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    np.array([0.2, 0.2, 0.2, 0.2, 0.2]),
    np.array([0.2, 0.2, 0.2, 0.2, 0.2])
]

# Dynamic programming table and allocation tracker initialization
dp = np.full((NUM_COUNTIES + 1, MAX_SUPPLY + 1, len(BETAS)), np.inf)
dp[NUM_COUNTIES, :, :] = 0  # Base case: no cost for the terminal condition
optimal_allocations = np.zeros((NUM_COUNTIES, MAX_SUPPLY + 1, len(BETAS), NUM_COUNTIES))

# Adjust the dynamic programming algorithm to store allocations
for i in reversed(range(NUM_COUNTIES)):
    for Si in range(MAX_SUPPLY + 1):
        for beta_index, beta in enumerate(BETAS):
            total_allocation = 0  # Track the total allocated amount
            for a_idx, a in enumerate(A_realizations):
                for d_idx, d in enumerate(D_realizations[i]):
                    d_prob = D_probabilities[i][d_idx]
                    for xi in range(min(Si + a, d) + 1):
                        new_Si = Si + a - xi
                        if new_Si <= MAX_SUPPLY:  # Ensure new supply index is within bounds
                            supply_prop = xi / (Si + a)
                            demand_prop = d / d_prob
                            if supply_prop - demand_prop <= beta:
                                total_allocation += xi
                                if total_allocation <= MAX_SUPPLY:  # Check if total allocation exceeds total supply
                                    cost = C_UI[i] * (d - xi) * d_prob
                                    future_cost = dp[i + 1, new_Si, beta_index]
                                    expected_cost = cost + future_cost
                                    if expected_cost * A_probabilities[a_idx] < dp[i, Si, beta_index]:
                                        dp[i, Si, beta_index] = expected_cost * A_probabilities[a_idx]
                                        optimal_allocations[i, Si, beta_index, i] = xi  # Store optimal allocation for county i


# Print optimal expected costs and allocations for each beta
for beta_index, beta in enumerate(BETAS):
    print(f"\nFor beta = {beta}:")
    print(f"Optimal Expected Cost: {dp[0, initial_supply, beta_index]}")
    print("Optimal Allocations:")
    for i in range(NUM_COUNTIES):
        optimal_xi = optimal_allocations[i, initial_supply, beta_index, i]  # Print optimal allocation for county i
        print(f"County {i + 1}: Allocated {optimal_xi} units")
