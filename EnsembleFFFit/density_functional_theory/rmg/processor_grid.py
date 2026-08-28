import math
import itertools
import numpy as np


def get_min_middle_max_indices(values):
    """Get indices of the minimum, middle, and maximum values in an array of length 3."""
    max_idx, min_idx = np.argmax(values), np.argmin(values)
    if max_idx == min_idx:  # If all values are the same
        return 0, 1, 2
    middle_idx = next(iter(set(range(3)) - {max_idx, min_idx}))
    return min_idx, middle_idx, max_idx


def generate_gpu_mapping(value, gpus_per_node=8):
    """Generate possible shifts for GPU distribution based on input grid density values."""
    if value < gpus_per_node:
        return np.arange(0, gpus_per_node - value + 1)

    low_bound = -1 * (gpus_per_node - 1)
    high_bound = gpus_per_node

    return np.arange(low_bound, high_bound + 1)


def weighting_function(x, sigma, x_ideal, alpha=0.2, beta=0.1, tolerance=1e-1):
    """Compute the weighting function to optimize GPU mapping selection."""
    penalty_factor = alpha if x < x_ideal else beta
    penalty = penalty_factor * (np.abs(x - x_ideal) / x_ideal)
    return penalty * max(sigma, tolerance)  # Prevent zero sigma issues


def evaluate_combination(grid_values, modified_grid, min_idx, mid_idx, max_idx,
                         target_nodes, gpus_per_node, grid_divisibility_exponent, fix_nodes):
    """Evaluate a specific GPU distribution combination and compute its weighted value."""
    total_gpus = np.prod(modified_grid)

    if 0 in modified_grid: # or total_gpus % gpus_per_node != 0:
        return None, None  # Invalid configuration

    if not total_gpus % (2**grid_divisibility_exponent) == 0:
            return None, None

    if fix_nodes == True:
        # Check that the solved gpus can be mapped to the target nodes
        if total_gpus / gpus_per_node > target_nodes:
            return None, None

    per_grid_density = grid_values / modified_grid
    sigma = np.std(per_grid_density)
    target_gpus = target_nodes * gpus_per_node
    weighted_value = weighting_function(total_gpus, sigma, target_gpus)

    return modified_grid.tolist(), weighted_value


def find_best_divisible_combination(grid_values, initial_grid, combinations,
                                    min_idx, mid_idx, max_idx, target_nodes,
                                    gpus_per_node, grid_divisibility_exponent,
                                    fix_nodes):
    """Find the optimal GPU distribution that minimizes the weighting function."""
    best_combo, best_value = None, float('inf')

    for combo in combinations:
        adjustment = [0, 0, 0]
        adjustment[max_idx] = combo[0]
        adjustment[mid_idx] = combo[1]
        adjustment[min_idx] = combo[2]
        modified_grid = initial_grid + np.array(adjustment)

        new_combo, weighted_value = evaluate_combination(
            grid_values, modified_grid, min_idx, mid_idx, max_idx,
            target_nodes, gpus_per_node, grid_divisibility_exponent,
            fix_nodes
        )

        if new_combo and weighted_value < best_value:
            best_value = weighted_value
            best_combo = new_combo

    return best_combo, best_value if best_value != float('inf') else None


def get_processor_grid(grid_values, target_nodes, gpus_per_node=8, kpoint_multiplier=1, grid_divisibility_exponent=3, fix_nodes=False):
    """
    Determine the optimal processor grid distribution given grid values and constraints.

    :param grid_values: 3D grid values representing x, y, and z processor grid densities.
    :param target_nodes: Total number of desired nodes.
    :param gpus_per_node: Number of GPUs per node.
    :param kpoint_multiplier: How many-fold to replicate the processor grid across
        k-point-parallel groups (RMG's own 'kpoint_distribution' input keyword, i.e.
        pct.pe_kpoint -- NOT the total number of k-points). 1 means no k-point
        parallelism: a single grid replica processes every k-point sequentially,
        and total_nodes below reflects the processor grid alone.
    :param grid_divisibility_exponent: Exponential factor for grid divisibility.
    :return: Optimal processor grid as a string and required number of nodes.
    """
    # Normalize grid values based on the smallest grid size and gpus_per_node
    normalized_grid = np.array(grid_values) / np.min(grid_values)
    scaling_factor = math.ceil(np.prod(normalized_grid) / gpus_per_node)
    renormalized_grid = np.array([math.ceil(g / scaling_factor) for g in normalized_grid])
    min_idx, mid_idx, max_idx = get_min_middle_max_indices(renormalized_grid)

    # Set the upper limit for the lowest processor grid dimension
    max_grid_factor = int(np.ceil((target_nodes * gpus_per_node) ** (1/3)))

    # Initialize starting processor grid
    best_processor_grid = np.floor(renormalized_grid).astype(int)
    best_function_value = float('inf')

    while max_grid_factor > 0:
        # Generate candidate grid by scaling normalized grid
        candidate_grid = np.round(renormalized_grid * max_grid_factor).astype(int)
        total_gpus = np.prod(candidate_grid)
        max_shift_factor = int(np.ceil((max_grid_factor / 2) * np.ceil(renormalized_grid[max_idx])))

        # Generate possible shifts for GPU allocation
        max_shifts = generate_gpu_mapping(candidate_grid[max_idx], max_shift_factor)
        mid_shifts = generate_gpu_mapping(candidate_grid[mid_idx], max_shift_factor)
        min_shifts = np.array([-1, 0, 1])  # Small adjustments

        # Generate all possible shift combinations
        combinations = itertools.product(max_shifts, mid_shifts, min_shifts)

        # Find the best valid GPU grid configuration
        best_combo, best_value = find_best_divisible_combination(
            grid_values, candidate_grid, combinations, min_idx, mid_idx, max_idx,
            target_nodes, gpus_per_node, grid_divisibility_exponent, fix_nodes
        )

        # Update best configuration if improvement is found
        if best_value is not None and best_value < best_function_value:
            best_processor_grid, best_function_value = best_combo, best_value
        max_grid_factor -= 1  # Reduce search space

    # Compute required number of nodes
    total_nodes = math.ceil(np.prod(best_processor_grid) / gpus_per_node)
    total_nodes = max(1, kpoint_multiplier * total_nodes)

    return ' '.join(map(str, best_processor_grid)), total_nodes
