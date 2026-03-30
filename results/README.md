# Experimental Results

## Aggregated Results

`aggregated_results.json` contains the averaged results across all instances, as reported in the paper tables. This includes:

- **Small instances** (30 instances, n=10, m∈{2,3,5}): All 7 methods compared across 3 failure rates
- **Medium instances, 5 machines** (8–9 instances per rate): Full hybrid comparison (GA-std vs RL-GA-GNN)
- **Medium instances, 8 machines** (8–9 instances per rate): GA-std + baselines, partial hybrid data
- **Medium instances, 10 machines** (6 instances per rate): GA-std + baselines only

## Per-Instance Results

The `lambda_0.01/`, `lambda_0.05/`, and `lambda_0.10/` directories contain individual JSON result files for each instance-rate combination. Each file includes per-method metrics (makespan, CVaR, on-time percentage, CPU time) and the best schedule found.

To regenerate per-instance results, see the "Reproducing Paper Results" section in the main README.

## Key Findings

| Comparison | Small (n=10) | Medium 5m (n=50) |
|---|---|---|
| GA-std vs best heuristic | −8% to −13% | −8% to −9% |
| Hybrid vs GA-std | ≈0% | −0.4% to −0.7% |
| Hybrid CVaR vs GA-std CVaR | ≈0% | −0.5% to −0.8% |
