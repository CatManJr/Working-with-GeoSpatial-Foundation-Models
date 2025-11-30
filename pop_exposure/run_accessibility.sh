#!/bin/bash

# Run accessibility analysis for different bandwidths and different number of agents
for bandwidth in 250 500 1000 2500
do
    for num_agents in 1 2 3 4 5
    do
        echo "Running accessibility analysis with bandwidth: $bandwidth and num_agents: $num_agents"
        uv run pop_exposure/accessibility.py --bandwidth $bandwidth --num_agents $num_agents
    done
done