#!/bin/bash

# Run accessibility analysis for different bandwidths and different number of agents
for bandwidth in 250 500 1000 2500
    do
        echo "Running accessibility analysis with bandwidth: $bandwidth
        uv run pop_exposure/accessibility.py --bandwidth $bandwidth
    done
done