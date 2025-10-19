#!/usr/bin/env bash
set -euo pipefail

# Start Pokemon Showdown locally
cd /workspace/pokemon-showdown
node pokemon-showdown start --no-security &

sleep 5  # wait for server to boot

# Run training (modify as needed)
cd /workspace/gymnasium_envrionments/scripts
yes | python run.py train cli --gym showdown --domain random --task max DQN

# Save results to GCS
gsutil -m cp -r /root/cares_rl_logs gs://pokemon-rl-jbro914-rl-logs/cares_rl_logs/
