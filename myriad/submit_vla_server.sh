#!/bin/bash -l
#$ -S /bin/bash
#$ -N openvla_server
#$ -l h_rt=08:00:00
#$ -l gpu=1
#$ -ac allow=L
#$ -l mem=40G
#$ -l tmpfs=20G
#$ -wd /home/ucabj89
#$ -o logs/openvla_server.o
#$ -e logs/openvla_server.e

mkdir -p "$HOME/logs"

echo "=== Job started: $(date) ==="
echo "=== Node: $(hostname) ==="
nvidia-smi

module purge
module load pytorch/2.1.0/gpu

source "$HOME/venvs/openvla/bin/activate"

export OPENVLA_MODEL_PATH="$HOME/openvla-7b"
export OPENVLA_PORT=8000

echo "$(hostname)" > "$HOME/openvla_node.txt"
echo "=== Server starting on $(hostname):$OPENVLA_PORT ==="

python "$HOME/openvla_server.py"
