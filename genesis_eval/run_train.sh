source ../.venv/bin/activate
python -u train.py --max_iterations=${1:-1001} --num_envs=${2:-1024}
