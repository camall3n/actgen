# actgen
Explicit Action Generalization

This is the official code for paper [Characterizing the Action-Generalization Gap in Deep Q-Learning](https://arxiv.org/abs/2205.05588).

## Installing
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Training a Q-Learning agent 
```bash
python3 -m actgen.train --tag experiment_name --env_name atari_env --agent dqn [--options] 
```
use different option flags to use different action augmentation themes

## Plotting the result
plot the results of a specific training run using the same `--tag` specified in training
```bash
python3 -m actgen.plot --tag experiment_name --plot_atari
```
