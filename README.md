# actgen
Action Generalization

## Installing
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Running
```bash
python3 -m actgen.train
```

## Experiment 1: update q(s,a) and observe q(s, \tilde(a))
```bash
python3 -m actgen.change_q
```
## Testing
```bash
python3 -m actgen.test
```