from actgen.wrappers.torch_interface import test_torch_interface
from actgen.wrappers.duplicate_actions import test_duplicate_action_env
from actgen.train import main as train

def test_train():
    train(test=True)
