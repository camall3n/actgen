from actgen.wrappers.torch_interface import test_torch_interface
from actgen.wrappers.duplicate_actions import test_duplicate_action_env
from actgen.train import main as train
from actgen.test import main as testing


def test_train():
    train(test=True)


def test_testing():
    testing(test=True)
