from actgen.wrappers.torch_interface import test_torch_interface
from actgen.wrappers.duplicate_actions import test_duplicate_action_env
from actgen.train import main as train
from actgen.evaluate_dqn import main as evaluate_dqn
from actgen.change_q import main as change_q


def test_train():
    train(test=True)


def test_evaluate_dqn():
    evaluate_dqn(test=True)


def test_change_q():
    change_q(test=True)
