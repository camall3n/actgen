from matplotlib import pyplot as plt
import numpy as np


def build_confusion_matrix(q_deltas, num_duplicate):
    """
    builds a confusion matrix for the direction of the bumps in q_delta for one state

    :param
        q_deltas (np.array): each row corresponds to the latest q_delta from one action
                            shape = (num_actions, len(q_delta_for_one_action)) = (num_actions, num_actions)
        num_duplicate: number of set of duplicate actions
    :return
           a confusion matrix, where row corresponds to the action being updated, and col in that row correspond to
           the delta for that intervention.
           The index is arranged so that similar actions are adjacent: (a1, a2, .., an, b1, b2, .., bn, ..)
    """
    # arrange index so similar actions are adjacent
    num_actions = len(q_deltas)
    num_original_actions = int(num_actions / num_duplicate)
    new_idx = []
    for i in range(num_original_actions):
        new_idx += list(range(i, num_actions, num_original_actions))
    assert len(new_idx) == num_actions
    q_deltas = q_deltas[new_idx]

    mat = np.zeros((num_actions, num_actions))
    for action, q_delta in enumerate(q_deltas):
        # normalize q_delta by min-max scaling to [0, 1]
        q_delta = (q_delta - np.min(q_delta)) / (np.max(q_delta) - np.min(q_delta))
        # put row in matrix
        q_delta = q_delta[new_idx]
        mat[action] = q_delta
    return mat


def calc_g_score(avg_confusion_mat, num_duplicate):
    """
    calculate the two g scores based on an average confusion matrix

    :param
        avg_confusion_mat: a confusion matrix that's the average of multiple matrix over different states
        num_duplicate: number of set of duplicate actions
    :return: +g, -g
    """
    num_actions = len(avg_confusion_mat)
    num_original_actions = int(num_actions / num_duplicate)
    # matrix diagonal should be all 1's
    for i in range(num_actions):
        assert avg_confusion_mat[i, i] == 1

    # get plus_g score
    plus_g_sum = 0
    for i in range(num_original_actions):
        similar_square_block = avg_confusion_mat[i * num_duplicate: (i + 1) * num_duplicate,
                               i * num_duplicate: (i + 1) * num_duplicate]
        plus_g_sum += np.sum(similar_square_block)
    plus_g = plus_g_sum - 1 * len(avg_confusion_mat)  # subtract 1's from diagonal
    num_plus = len(avg_confusion_mat) * num_duplicate - len(
        avg_confusion_mat)  # number of entries used for the +g score
    plus_g = plus_g / num_plus  # average the sum

    # get minus_g score
    minus_g_sum = np.sum(avg_confusion_mat) - plus_g_sum
    num_minus = len(avg_confusion_mat) ** 2 - num_plus - len(
        avg_confusion_mat)  # number of entries used for the -g score
    minus_g = minus_g_sum / num_minus  # average the sum

    return plus_g, minus_g


def plot_confusion_matrix(mat, num_duplicate, num_states):
    plt.figure()
    plt.title(f"average confusion matrix: {num_duplicate} sets of duplicate action, {num_states} states")
    plt.xlabel(r'normalized $\Delta$q(s,a)')
    plt.ylabel("action being updated")
    plt.imshow(mat)
    plt.colorbar()
    plt.show()
