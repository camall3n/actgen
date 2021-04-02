import numpy as np

class PendulumExpert:
    @staticmethod
    def convert_state(state):
        y, x, theta_dot = state
        state = {
            'x': x,
            'y': y,
            'theta': np.arctan2(y, x)/np.pi,
            'theta_dot': theta_dot,
        }
        if state['theta'] > 1/2:
            state['theta'] -= 2
        mass = 1.0
        gravity = 9.81
        radius = 1.0
        velocity = theta_dot * radius
        state['potential_energy'] = 2 * mass * gravity * (y + 1)
        state['kinetic_energy'] = 0.5 * mass * (velocity)**2
        state['energy'] = state['potential_energy'] + state['kinetic_energy']
        return state

    @classmethod
    def act(cls, state):
        annotated_state = cls.convert_state(state)
        energy_difference = annotated_state['energy'] - 9.81*2*2
        if np.abs(energy_difference) > 0.1:
            a = (-np.sign(energy_difference) * np.sign(annotated_state['theta_dot']))
        else:
            a = -np.sign(annotated_state['x'])
        # remap action from {-1.0, 0.0, 1.0} -> {0, 1, 2}
        if a > 0:
            a = 2
        elif a < 0:
            a = 0
        else:
            a = 1
        return a
