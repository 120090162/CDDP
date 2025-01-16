from hopper_test import hopper
from ciddp import CDDP
import numpy as np

import warnings
warnings.filterwarnings("ignore", message="delta_grad == 0.0. Check if the approximated function is linear.")
warnings.filterwarnings("ignore", message="UserWarning")


if __name__ == '__main__':
    system = hopper()
    system.set_cost(np.zeros((7, 7)), system.dt * np.identity(2))
    Q_f = np.identity(7)
    Q_f[np.eye(7).astype(np.bool)] = np.array([50,50,50,10,10,10,0])
    system.set_final_cost(Q_f)

    solver = CDDP(system, np.array([0,1,0,0,0,0,0.9]), horizon=300, asymm=False)
    #solve for initial trajectories
    system.set_goal(np.array([0, 2, 0, 0, 0, 0, 0]))
    solver.backward_pass()
    solver.forward_pass()

    system.set_goal(np.array([5, 1, 0, 0, 0, 0, 0]))
    for i in range(20):
        solver.backward_pass()
        solver.forward_pass()
    solver.system.draw_trajectories(solver.x_trajectories)