betterfish_example.py
```python
"""
This is a recreation of the default example in the README of fishchips
https://github.com/xzackli/fishchips-public

Fisher matrix calculation is 10x faster on my machine for this example. (0.01s vs 0.1s)
CLASS computations are sped up with multiprocessing, so the improvement depends heavily on the machine. On mine it is ~3x. (7s vs 18s)
"""

from betterfish import Observables, CMB_S4, white_noise
import numpy as np
import fishchips.util
import matplotlib.pyplot as plot

obs = Observables(
        classy_template={'output': 'tCl pCl lCl',
                'l_max_scalars': 2500,
                'lensing': 'yes'},
        params=['A_s', 'n_s', 'tau_reio'],
        fiducial=[2.1e-9, 0.968, 0.066],
        dx=[1.e-10, 2.e-02, 1.e-02]
    )

obs.compute(l_max=2500)

experiment = CMB_S4(noise_curves=white_noise(7., 33., 56.), l_max=2500)
f = experiment.get_fisher(obs)

cov = np.linalg.inv(f)
fishchips.util.plot_triangle(obs, cov)
plot.show()
```