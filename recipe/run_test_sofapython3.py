"""Runtime check that the bindings work, not just import.

A point mass falls under gravity through an implicit Euler solve. A Python
Controller (pybind11 trampoline for virtual overrides) counts steps and writes
to a Data field through a numpy view; the test then reads the state back and
compares the fall with the analytic value.
"""
import sys

import numpy as np
import Sofa
import Sofa.Core
import Sofa.Simulation
import SofaRuntime  # noqa: F401  (registers the default plugin search paths)

STEPS = 50
DT = 0.01
G = -9.81


class Counter(Sofa.Core.Controller):
    def __init__(self, dofs, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dofs = dofs
        self.calls = 0

    def onAnimateEndEvent(self, event):
        self.calls += 1
        if self.calls == 1:
            # write through a numpy view: give the point a lateral velocity
            with self.dofs.velocity.writeableArray() as v:
                v[0][0] = 1.0


def main():
    print(f"python {sys.version.split()[0]}  numpy {np.__version__}")
    root = Sofa.Core.Node("root")
    root.addObject("RequiredPlugin", pluginName=[
        "Sofa.Component.ODESolver.Backward",
        "Sofa.Component.LinearSolver.Iterative",
        "Sofa.Component.StateContainer",
        "Sofa.Component.Mass",
    ])
    root.addObject("DefaultAnimationLoop")
    root.dt = DT
    root.gravity = [0.0, G, 0.0]
    body = root.addChild("body")
    body.addObject("EulerImplicitSolver", rayleighStiffness=0.0, rayleighMass=0.0)
    body.addObject("CGLinearSolver", iterations=25, tolerance=1e-12, threshold=1e-12)
    dofs = body.addObject("MechanicalObject", template="Vec3d", position=[[0.0, 0.0, 0.0]])
    body.addObject("UniformMass", totalMass=1.0)
    counter = root.addObject(Counter(dofs, name="counter"))

    Sofa.Simulation.init(root)
    for _ in range(STEPS):
        Sofa.Simulation.animate(root, DT)

    p = np.asarray(dofs.position.value)[0]
    t = STEPS * DT
    print(f"steps={STEPS} controller_calls={counter.calls} position={p.round(4).tolist()} "
          f"analytic_y={0.5 * G * t * t:.4f}")

    assert counter.calls == STEPS, "Python controller callback did not fire every step"
    assert np.isfinite(p).all(), "solution went non-finite"
    # backward Euler overshoots the analytic fall slightly; 5% covers it
    assert abs(p[1] - 0.5 * G * t * t) < 0.05 * abs(0.5 * G * t * t) + abs(G) * DT * t, p
    assert p[0] > 0.4, "velocity written from Python did not reach the solver"

    # A call that matches no overload must raise TypeError. Built against
    # pybind11 3.1.0 this segfaults instead (pybind/pybind11#6183).
    try:
        body.addObject(1.0)
    except TypeError:
        pass
    else:
        raise AssertionError("addObject(1.0) did not raise TypeError")
    print("OK")


if __name__ == "__main__":
    main()
