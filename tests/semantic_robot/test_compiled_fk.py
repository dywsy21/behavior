"""Fast pose-only FK must agree with the unchanged screw/Jacobian path."""
import json
from pathlib import Path
import unittest

import numpy as np

from semantic_robot.v2.kinematics import (
    RobotModel, _compile_twist, _compiled_exp, se3_exp,
)
from test_v2 import fixture


class CompiledFKTests(unittest.TestCase):
    def test_exponential_random_nonunit_and_zero_twists(self):
        rng = np.random.default_rng(187)
        screws = [np.zeros(6), np.r_[rng.normal(size=3), np.zeros(3)]]
        for norm in (1e-12, 1e-10, 1e-8, .3, 1., 4.):
            for _ in range(8):
                axis = rng.normal(size=3); axis *= norm / np.linalg.norm(axis)
                screws.append(np.r_[rng.normal(size=3), axis])
        for screw in screws:
            constant = _compile_twist(screw)
            for value in (0., 1e-12, -.002, .3, -2.7, 6.1):
                with self.subTest(norm=np.linalg.norm(screw[3:]), value=value):
                    np.testing.assert_allclose(_compiled_exp(constant, value),
                                               se3_exp(screw, value), atol=2e-12, rtol=1e-13)

    def test_real_calibration_all_links_and_many_configurations(self):
        model = RobotModel(json.loads((Path(__file__).parent / "fixtures/r1pro_folded_fk.json").read_text()))
        rng = np.random.default_rng(81)
        poses = [model.reference.copy(), model.lower.copy(), model.upper.copy()]
        poses += [rng.uniform(model.lower, model.upper) for _ in range(24)]
        for q in poses:
            for name in model.links:
                legacy, _ = model.evaluate(q, name)
                np.testing.assert_allclose(model.forward(q, name), legacy, atol=3e-12, rtol=0)

    def test_calibration_replacement_and_inplace_edits_invalidate_cache(self):
        model, state = fixture()
        name = "camera_head"
        model.forward(state.q, name)
        home, screws = model.links[name]
        home = home.copy(); home[0, 3] += .2
        model.links[name] = (home, screws.copy())
        np.testing.assert_allclose(model.forward(state.q, name), model.evaluate(state.q, name)[0], atol=1e-14)
        home[1, 3] -= .1
        np.testing.assert_allclose(model.forward(state.q, name), model.evaluate(state.q, name)[0], atol=1e-14)
        q = state.q.copy(); q[0] = .07
        model.forward(q, name)
        model.links[name][1][0, 0] = 1.
        np.testing.assert_allclose(model.forward(q, name), model.evaluate(q, name)[0], atol=1e-14)
        model.reference[0] += .03
        np.testing.assert_allclose(model.forward(q, name), model.evaluate(q, name)[0], atol=1e-14)

    def test_cache_is_bounded_and_returns_independent_arrays(self):
        model, state = fixture()
        expected = model.evaluate(state.q, "right")[0]
        model.forward(state.q, "right")[:] = 99.
        np.testing.assert_array_equal(model.forward(state.q, "right"), expected)
        for i in range(550):
            q = state.q.copy(); q[11] = .0001 * i
            model.forward(q, "right")
        self.assertLessEqual(len(model._fk_cache), 512)
        self.assertEqual(len(model._compiled_links), 1)

    def test_invalid_q_still_rejected(self):
        model, state = fixture()
        for q in (np.zeros(17), np.full(18, np.nan), np.full(18, np.inf)):
            with self.assertRaises(ValueError):
                model.forward(q, "right")


if __name__ == "__main__":
    unittest.main()
