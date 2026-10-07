"""Contract tests shared by all copula model implementations."""

from __future__ import annotations

import unittest

import numpy as np

from src.copulas import CopulaModel, GaussianCopulaModel, StudentTCopulaModel


class CopulaModelContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        rng = np.random.default_rng(20261007)
        drivers = rng.standard_t(df=6, size=(500, 3))
        cls.data = np.column_stack(
            [
                drivers[:, 0],
                0.65 * drivers[:, 0] + 0.75 * drivers[:, 1],
                0.30 * drivers[:, 0] + 0.95 * drivers[:, 2],
            ]
        )

    def _exercise_contract(self, model: CopulaModel[object]) -> None:
        self.assertIsInstance(model, CopulaModel)
        fitted = model.fit(self.data)
        uniforms = model.simulate_uniforms(
            fitted,
            size=400,
            rng=np.random.default_rng(1234),
        )

        self.assertEqual(uniforms.shape, (400, self.data.shape[1]))
        self.assertTrue(((uniforms > 0) & (uniforms < 1)).all())
        correlation = model.correlation(fitted)
        self.assertEqual(correlation.shape, (self.data.shape[1],) * 2)
        np.testing.assert_allclose(np.diag(correlation), 1.0)
        tail = model.lower_tail_dependence(fitted)
        self.assertGreaterEqual(tail, 0.0)
        self.assertLessEqual(tail, 1.0)
        self.assertTrue(model.parameters(fitted))

    def test_student_t_model_satisfies_contract(self) -> None:
        self._exercise_contract(StudentTCopulaModel((2.05, 50.0)))

    def test_gaussian_model_satisfies_contract(self) -> None:
        model = GaussianCopulaModel()
        self._exercise_contract(model)
        fitted = model.fit(self.data)
        self.assertEqual(model.lower_tail_dependence(fitted), 0.0)


if __name__ == "__main__":
    unittest.main()

