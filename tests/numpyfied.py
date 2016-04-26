import numpy as np
from unittest import TestCase
from bnol import data

class NumpyfiedTestCase(TestCase):
    def _super(self, method, *args):
        getattr(super(NumpyfiedTestCase, self), method)(*args)

    def assertAllTrue(self, expr, msg=None):
        self._super('assertTrue', np.alltrue(expr), msg)

    def assertIsClose(self, first, second, msg=None):
        self._super('assertTrue', np.isclose(first, second, data.epsilon()), msg)
