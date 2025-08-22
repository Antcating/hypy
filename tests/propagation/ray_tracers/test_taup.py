import unittest
from hypy.propagation.ray_tracers.taup import TauPTracer
import os
import obspy


class TestTravelTimeTables(unittest.TestCase):
    def setUp(self):
        self.model = "iasp91"
        self.phases = ["P", "S"]
        self.max_depth = 100.0
        self.max_distance = 1000.0
        self.depth_step = 10.0
        self.distance_step = 50.0

        obspy_path = os.path.dirname(obspy.__file__)
        model_path = os.path.join(obspy_path, "taup", "data", "iasp91.tvel")
        self.tracer = TauPTracer(model_path=model_path)

    def test_get_arrival(self):
        depth = 50.0
        distance = 500.0
        time_p, angle_p = self.tracer.get_arrival(depth, distance, "P")
        time_s, angle_s = self.tracer.get_arrival(depth, distance, "S")
        self.assertIsInstance(time_p, float)
        self.assertAlmostEqual(time_p, 65.54952023370768, places=3)
        self.assertIsInstance(angle_p, float)
        self.assertAlmostEqual(angle_p, 89.6292410507832, places=1)
        self.assertIsInstance(time_s, float)
        self.assertAlmostEqual(time_s, 117.19924749280987, places=3)
        self.assertIsInstance(angle_s, float)
        self.assertAlmostEqual(angle_s, 88.83770931419198, places=1)

    def test_invalid_phase(self):
        depth = 50.0
        distance = 500.0
        with self.assertRaises(ValueError):
            self.tracer.get_arrival(depth, distance, "X")


if __name__ == "__main__":
    unittest.main()
