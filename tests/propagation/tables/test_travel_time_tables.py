import unittest
from hypy.propagation.tables.travel_time_tables import TravelTimeTables

class TestTravelTimeTables(unittest.TestCase):
    def setUp(self):
        self.model = "iasp91"
        self.phases = ["P", "S"]
        self.max_depth = 100.0
        self.max_distance = 1000.0
        self.depth_step = 10.0
        self.distance_step = 50.0
        self.tables = TravelTimeTables(
            model=self.model,
            phases=self.phases,
            max_depth=self.max_depth,
            max_distance=self.max_distance,
            depth_step=self.depth_step,
            distance_step=self.distance_step,
            overwrite=True,
        )

    def test_table_creation(self):
        self.assertIsNotNone(self.tables.depth_grid)
        self.assertIsNotNone(self.tables.distance_grid)
        self.assertIn("P", self.tables.travel_time_tables)
        self.assertIn("S", self.tables.travel_time_tables)

    def test_compute_travel_time(self):
        depth = 50.0
        distance = 500.0
        travel_time_p = self.tables.compute_travel_time("P", depth, distance)
        travel_time_s = self.tables.compute_travel_time("S", depth, distance)
        self.assertIsInstance(travel_time_p, float)
        self.assertAlmostEqual(travel_time_p, 65.54952023370768, places=3)
        self.assertIsInstance(travel_time_s, float)
        self.assertAlmostEqual(travel_time_s, 117.19924749280987, places=3)

    def test_compute_takeoff_angle(self):
        depth = 50.0
        distance = 500.0
        takeoff_angle_p = self.tables.compute_takeoff_angle("P", depth, distance)
        takeoff_angle_s = self.tables.compute_takeoff_angle("S", depth, distance)
        self.assertIsInstance(takeoff_angle_p, float)
        self.assertIsInstance(takeoff_angle_s, float)

        self.assertAlmostEqual(takeoff_angle_p, 89.35376553407343, places=1)
        self.assertAlmostEqual(takeoff_angle_s, 88.82780282349177, places=1)
        
    def test_get_arrival(self):
        depth = 50.0
        distance = 500.0
        time_p, angle_p = self.tables.get_arrival(depth, distance, "P")
        time_s, angle_s = self.tables.get_arrival(depth, distance, "S")
        self.assertIsInstance(time_p, float)
        self.assertAlmostEqual(time_p, 65.54952023370768, places=3)
        self.assertIsInstance(angle_p, float)
        self.assertAlmostEqual(angle_p, 89.35376553407343, places=1)
        self.assertIsInstance(time_s, float)
        self.assertAlmostEqual(time_s, 117.19924749280987, places=3)
        self.assertIsInstance(angle_s, float)
        self.assertAlmostEqual(angle_s, 88.82780282349177, places=1)


    def test_invalid_phase(self):
        depth = 50.0
        distance = 500.0
        with self.assertRaises(ValueError):
            self.tables.get_arrival(depth, distance, "X")

if __name__ == "__main__":
    unittest.main()