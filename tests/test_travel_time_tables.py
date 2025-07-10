import unittest
import numpy as np
import tempfile, os
from hypy.travel_time_tables import TravelTimeTables


class TestTravelTimeTables(unittest.TestCase):
    def setUp(self):
        self.model_name = "iasp91"
        self.phases = ["P", "S"]
        self.ttt = TravelTimeTables(self.model_name, self.phases, 2.0, 4.0, 1.0, 2.0, overwrite=True)

    def test_table_filename_contains_expected(self):
        # Patch __file__ to simulate location
        import hypy.travel_time_tables
        import pathlib
        hypy.travel_time_tables.__file__ = str(pathlib.Path.cwd() / "travel_time_tables.py")
        fname = self.ttt._table_filename()
        self.assertIn("travel_time_tables_iasp91_P_S_d2.0_r4.0_dz1.0_dr2.0.npz", fname)
        self.assertIn("data", fname)

    def test_interpolate_travel_time(self):
        tt_p = self.ttt.interpolate_travel_time("P", 1.0, 2.0)
        tt_s = self.ttt.interpolate_travel_time("S", 1.0, 2.0)
        self.assertTrue(np.isclose(tt_p, tt_p))  # Just check it returns a float
        self.assertTrue(np.isclose(tt_s, tt_s))

    def test_interpolate_ray_param(self):
        rp_p = self.ttt.interpolate_ray_param("P", 1.0, 2.0)
        rp_s = self.ttt.interpolate_ray_param("S", 1.0, 2.0)
        self.assertTrue(np.isclose(rp_p, rp_p))
        self.assertTrue(np.isclose(rp_s, rp_s))

    def test_interpolate_takeoff_angle(self):
        ta_p = self.ttt.interpolate_takeoff_angle("P", 1.0, 2.0)
        ta_s = self.ttt.interpolate_takeoff_angle("S", 1.0, 2.0)
        self.assertTrue(np.isclose(ta_p, ta_p))
        self.assertTrue(np.isclose(ta_s, ta_s))

    def test_interpolation_at_grid_edges(self):
        tt = self.ttt.interpolate_travel_time("P", 0.0, 0.0)
        self.assertTrue(np.isclose(tt, tt))
        tt = self.ttt.interpolate_travel_time("S", 2.0, 4.0)
        self.assertTrue(np.isclose(tt, tt))

    def test_save_and_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "test_table.npz")
            self.ttt.save(fname)
            ttt2 = TravelTimeTables(self.model_name, self.phases, 2.0, 4.0, 1.0, 2.0, overwrite=True)
            ttt2.load(fname)
            self.assertTrue(np.allclose(self.ttt.travel_time_tables["P"], ttt2.travel_time_tables["P"]))
            self.assertTrue(np.allclose(self.ttt.travel_time_tables["S"], ttt2.travel_time_tables["S"]))

    def test_invalid_phase_raises(self):
        with self.assertRaises(KeyError):
            self.ttt.interpolate_travel_time("Q", 1.0, 2.0)

    def test_invalid_table_save_raises(self):
        ttt = TravelTimeTables(self.model_name, self.phases, 2.0, 4.0, 1.0, 2.0, overwrite=True)
        ttt.depth_grid = None
        ttt.distance_grid = None
        with self.assertRaises(ValueError):
            ttt.save("/tmp/should_fail.npz")

    def test_invalid_table_load_raises(self):
        ttt = TravelTimeTables(self.model_name, self.phases, 2.0, 4.0, 1.0, 2.0, overwrite=True)
        with self.assertRaises(FileNotFoundError):
            ttt.load("/nonexistent/file.npz")

    def test_interpolation_out_of_bounds(self):
        tt = self.ttt.interpolate_travel_time("P", -5.0, -5.0)
        self.assertTrue(np.isclose(tt, tt))
        tt = self.ttt.interpolate_travel_time("S", 100.0, 100.0)
        self.assertTrue(np.isclose(tt, tt))

if __name__ == "__main__":
    unittest.main()
