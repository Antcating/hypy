import unittest
import numpy as np
from obspy.taup import TauPyModel
from relocpy.solve import EventLocator

from obspy.geodetics import kilometer2degrees

class TestEventLocator(unittest.TestCase):
    def setUp(self):
        self.stations = [
            (0, 10), (0, 49), (0, 149), (0, 189),
            (60, 10), (60, 49), (60, 149), (60, 189),
        ]
        self.model = TauPyModel(model="iasp91")
        self.events = [(90, 100), (80, 120)]
        self.event_initials = [(90, 110), (80, 130)]
        self.arrivals_times = np.zeros((len(self.events), len(self.stations)))
        for event_idx, event in enumerate(self.events):
            for station_idx, station in enumerate(self.stations):
                distance_km = np.sqrt((event[1] - station[1]) ** 2)
                distance_deg = kilometer2degrees(distance_km)
                arrivals = self.model.get_ray_paths(source_depth_in_km=event[0], distance_in_degree=distance_deg, phase_list=["P", "p"], receiver_depth_in_km=station[0])
                arrival = arrivals[0]
                self.arrivals_times[event_idx, station_idx] = arrival.time
        self.locator = EventLocator(self.events, self.stations, self.model)

    def test_dt_dr(self):
        distance_km = 100
        source_depth = 10
        receiver_depth = 0
        dt_dr = self.locator._dt_dr(distance_km, source_depth, receiver_depth)
        self.assertIsInstance(dt_dr, float)

    def test_dt_dz(self):
        distance_km = 100
        source_depth = 10
        receiver_depth = 0
        dt_dz = self.locator._dt_dz(distance_km, source_depth, receiver_depth)
        self.assertIsInstance(dt_dz, float)

    def test_locate(self):
        locations = self.locator.locate(self.event_initials, self.arrivals_times)
        self.assertTrue(np.all(np.abs(locations - np.array(self.events)) < 2))
        self.assertEqual(locations.shape, (len(self.event_initials), 2))

        print(locations)

if __name__ == '__main__':
    unittest.main()