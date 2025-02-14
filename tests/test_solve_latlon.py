import unittest
import numpy as np
from obspy.taup import TauPyModel
from relocpy.solve import EventLocator

from obspy.geodetics import kilometer2degrees

class TestEventLocator(unittest.TestCase):
    def setUp(self):
        # lat, lon, depth (km)
        self.stations = [
            (10.05, 10.05, 0),
            (10.05, 10.35, 0),
            (10.05, 10.65, 0),
            (10.05, 10.95, 0),

            (10.35, 10.05, 0),
            (10.35, 10.35, 0),
            (10.35, 10.65, 0),
            (10.35, 10.95, 0),
            
            (10.65, 10.05, 0),
            (10.65, 10.35, 0),
            (10.65, 10.65, 0),
            (10.65, 10.95, 0),

            (10.95, 10.05, 0),
            (10.95, 10.35, 0),
            (10.95, 10.65, 0),
            (10.95, 10.95, 0),
            
        ]
        self.model = TauPyModel(model="iasp91")
        self.events = [(10.3, 10.3, 8)]
        self.event_initials = [(10.2, 10.4, 10)]
        self.arrivals_times = np.zeros((len(self.events), len(self.stations)))
        for event_idx, event in enumerate(self.events):
            for station_idx, station in enumerate(self.stations):
                distance_km = np.sqrt((event[1] - station[1]) ** 2)
                distance_deg = kilometer2degrees(distance_km)
                arrivals = self.model.get_travel_times_geo(
                    source_latitude_in_deg=event[0], source_longitude_in_deg=event[1], source_depth_in_km=event[2],
                    receiver_latitude_in_deg=station[0], receiver_longitude_in_deg=station[1],
                    phase_list=["P", "p"])
                if len(arrivals) == 0:
                    raise ValueError("No arrivals found")
                arrival = arrivals[0]
                self.arrivals_times[event_idx, station_idx] = arrival.time
        self.locator = EventLocator(self.events, self.stations, self.model, dim=3)


    def test_locate(self):
        locations = self.locator.locate(self.event_initials, self.arrivals_times)
        self.assertTrue(np.all(np.abs(locations - np.array(self.events)) < 2))
        self.assertEqual(locations.shape, (len(self.event_initials), 3))

        print(locations)