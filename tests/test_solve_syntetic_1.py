# import unittest
# import numpy as np
# from obspy.taup import TauPyModel
# from relocpy.solve import EventLocator

# from obspy.geodetics import kilometer2degrees

# class TestEventLocator(unittest.TestCase):
#     def setUp(self):
#         self.stations = [
#             (0, 10), (0, 49), (0, 149), (0, 189),
#             (2, 10), (2, 49), (2, 149), (2, 189),
#         ]
#         self.model = TauPyModel(model="iasp91")
#         self.events = [(90, 120), (80, 140)]
#         self.arrivals_times = np.zeros((len(self.events), len(self.stations)))
#         for event_idx, event in enumerate(self.events):
#             for station_idx, station in enumerate(self.stations):
#                 distance_km = np.sqrt((event[1] - station[1]) ** 2)
#                 distance_deg = kilometer2degrees(distance_km)
#                 arrivals = self.model.get_ray_paths(source_depth_in_km=event[0], distance_in_degree=distance_deg, phase_list=["P", "p"], receiver_depth_in_km=station[0])
#                 if len(arrivals) == 0:
#                     raise ValueError("No arrivals found")
#                 arrival = arrivals[0]
#                 self.arrivals_times[event_idx, station_idx] = arrival.time + (3 * (event_idx + 1))
#         self.locator = EventLocator(self.events, self.stations, self.model)

#     def test_dt_dr(self):
#         distance_km = 100
#         source_depth = 10
#         receiver_depth = 0
#         dt_dr = self.locator._dt_dr_num(distance_km, source_depth, receiver_depth)
#         self.assertIsInstance(dt_dr, float)

#     def test_dt_dz(self):
#         distance_km = 100
#         source_depth = 10
#         receiver_depth = 0
#         dt_dz = self.locator._dt_dz_num(distance_km, source_depth, receiver_depth)
#         self.assertIsInstance(dt_dz, float)

#     # def test_locate(self):
#     #     locations, origin_times = self.locator.locate(self.arrivals_times, max_iterations=50)
#     #     self.assertTrue(np.all(np.abs(locations - np.array(self.events)) < 1))
#     #     self.assertEqual(locations.shape, (len(self.events), 2))

#     #     print(locations)

#     def test_init_location(self):
#         locations = self.locator.init_locations(self.arrivals_times)
#         self.assertEqual(locations.shape, (len(self.events), 2))
#         self.assertTrue(np.all(np.abs(locations[:, 0] - 10) == 0))
#         print(locations)

# if __name__ == '__main__':
#     unittest.main()