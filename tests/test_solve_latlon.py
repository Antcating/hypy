# import unittest
# import numpy as np
# from obspy.taup import TauPyModel
# from obspy.core.event import Pick

# from relocpy.solve import EventLocator
# from relocpy.arrival import Arrival

# from obspy.geodetics import kilometer2degrees

# class TestEventLocator(unittest.TestCase):
#     def setUp(self):
#         # lat, lon, depth (km)
#         self.stations = [
#             (10.05, 10.05, 0),
#             (10.05, 10.35, 0),
#             (10.05, 10.65, 0),
#             (10.05, 10.95, 0),

#             (10.35, 10.05, 0),
#             (10.35, 10.35, 0),
#             (10.35, 10.65, 0),
#             (10.35, 10.95, 0),
            
#             (10.65, 10.05, 0),
#             (10.65, 10.35, 0),
#             (10.65, 10.65, 0),
#             (10.65, 10.95, 0),

#             (10.95, 10.05, 0),
#             (10.95, 10.35, 0),
#             (10.95, 10.65, 0),
#             (10.95, 10.95, 0),
            
#         ]
#         self.model = TauPyModel(model="iasp91")
#         self.events = [(10.1, 10.3, 8), (10.8, 10.6, 12)]
#         self.arrivals_times = np.zeros((len(self.events), len(self.stations)))
#         for event_idx, event in enumerate(self.events):
#             for station_idx, station in enumerate(self.stations):
#                 arrivals = self.model.get_travel_times_geo(
#                     source_latitude_in_deg=event[0], source_longitude_in_deg=event[1], source_depth_in_km=event[2],
#                     receiver_latitude_in_deg=station[0], receiver_longitude_in_deg=station[1],
#                     phase_list=["P", "p"])
#                 if len(arrivals) == 0:
#                     raise ValueError("No arrivals found")
#                 arrival = arrivals[0]
#                 arr = Pick(
#                     time=arrival.time + (3 * (event_idx + 1)),

#                     phase="P") 
#                 t = arr.time
                
#                 self.arrivals_times[event_idx, station_idx] = arrival.time + (3 * (event_idx + 1))
#         self.locator = EventLocator(self.events, self.stations, self.model, dim=3)


#     def test_locate(self):
#         locations, origin_times = self.locator.locate(self.arrivals_times, max_iterations=50)
#         self.assertTrue(np.all(np.abs(locations - np.array(self.events)) < 1))
#         self.assertTrue(np.all(np.abs(origin_times - np.array([3, 6])) < 1))
#         self.assertEqual(locations.shape, (len(self.events), 3))

#         print(locations)

#     # def test_init_location(self):
#     #     locations = self.locator.init_locations(self.arrivals_times)
#     #     self.assertEqual(locations.shape, (len(self.events), 3))
#     #     self.assertTrue(np.all(np.abs(locations[:, 2] - 10) == 0))
#     #     print(locations)

# if __name__ == '__main__':
#     unittest.main()