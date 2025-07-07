import unittest

import numpy as np

from obspy.core.inventory import Inventory, Network, Station, Channel, Site
from obspy.core.event import Origin, Event, Magnitude, Pick, Arrival, WaveformStreamID
from obspy.taup import TauPyModel
from obspy.taup.taup_create import build_taup_model
from obspy import read_inventory

from obspy.clients.fdsn import Client
from obspy import UTCDateTime

from relocpy.solve import EventLocator

class TestEventLocator(unittest.TestCase):
    def setUp(self):
        self.model = TauPyModel(model="giimod")

    def test_locate(self):
        # Test stations
        reference_p_phases = {
            "MEFR": UTCDateTime("2023-02-07T21:14:33.070000Z"),
            "ARIL": UTCDateTime("2023-02-07T21:14:33.320000Z"),
            "PZL": UTCDateTime("2023-02-07T21:14:34.310000Z"),
            "HMRA": UTCDateTime("2023-02-07T21:14:35.010000Z"),
            "NTGD": UTCDateTime("2023-02-07T21:14:35.220000Z"),
            "SLMZ": UTCDateTime("2023-02-07T21:14:35.300000Z"),
            "YITV":  UTCDateTime("2023-02-07T21:14:36.080000Z"),
            "RGMN":  UTCDateTime("2023-02-07T21:14:36.680000Z"),
            "ROI": UTCDateTime("2023-02-07T21:14:37.490000Z"),
            "NAMA":  UTCDateTime("2023-02-07T21:14:37.650000Z"),
            "HMDT": UTCDateTime("2023-02-07T21:14:38.560000Z"),
            "SLTI": UTCDateTime("2023-02-07T21:14:39.550000Z"),
            "VJRH": UTCDateTime("2023-02-07T21:14:39.720000Z"),
            "JER": UTCDateTime("2023-02-07T21:14:41.580000Z")
        }


        # self.client = Client(base_url="https://seis.gsi.gov.il")
        self.inv = read_inventory("tests/giimod.xml")

        # Generate synthetic picks
        self.events = []
        event = Event()

        for station, time in reference_p_phases.items():
            pick = Pick(
                time=time,
                phase_hint="S",
                waveform_id=WaveformStreamID(
                    network_code="IS",
                    station_code=station,
                    channel_code="ENZ",
                )
            )
            event.picks.append(pick)
        self.events.append(event)

        self.locator = EventLocator(self.model, dim=3, phases=["S"])
        self.locator.load_data(self.inv, self.events)
        self.locator.init_locations()
        locations, origin_times, _ = self.locator.locate(max_iterations=50)

        self.assertEqual(locations.shape, (len(self.events), 3))
        print(locations)
        print(origin_times)

    def test_locate_multiple_events(self):
        import pickle
        import time
        # Test stations
        self.inv = read_inventory("tests/data/inv.xml")
        with open("tests/data/events.pickle", "rb") as f:
            self.events = pickle.load(f)

        self.locator = EventLocator(self.model, dim=3, phases=["S"])
        self.locator.load_data(self.inv, self.events)
        self.locator.init_locations()
        start = time.time()
        locations, origin_times, _ = self.locator.locate(max_iterations=50)
        end = time.time()
        print(f"Time taken: {end - start} seconds")
        print(locations)
        print(origin_times)

if __name__ == '__main__':
    unittest.main()