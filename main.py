"""
main.py
-------
Example usage of EventLocator and TravelTimeTables for earthquake location.
"""
from obspy.taup import TauPyModel
from obspy.core import UTCDateTime
from obspy.core.event import Event, Pick, WaveformStreamID
from obspy.core.inventory import Inventory, Network, Station, Site
from relocpy.event_locator import EventLocator

if __name__ == "__main__":
    # Load a TauPy model (e.g., IASP91)
    model = TauPyModel(model="iasp91")

    # Define synthetic stations and their locations
    station_coords = {
        "STA01": (10.0, 10.0, 0.0),
        "STA02": (10.05, 10.0, 0.0),
        "STA03": (10.25, 10.0, 0.0),
        "STA04": (10.0, 10.05, 0.0),
        "STA05": (10.0, 10.25, 0.0),
        "STA07": (10.07, 10.07, 0.0),
        "STA08": (10.15, 10.15, 0.0),
        "STA09": (10.2, 10.2, 0.0),
        "STA10": (10.3, 10.3, 0.0),
    }
    # Build inventory
    net = Network(code="NET", start_date=UTCDateTime("2020-01-01"))
    for sta, (lat, lon, elev) in station_coords.items():
        net.stations.append(
            Station(
                code=sta,
                latitude=lat,
                longitude=lon,
                elevation=elev,
                start_date=UTCDateTime("2020-01-01"),
                site=Site(name=sta),
            )
        )
    inv = Inventory(networks=[net], source="synthetic")

    # Define synthetic event
    event_latlon_depth = (10.05, 10.05, 10.0)  # lat, lon, depth in km
    origin_time = UTCDateTime("2024-01-01T01:01:01.100Z")
    # Generate synthetic picks
    event = Event()
    for sta, (lat, lon, elev) in station_coords.items():
        arrivals = model.get_travel_times_geo(
            source_latitude_in_deg=event_latlon_depth[0],
            source_longitude_in_deg=event_latlon_depth[1],
            source_depth_in_km=event_latlon_depth[2],
            receiver_latitude_in_deg=lat,
            receiver_longitude_in_deg=lon,
            phase_list=["tts"],
        )
        if not arrivals:
            print(f"No arrivals for station {sta}, skipping.")
            continue
        arr = arrivals[0]
        pick_time = origin_time + arr.time
        pick = Pick(
            time=pick_time,
            phase_hint="S",
            waveform_id=WaveformStreamID(
                network_code="NET",
                station_code=sta,
                channel_code="ENZ",
            ),
        )
        event.picks.append(pick)
    events = [event]

    locator = EventLocator(model, dim=3, phases=["S"])
    locator.load_data(inv, events)
    locator.init_locations()
    locations, origin_times, _ = locator.locate(max_iterations=500)

    print("Final event locations:")
    for i, loc in enumerate(locations):
        lat, lon, depth = loc
        print(f"Event {i}: Latitude: {lat:.4f}, Longitude: {lon:.4f}, Depth: {depth:.2f} km")
        print(f"Origin time: {origin_times[i]}")
