# hypy

## Overview
`hypy` is a Python package designed for locating seismic events based on seismic arrivals. It provides the `EventLocator` class, which facilitates the loading of seismic event data, calculation of travel times, and the determination of event locations using various seismic phases.

## Installation
To install `hypy`, you can use pip. Clone the repository and run the following command:

```
pip install .
```

Alternatively, you can install it directly from the source:

```
git clone <repository-url>
cd hypy
pip install .
```

## Usage
To use the `EventLocator` class, you need to import it from the package:

```python
from hypy import EventLocator
```

### Example
Here is a basic example of how to use the `EventLocator` class:

```python
from obspy.taup import TauPyModel
from hypy import EventLocator

# Load a seismic model
model = TauPyModel("ak135")

# Initialize the EventLocator
locator = EventLocator(model=model)

# Load your seismic data (Inventory and Events)
# locator.load_data(inv, events)

# Perform the location
# locations, origin_times, max_dx = locator.locate()
```

## Testing
To run the tests for the `hypy` package, navigate to the project directory and execute:

```
pytest tests/test_solve.py
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.