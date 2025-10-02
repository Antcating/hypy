# hypy

## Overview
`hypy` is a Python package designed for locating seismic events based on seismic arrivals. It provides the `EventLocator` class, which facilitates the loading of seismic event data, calculation of travel times, and the determination of event locations using various seismic phases.

## Installation
To install `hypy`, clone the repository and run the following commands:

```
git clone <repository-url>
cd hypy
pip install .
```

## Usage
To use the features of the package, you can use the EventLocator class:

```python
from hypy import EventLocator
```

### Example

```
TBD
```

## Testing

```
TBD
```

## Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue for any enhancements or bug fixes.

## Known issues

Equidistant geometries have ambiguity trade-off between depth and origin time therefore the inversion may not converge to correct source parameters (depth and origin time). This can be mitigated by providing a good initial guess of the source parameters.