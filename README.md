# PhotSat Thermal Model

This repository contains the implementation of an empirical thermal model developed for the PhotSat payload detector. The model is based on data obtained from thermal vacuum chamber tests performed at IEEC in March 2025 using the EHD-240903094236 sensor.

The model simulates the temperature evolution of the sensor during typical acquisition cycles in orbit, under different external thermal environments and operational modes.

## Contents

- `thermamodel.py`: Main module implementing the thermal model.
- `thermal_model_workflow.ipynb`: Jupyter notebook with usage examples, model calibration, and validation plots.
- `Figures/`: Directory to store plots of simulated thermal evolution and model residuals (optional).

## Installation

To set up the required environment, make sure you have Python 3.8 or later installed. You can install the model dependencies using pip:

```bash
pip install -r requirements.txt
```

If you're working in a controlled environment or inside a virtualenv, you can create one as follows:

```
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
pip install -r requirements.txt
```

To use the Jupyter notebook included in this repository:

```
pip install jupyter
jupyter notebook thermal_model_workflow.ipynb
```

## Model Overview

The model approximates the temperature evolution of the sensor using an exponential law:

\[
T(t) = T_{\rm eq} + (T_0 - T_{\rm eq}) \cdot \exp\left(-\frac{t}{\tau}\right)
\]

Where:

- \( T_0 \) is the initial temperature,
- \( T_{\rm eq} \) is the equilibrium temperature extracted from dark test frames,
- \( \tau \) is the characteristic thermal response time.

Both \( T_{\rm eq} \) and \( \tau \) are modeled as functions of external temperature and exposure time, using interpolation from vacuum test data.

## How to Use

You can use the model via script or interactively with the notebook:

```python
from thermamodel import ThermalModel

model = ThermalModel()
model.load("path/to/calibrated_model/")
times, temps = model.simulate_temperature_from_custom_timeline(
    time_model=my_time_frame,
    external_temp=0.0,
    exposure_time_min=3.125 / 60
)
```

For full examples, open the notebook:

```bash
jupyter notebook thermal_model_workflow.ipynb
```

## Applications

* Predicting sensor temperature evolution during acquisition bursts.
* Evaluating operational risks under varying thermal conditions.
* Supporting payload calibration strategies and mission planning.
* Embedding in photometric pipelines to estimate background or bias drifts.

## License

MIT License

(c) 2025 E.G. Ferrer (IEEC/ICCUB)
