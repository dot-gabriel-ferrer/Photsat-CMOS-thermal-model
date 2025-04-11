import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from astropy.io import fits
from scipy.interpolate import LinearNDInterpolator
from scipy.optimize import curve_fit
import os
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl


class ThermalModel:
    """
    ThermalModel class for modeling the thermal behavior of an image sensor in orbit.

    This class uses calibration data (sensor temperature, external temp, exposure time, etc.)
    to fit thermal rise curves and interpolate them to simulate orbital temperature evolution
    during image capture bursts.
    """

    def __init__(self):
        self.interpolators = None  # Holds interpolated functions for T_ext, T_max, tau
        self.fit_df = None         # DataFrame with fitted parameters per (external_temp, exposure_time)

    @staticmethod
    def extract_temperature_data_from_fits(directory, ext_filter="fits"):
        """
        Loads all FITS headers from a directory and extracts sensor temperature, exposure, etc.

        Returns
        -------
        df : pd.DataFrame
            Contains timestamp, sensor_temp, external_temp, exposure_time, gain, frame_number
        """
        directory = Path(directory).resolve()
        all_data = []

        for root, _, files in os.walk(directory):
            for fname in files:
                if fname.lower().endswith(ext_filter):
                    fpath = Path(root) / fname
                    try:
                        hdr = fits.getheader(fpath)

                        ts = None
                        for key in ["IMESTAMP", "TIMESTAMP", "DATE-OBS", "TIME"]:
                            if key in hdr:
                                ts = hdr[key]
                                break

                        t_sensor = hdr.get('TEMP')
                        t_external = hdr.get('EXTTEMP')
                        exp = hdr.get('EXPTIME')
                        gain = hdr.get('EXPGAIN')
                        fnum = hdr.get('FRAMENUM')

                        if None in [t_sensor, exp, ts]:
                            continue

                        all_data.append({
                            "path": str(fpath),
                            "timestamp": float(ts),
                            "sensor_temp": float(t_sensor),
                            "external_temp": float(t_external) if t_external is not None else np.nan,
                            "exposure_time": float(exp),
                            "gain": float(gain) if gain is not None else np.nan,
                            "frame_number": int(fnum) if fnum is not None else np.nan,
                        })

                    except Exception as e:
                        print(f"[ERROR] {fpath.name}: {e}")

        df = pd.DataFrame(all_data)
        df.sort_values(by="timestamp", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def plot_temperature_evolution(df):
        """
        Plot sensor temperature over time and show statistics.
        """
        plt.figure(figsize=(12, 6))
        plt.plot(df["timestamp"] - df["timestamp"].min(), df["sensor_temp"], label="Sensor Temp (°C)", marker='o')
        if "external_temp" in df.columns and df["external_temp"].notna().any():
            plt.plot(df["timestamp"] - df["timestamp"].min(), df["external_temp"], label="External Temp (°C)", linestyle='--')

        plt.xlabel("Time since start (s)")
        plt.ylabel("Temperature (°C)")
        plt.title("Sensor Temperature Evolution During Capture")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()
        print("\n📈 Temperature Statistics:")
        print(df["sensor_temp"].describe())

    @staticmethod
    def group_by_external_temp_and_exptime(df, tol_temp=0.5, tol_exp=0.05):
        """
        Groups data by (external_temp, exposure_time) with tolerance to bin values.

        Returns
        -------
        dict: {(T_ext, t_exp): pd.DataFrame}
        """
        groups = {}
        for _, row in df.iterrows():
            t_ext = row["external_temp"]
            exp = row["exposure_time"]
            matched_key = None
            for key in groups:
                if abs(key[0] - t_ext) <= tol_temp and abs(key[1] - exp) <= tol_exp:
                    matched_key = key
                    break
            if matched_key is None:
                matched_key = (t_ext, exp)
                groups[matched_key] = []
            groups[matched_key].append(row)
        return {k: pd.DataFrame(v) for k, v in groups.items()}

    @staticmethod
    def exclude_by_temperature(grouped, excluded_temps, tol=0.5):
        """
        Returns keys of groups to exclude based on T_ext.
        """
        return [key for key in grouped if any(abs(key[0] - t) <= tol for t in excluded_temps)]

    def build_interpolated_model(self, grouped, exclude_keys=[]):
        """
        Fits the thermal model to each group and builds interpolators.
        """
        points, t_exts, t_maxs, taus = [], [], [], []

        for key, df in grouped.items():
            if key in exclude_keys or len(df) < 5:
                continue
            try:
                t0 = df['timestamp'].min()
                times = df['timestamp'] - t0
                temps = df['sensor_temp'].values

                def thermal_model(t, T_ext, T_max, tau):
                    return T_ext + (T_max - T_ext) * (1 - np.exp(-t / tau))

                popt, _ = curve_fit(
                    thermal_model, times, temps,
                    p0=[temps.min(), temps.max(), 10.0],
                    bounds=([temps.min()-5, temps.max()-5, 0.1], [temps.max()+5, temps.max()+10, 500]),
                    maxfev=10000
                )

                points.append(key)
                t_exts.append(popt[0])
                t_maxs.append(popt[1])
                taus.append(popt[2])

            except Exception as e:
                print(f"[ERROR] Fit failed for {key}: {e}")

        self.fit_df = pd.DataFrame({
            "external_temp": np.array(points)[:, 0],
            "exposure_time": np.array(points)[:, 1],
            "T_ext": t_exts,
            "T_max": t_maxs,
            "tau": taus
        })

        self.interpolators = {
            "T_ext": LinearNDInterpolator(points, t_exts),
            "T_max": LinearNDInterpolator(points, t_maxs),
            "tau": LinearNDInterpolator(points, taus)
        }

        return self.interpolators, self.fit_df

    def predict_parameters(self, external_temp, exposure_time):
        """
        Predicts T_ext, T_max, tau for any (external_temp, exposure_time).
        """
        point = np.array([[external_temp, exposure_time]])
        return {
            "T_ext": self.interpolators["T_ext"](point)[0],
            "T_max": self.interpolators["T_max"](point)[0],
            "tau": self.interpolators["tau"](point)[0],
        }

    def simulate(self, external_temp, exposure_time, n_frames=16, frame_pause=0.0, burst_pause=10.0,
                 n_bursts=2, only_when_exposing=True, return_kelvin=False):
        """
        Simulates orbital temperature profile from a predicted parameter set.
        """
        fit_params = self.predict_parameters(external_temp, exposure_time)
        return self.simulate_orbital_temperature_from_fit(
            fit_params, exposure_time, n_frames,
            frame_pause, burst_pause, n_bursts,
            only_when_exposing=only_when_exposing,
            return_kelvin=return_kelvin
        )

    def simulate_orbital_temperature_from_fit(
    self,
    fit_params,
    exposure_time,
    n_frames,
    frame_pause=0.0,
    burst_pause=10.0,
    n_bursts=3,
    tau_cool=None,
    return_kelvin=False,
    only_when_exposing=True,
    force_external_temp=None
    ):
        """
        Simulates orbital sensor temperature using thermal model fit parameters.

        Parameters
        ----------
        fit_params : dict
            Must contain keys 'T_ext', 'T_max', 'tau'.
        exposure_time : float
            Exposure time per frame (s).
        n_frames : int
            Number of frames per burst.
        frame_pause : float
            Pause between frames (s).
        burst_pause : float
            Pause between bursts (s).
        n_bursts : int
            Number of bursts to simulate.
        tau_cool : float or None
            Cooling time constant. If None, uses 2 * tau.
        return_kelvin : bool
            If True, output temperatures in Kelvin.
        only_when_exposing : bool
            If True, return only values during exposures.
        force_external_temp : float or None
            If set, overrides fit_params["T_ext"] and uses this value as external temperature.

        Returns
        -------
        times : np.ndarray
            Time stamps in seconds.
        temps : np.ndarray
            Simulated temperatures in °C or K.
        """
        def to_kelvin(c): return np.array(c) + 273.15
        def to_celsius(k): return np.array(k) - 273.15

        # Use forced T_ext if provided
        T_ext_C = force_external_temp if force_external_temp is not None else fit_params["T_ext"]
        T_max_C = fit_params["T_max"]
        tau_heat = fit_params["tau"]
        tau_cool = tau_cool if tau_cool is not None else 2 * tau_heat

        # Convert to Kelvin for internal math
        T_ext_K = to_kelvin(T_ext_C)
        T_max_K = to_kelvin(T_max_C)

        def heat(T_prev_K, dt):
            return T_ext_K + (T_prev_K - T_ext_K) * np.exp(-dt / tau_heat) + (T_max_K - T_ext_K) * (1 - np.exp(-dt / tau_heat))

        def cool(T_prev_K, dt):
            return T_ext_K + (T_prev_K - T_ext_K) * np.exp(-dt / tau_cool)

        times, temps = [], []
        all_times, all_temps = [], []

        T_curr_K = T_ext_K
        t = 0.0

        for b in range(n_bursts):
            for f in range(n_frames):
                T_curr_K = heat(T_curr_K, exposure_time)
                if only_when_exposing:
                    temps.append(T_curr_K)
                    times.append(t)
                all_temps.append(T_curr_K)
                all_times.append(t)
                t += exposure_time + frame_pause

            for _ in range(int(burst_pause)):
                T_curr_K = cool(T_curr_K, 1.0)
                if not only_when_exposing:
                    temps.append(T_curr_K)
                    times.append(t)
                all_temps.append(T_curr_K)
                all_times.append(t)
                t += 1.0

        temps_out = np.array(temps) if only_when_exposing else np.array(all_temps)
        times_out = np.array(times) if only_when_exposing else np.array(all_times)

        return times_out, temps_out if return_kelvin else to_celsius(temps_out)


    def save(self, output_dir):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "interpolators.pkl", "wb") as f:
            pickle.dump(self.interpolators, f)
        self.fit_df.to_csv(output_dir / "fit_parameters.csv", index=False)

    def load(self, model_dir):
        model_dir = Path(model_dir)
        with open(model_dir / "interpolators.pkl", "rb") as f:
            self.interpolators = pickle.load(f)
        self.fit_df = pd.read_csv(model_dir / "fit_parameters.csv")

    def map_temperatures_to_custom_times(self, time_model, external_temp, exposure_time, tau_cool=None, force_external_temp=True):
        """
        Map predicted sensor temperatures to custom time points generated by a Time_frame instance,
        accounting for heating and cooling between frames based on real time intervals.

        Parameters
        ----------
        time_model : Time_frame
            Instance of the Time_frame class.
        external_temp : float
            User-defined external temp (ºC) inside the satellite.
        exposure_time : float
            Exposure time per frame (s).
        tau_cool : float or None
            Cooling time constant (s). If None, defaults to 2 * tau.
        force_external_temp : bool
            If True, uses `external_temp` directly instead of interpolated `T_ext`.

        Returns
        -------
        frame_times_sec : np.ndarray
            Times (s) of each frame.
        temperatures : np.ndarray
            Simulated temperatures at each time.
        """
        frame_times_min = time_model()
        frame_times_sec = np.array(frame_times_min) * 60.0

        fit_params = self.predict_parameters(external_temp, exposure_time)

        T_ext = external_temp if force_external_temp else fit_params["T_ext"]
        T_max = fit_params["T_max"]
        tau = fit_params["tau"]
        tau_cool = tau_cool if tau_cool is not None else 2 * tau

        def heat(T_prev, dt):
            return T_ext + (T_prev - T_ext) * np.exp(-dt / tau) + (T_max - T_ext) * (1 - np.exp(-dt / tau))

        def cool(T_prev, dt):
            return T_ext + (T_prev - T_ext) * np.exp(-dt / tau_cool)

        temps = [T_ext]
        T_curr = T_ext
        last_time = frame_times_sec[0]

        for t in frame_times_sec[1:]:
            dt = t - last_time
            T_curr = heat(T_curr, dt) if dt <= exposure_time * 1.5 else cool(T_curr, dt)
            temps.append(T_curr)
            last_time = t

        return frame_times_sec, np.array(temps)
    
    def simulate_temperature_from_custom_timeline(
        self,
        time_model,
        external_temp,
        exposure_time_min,
        frame_pause_min=0.0,
        tau_cool=None,
        force_external_temp=True,
        threshold_buffer_min=0.5,
        show_plot=False
    ):
        """
        Simulate the sensor temperature evolution using a custom timeline, with explicit control
        over heating and cooling durations per frame (exposure vs pause).

        Parameters
        ----------
        time_model : callable
            Instance of a Time_frame class or similar, which returns frame times in minutes when called.
        external_temp : float
            External temperature (°C) used as a baseline for radiative cooling.
        exposure_time_min : float
            Time in minutes during which the sensor is active and heating (true exposure).
        frame_pause_min : float
            Time in minutes between frames where the sensor is idle and cooling.
        tau_cool : float or None
            Cooling time constant (in seconds). If None, defaults to 2 * tau.
        force_external_temp : bool
            If True, uses the user-supplied external_temp rather than interpolated T_ext.
        threshold_buffer_min : float
            Additional margin (in minutes) added to exposure time for classifying heating vs cooling intervals.
        show_plot : bool
            If True, display a plot of Δt intervals and threshold comparison.

        Returns
        -------
        frame_times_sec : np.ndarray
            Timestamps (s) of each frame.
        temps : np.ndarray
            Sensor temperature (°C) at each frame.
        threshold_sec : float
            Threshold used to classify intervals (in seconds).
        num_cooling : int
            Number of intervals classified as passive cooling.
        num_heating : int
            Number of intervals classified as acquisition-driven heating.
        """

        if not callable(time_model):
            raise TypeError("`time_model` must be callable (e.g., an instance of Time_frame with __call__).")

        frame_times_min = time_model()
        frame_times_sec = np.array(frame_times_min) * 60.0
        dt_sec = np.diff(frame_times_sec)

        # Predict parameters
        fit_params = self.predict_parameters(external_temp, exposure_time_min)
        T_ext = external_temp if force_external_temp else fit_params["T_ext"]
        T_max = fit_params["T_max"]
        tau = fit_params["tau"]
        tau_cool = tau_cool if tau_cool is not None else 2 * tau

        threshold_sec = (exposure_time_min + threshold_buffer_min) * 60

        def heat(T_prev, dt):
            return T_ext + (T_prev - T_ext) * np.exp(-dt / tau) + (T_max - T_ext) * (1 - np.exp(-dt / tau))

        def cool(T_prev, dt):
            return T_ext + (T_prev - T_ext) * np.exp(-dt / tau_cool)

        temps = [T_ext]
        T_curr = T_ext
        last_time = frame_times_sec[0]

        for t in frame_times_sec[1:]:
            dt = t - last_time

            if dt <= threshold_sec:
                # Heating phase
                T_curr = heat(T_curr, exposure_time_min * 60)

                # Cooling phase (only if frame_pause is defined)
                if frame_pause_min > 0:
                    cooling_dt = max(0.0, frame_pause_min * 60)
                    T_curr = cool(T_curr, cooling_dt)

            else:
                # Passive cooling during longer gaps
                T_curr = cool(T_curr, dt)

            temps.append(T_curr)
            last_time = t

        # Classification info
        heating_flags = dt_sec <= threshold_sec
        cooling_flags = dt_sec > threshold_sec
        num_heating = np.sum(heating_flags)
        num_cooling = np.sum(cooling_flags)

        # Optional plot
        if show_plot:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 4))
            plt.plot(dt_sec, label="Δt between frames (s)")
            plt.axhline(y=threshold_sec, color="red", linestyle="--", label=f"Threshold = {threshold_sec:.1f} s")
            plt.xlabel("Frame Index")
            plt.ylabel("Δt (s)")
            plt.title("Classification of Intervals: Heating vs Cooling")
            plt.grid(True)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return frame_times_sec, np.array(temps), threshold_sec, num_cooling, num_heating
    
    def find_external_temp_range_for_equilibrium(self, T_eq_target, exposure_time, mode_params, tol=0.5, plot=True):
        """
        Find and visualize the range of external temperatures needed to reach a desired sensor equilibrium temperature.

        Parameters
        ----------
        T_eq_target : float
            Desired sensor equilibrium temperature in °C.
        exposure_time : float
            Exposure time per frame (in seconds).
        mode_params : dict
            Defines the acquisition mode:
                - 'n_frames' : int
                - 'n_bursts' : int
                - 'frame_pause' : float (s)
                - 'burst_pause' : float (s)
        tol : float
            Acceptable difference (°C) between simulated and target equilibrium temperature.
        plot : bool
            Whether to show a visual summary of the estimation.

        Returns
        -------
        valid_ext_temps : list of float
            External temperatures (°C) that yield equilibrium temperature within tolerance.
        """


        if self.interpolators is None:
            raise RuntimeError("Model interpolators not built. Run build_interpolated_model first.")

        # Estética estilo LaTeX-like
        mpl.rcParams['font.family'] = 'serif'
        #mpl.rcParams['font.serif'] = ['Computer Modern Roman']
        mpl.rcParams['text.usetex'] = False  # Cambiar a True si tienes LaTeX instalado

        sns.set_style("whitegrid")

        candidate_ext_temps = np.linspace(-20, 50, 200)
        simulated_eq_temps = []
        valid_ext_temps = []

        for ext_temp in candidate_ext_temps:
            try:
                times, temps = self.simulate(
                    external_temp=ext_temp,
                    exposure_time=exposure_time,
                    n_frames=mode_params["n_frames"],
                    frame_pause=mode_params.get("frame_pause", 0.0),
                    burst_pause=mode_params.get("burst_pause", 10.0),
                    n_bursts=mode_params.get("n_bursts", 2),
                    only_when_exposing=True
                )
                T_eq_sim = np.mean(temps[-mode_params["n_frames"]:])
                simulated_eq_temps.append(T_eq_sim)

                if abs(T_eq_sim - T_eq_target) <= tol:
                    valid_ext_temps.append(ext_temp)
            except Exception as e:
                simulated_eq_temps.append(np.nan)
                continue

        if plot:
            plt.figure(figsize=(10, 5))
            sns.lineplot(x=candidate_ext_temps, y=simulated_eq_temps, color='black', linewidth=2, label=r"Simulated $T_{\mathrm{eq}}$")
            plt.axhline(T_eq_target, color="green", linestyle="--", label=fr"Target $T_{{\mathrm{{eq}}}}$ = {T_eq_target:.1f}°C")
            plt.fill_between(candidate_ext_temps,
                            T_eq_target - tol,
                            T_eq_target + tol,
                            color="green", alpha=0.15, label=f"±{tol}°C tolerance")
            if valid_ext_temps:
                valid_y = [simulated_eq_temps[candidate_ext_temps.tolist().index(v)] for v in valid_ext_temps]
                sns.scatterplot(x=valid_ext_temps, y=valid_y, color="red", s=60, label=r"Valid $T_{\mathrm{ext}}$")

            plt.xlabel(r"External Temperature ($T_{\mathrm{ext}}$) [°C]")
            plt.ylabel(r"Sensor Equilibrium Temp ($T_{\mathrm{eq}}$) [°C]")
            plt.title("Required External Temperature for Sensor Thermal Equilibrium")
            plt.legend()
            plt.tight_layout()
            plt.show()

        return valid_ext_temps


    def compare_modes_equilibrium_temperature(self, T_eq_target, mode_list, tol=0.5, plot=True):
        """
        Compare multiple operational modes and find the common external temperature range that yields a desired T_eq.

        Parameters
        ----------
        T_eq_target : float
            Desired sensor equilibrium temperature in °C.
        mode_list : list of dict
            Each dict must define a mode with keys:
                - 'name': str
                - 'exposure_time': float (s)
                - 'n_frames': int
                - 'n_bursts': int
                - 'frame_pause': float (s)
                - 'burst_pause': float (s)
        tol : float
            Acceptable temperature error (°C) for equilibrium.
        plot : bool
            Whether to show plot comparing all modes.

        Returns
        -------
        common_valid_range : tuple (min_temp, max_temp)
            Range of external temps valid for all modes.
        per_mode_valid_ranges : dict
            Dictionary of mode name -> list of valid temperatures.
        """
        import seaborn as sns
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        import numpy as np

        if self.interpolators is None:
            raise RuntimeError("Model interpolators not built. Run build_interpolated_model first.")

        # Configuración visual
        mpl.rcParams['font.family'] = 'serif'
        mpl.rcParams['font.serif'] = ['Computer Modern Roman']
        mpl.rcParams['text.usetex'] = False
        sns.set_style("whitegrid")

        candidate_ext_temps = np.linspace(-20, 50, 300)
        per_mode_valid = {}
        mode_curves = {}

        for mode in mode_list:
            name = mode.get("name", "Unnamed")
            exposure_time = mode["exposure_time"]
            valid_ext_temps = []
            T_eq_sim = []

            for ext_temp in candidate_ext_temps:
                try:
                    times, temps = self.simulate(
                        external_temp=ext_temp,
                        exposure_time=exposure_time,
                        n_frames=mode["n_frames"],
                        frame_pause=mode.get("frame_pause", 0.0),
                        burst_pause=mode.get("burst_pause", 10.0),
                        n_bursts=mode.get("n_bursts", 2),
                        only_when_exposing=True
                    )
                    T_final = np.mean(temps[-mode["n_frames"]:])
                    T_eq_sim.append(T_final)

                    if abs(T_final - T_eq_target) <= tol:
                        valid_ext_temps.append(ext_temp)
                except Exception as e:
                    T_eq_sim.append(np.nan)
                    continue

            per_mode_valid[name] = valid_ext_temps
            mode_curves[name] = T_eq_sim

        # Calcular intersección de rangos
        sets = [set(np.round(per_mode_valid[m], 2)) for m in per_mode_valid]
        common_valid = sorted(set.intersection(*sets)) if sets else []
        common_range = (min(common_valid), max(common_valid)) if common_valid else (None, None)

        if plot:
            plt.figure(figsize=(12, 6))
            colors = sns.color_palette("colorblind", n_colors=len(mode_list))

            y_min, y_max = T_eq_target - 5, T_eq_target + 5

            for i, mode in enumerate(mode_list):
                name = mode["name"]
                color = colors[i]

                # Curva simulada
                sns.lineplot(x=candidate_ext_temps,
                            y=mode_curves[name],
                            color=color,
                            linewidth=2,
                            label=fr"{name}")

                # Rango válido (relleno vertical)
                valid_x = per_mode_valid[name]
                if valid_x:
                    x_min, x_max = min(valid_x), max(valid_x)
                    plt.fill_betweenx(y=[y_min, y_max],
                                    x1=x_min, x2=x_max,
                                    color=color, alpha=0.15)

                    # Añadir marcas diagonales dentro del rango
                    for x_mark in np.linspace(x_min, x_max, 8):
                        plt.plot([x_mark - 0.3, x_mark + 0.3],
                                [T_eq_target - 0.4, T_eq_target + 0.4],
                                color=color, alpha=0.6, linewidth=1)

            # T_eq objetivo
            plt.axhline(T_eq_target, color="black", linestyle="--", label=fr"Target $T_{{\mathrm{{eq}}}}$ = {T_eq_target}°C")
            plt.fill_between(candidate_ext_temps,
                            T_eq_target - tol,
                            T_eq_target + tol,
                            color="gray", alpha=0.1, label=f"±{tol}°C tolerance")

            # Rango común
            if common_valid:
                plt.axvline(common_range[0], color="black", linestyle=":", linewidth=1)
                plt.axvline(common_range[1], color="black", linestyle=":", linewidth=1)
                common_label = f"Common range: [{common_range[0]:.1f}°C – {common_range[1]:.1f}°C]"
                plt.plot([], [], color="black", linestyle=":", label=common_label)

            plt.xlabel(r"External Temperature ($T_{\mathrm{ext}}$) [°C]")
            plt.ylabel(r"Equilibrium Sensor Temp ($T_{\mathrm{eq}}$) [°C]")
            plt.title("Thermal Equilibrium Comparison Across Modes")
            #plt.ylim(y_min, y_max)
            plt.legend()
            plt.tight_layout()
            plt.show()

        return common_range, per_mode_valid












