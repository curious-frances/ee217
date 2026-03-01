import argparse
import numpy as np
import pandas as pd

def build_dt(df, default_dt=0.02):
    if "dt" in df.columns:
        dt = df["dt"].to_numpy(dtype=float)
        dt[dt <= 0] = default_dt
        return dt

    if "t" in df.columns:
        t = df["t"].to_numpy(dtype=float)
        dt = np.diff(t, prepend=t[0])
        dt[dt <= 0] = default_dt
        return dt

    return np.full(len(df), default_dt)

def run_kalman_filter(df,
                      meas_std=0.35,
                      accel_std=6.0,
                      pos_var0=10.0,
                      vel_var0=50.0,
                      default_dt=0.02):

    dt_series = build_dt(df, default_dt)

    n = len(df)

    # State: [x, y, vx, vy]
    x = np.zeros((4, 1))
    P = np.diag([pos_var0, pos_var0, vel_var0, vel_var0])

    # Initialize from first measurement
    first_valid = df[["x_meas", "y_meas"]].dropna().iloc[0]
    x[0, 0] = first_valid["x_meas"]
    x[1, 0] = first_valid["y_meas"]

    H = np.array([[1, 0, 0, 0],
                  [0, 1, 0, 0]])

    I = np.eye(4)

    meas_var = meas_std ** 2
    R = np.diag([meas_var, meas_var])

    x_out = []
    y_out = []
    vx_out = []
    vy_out = []

    for i in range(n):
        dt = dt_series[i]

        # State transition matrix
        F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])

        # Process noise (constant acceleration model)
        q = accel_std ** 2
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt2 * dt2

        Q = np.array([
            [dt4/4*q, 0,         dt3/2*q, 0],
            [0,        dt4/4*q,  0,        dt3/2*q],
            [dt3/2*q,  0,        dt2*q,    0],
            [0,        dt3/2*q,  0,        dt2*q]
        ])

        x = F @ x
        P = F @ P @ F.T + Q

        if not np.isnan(df.loc[i, "x_meas"]) and not np.isnan(df.loc[i, "y_meas"]):

            if "touch" in df.columns and df.loc[i, "touch"] == 0:
                pass  # skip update if touch column says no touch
            else:
                z = np.array([[df.loc[i, "x_meas"]],
                              [df.loc[i, "y_meas"]]])

                y_residual = z - H @ x
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)

                x = x + K @ y_residual
                P = (I - K @ H) @ P

        x_out.append(x[0, 0])
        y_out.append(x[1, 0])
        vx_out.append(x[2, 0])
        vy_out.append(x[3, 0])

    df["x_filt"] = x_out
    df["y_filt"] = y_out
    df["vx"] = vx_out
    df["vy"] = vy_out
    df["speed"] = np.sqrt(np.array(vx_out)**2 + np.array(vy_out)**2)

    return df

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--infile", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--meas_std", type=float, default=0.0076)
    parser.add_argument("--accel_std", type=float, default=6.0)
    parser.add_argument("--default_dt", type=float, default=0.01)

    args = parser.parse_args()

    df = pd.read_csv(args.infile)

    df_filtered = run_kalman_filter(
        df,
        meas_std=args.meas_std,
        accel_std=args.accel_std,
        default_dt=args.default_dt
    )

    df_filtered.to_csv(args.outfile, index=False)

    print("Filtered results written to:", args.outfile)