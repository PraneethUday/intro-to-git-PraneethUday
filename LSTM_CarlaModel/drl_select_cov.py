# drl_select_cov_gymnasium.py
"""
DRL CoV selection (Gymnasium-compatible)
- Uses a trained multi-output LSTM predictor (LSTM_MODEL/lstm_xyz_predictor.keras)
  and its scalers (LSTM_MODEL/scalers_xyz.pkl).
- Environment and API updated for Gymnasium (reset -> (obs, info); step -> (obs, reward, terminated, truncated, info)).
- Trains a PPO agent (stable-baselines3) to select one CoV among candidates to best match the
  predicted next position of the VU (vehicle id 45) while accounting for delay and trust.
- Saves trained PPO policy to ppo_select_cov.zip and evaluation csv to eval_predictions.csv.

Usage:
    python drl_select_cov_gymnasium.py
"""

import os
import random
import math
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf

# Use Gymnasium (maintained fork). SB3 supports it.
import gymnasium as gym
from gymnasium import spaces

from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# ---------------- Config ----------------
CSV_PATH = "trajectories.csv"           # must be present in same folder
LSTM_MODEL_DIR = "LSTM_MODEL"
LSTM_MODEL_NAME = "lstm_xyz_predictor.keras"
LSTM_SCALER_NAME = "scalers_xyz.pkl"

VU_ID = "45"         # primary vehicle id (string)
NUM_COV = 5          # number of candidate CoVs to choose from
SEQ_LEN = 8
PPO_TIMESTEPS = 40000
RANDOM_SEED = 42

# Reward coefficients
ALPHA_DELAY = 0.08
BETA_TRUST = 0.2
GAMMA_COST = 0.0

# Misc
OUT_PPO = "ppo_select_cov"
EVAL_CSV = "eval_predictions.csv"
VERBOSE = 1

# --------------- Utilities ----------------
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def auto_detect_cols(df):
    cols = [c.lower() for c in df.columns.tolist()]
    mapping = {}
    for candidate in ['time','t','sim_time','timestamp','frame','step']:
        if candidate in cols:
            mapping['time'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['vehicle_id','veh_id','id','actor_id','agent_id','vehicle']:
        if candidate in cols:
            mapping['id'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['x','pos_x','px','lon','longitude']:
        if candidate in cols:
            mapping['x'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['y','pos_y','py','lat','latitude']:
        if candidate in cols:
            mapping['y'] = df.columns[cols.index(candidate)]
            break
    for candidate in ['z','pos_z','pz','alt','height']:
        if candidate in cols:
            mapping['z'] = df.columns[cols.index(candidate)]
            break
    return mapping

def load_traces(csv_path):
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"{csv_path} not found. Put your trajectories CSV in the script folder.")
    df = pd.read_csv(csv_path)
    mapping = auto_detect_cols(df)
    required = ['time','id','x','y','z']
    if any(k not in mapping for k in required):
        raise ValueError(f"Could not detect required columns automatically. Detected: {mapping}. Please ensure CSV has time,id,x,y,z columns.")
    tcol = mapping['time']; idcol = mapping['id']; xcol = mapping['x']; ycol = mapping['y']; zcol = mapping['z']
    df = df[[tcol, idcol, xcol, ycol, zcol]].dropna()
    df = df.sort_values([idcol, tcol])
    traces = {}
    for vid, g in df.groupby(idcol):
        g = g.sort_values(tcol)
        coords = np.vstack([g[xcol].values.astype(float),
                            g[ycol].values.astype(float),
                            g[zcol].values.astype(float)]).T   # (T,3)
        times = g[tcol].values.astype(float)
        traces[str(vid)] = {"coords": coords, "times": times}
    return traces

def load_lstm_predictor(model_dir=LSTM_MODEL_DIR):
    mpath = os.path.join(model_dir, LSTM_MODEL_NAME)
    spath = os.path.join(model_dir, LSTM_SCALER_NAME)
    if not os.path.exists(mpath) or not os.path.exists(spath):
        raise FileNotFoundError(f"LSTM model or scalers not found in {model_dir}. Train LSTM first.")
    model = load_model(mpath)
    d = joblib.load(spath)
    scaler_X = d['scaler_X']
    scaler_y = d['scaler_y']
    return model, scaler_X, scaler_y

def ensure_seq_of_length(seq_coords, seq_len):
    L = seq_coords.shape[0]
    if L >= seq_len:
        return seq_coords[-seq_len:].copy()
    pad = np.repeat(seq_coords[0:1], repeats=(seq_len - L), axis=0)
    return np.vstack([pad, seq_coords]).astype(float)

def predict_next_pos_multi(raw_seq, model, scaler_X, scaler_y, predict_delta=True):
    arr = np.array(raw_seq, dtype=float)  # shape (seq_len,3)
    flat = arr.reshape(-1, 3)
    flat_scaled = scaler_X.transform(flat)
    inp = flat_scaled.reshape(1, arr.shape[0], 3)
    p_scaled = model.predict(inp, verbose=0)[0]   # shape (3,)
    p = scaler_y.inverse_transform(p_scaled.reshape(1, -1))[0]
    if predict_delta:
        return arr[-1] + p
    else:
        return p

# ---------------- Gymnasium Environment ----------------
class VuSelectEnv(gym.Env):
    """
    Gymnasium-compatible environment.
    Observation: [vu_pred(3)] + [cov_curr_pos (3*num_cov)] + [delays (num_cov)] + [trusts (num_cov)]
    Action: Discrete(num_cov)
    Reset -> (obs, info)
    Step -> (obs, reward, terminated, truncated, info)
    """
    metadata = {"render_modes": []}

    def __init__(self, traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y,
                 seq_len=SEQ_LEN, predict_delta=True):
        super().__init__()
        self.traces = traces
        self.vu_id = str(vu_id)
        self.cov_ids = [str(c) for c in cov_ids]
        self.num_cov = len(self.cov_ids)
        self.seq_len = seq_len
        self.predict_delta = predict_delta
        self.lstm_model = lstm_model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y

        obs_dim = 3 + 3 * self.num_cov + self.num_cov + self.num_cov
        self.observation_space = spaces.Box(low=-1e9, high=1e9, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.num_cov)

        # derive max_t across involved vehicles
        try:
            self.max_t = min(len(self.traces[self.vu_id]['coords']) - 2,
                             min(len(self.traces[c]['coords']) - 2 for c in self.cov_ids))
        except KeyError:
            raise KeyError("VU or CoV ids missing in traces")

        self.base_delays = np.random.uniform(0.5, 3.0, size=self.num_cov)
        self.trust_scores = np.random.uniform(0.4, 1.0, size=self.num_cov)
        self.t = self.seq_len

    def reset(self, seed=None, options=None):
        # Gymnasium-style reset -> returns (obs, info)
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)
        low = self.seq_len
        high = max(low + 1, self.max_t)
        self.t = np.random.randint(low, high)
        self.base_delays = np.random.uniform(0.5, 3.0, size=self.num_cov)
        self.trust_scores = np.random.uniform(0.4, 1.0, size=self.num_cov)
        obs = self._get_obs()
        info = {}
        return obs, info

    def _get_obs(self):
        # VU predicted next position
        vu_coords = self.traces[self.vu_id]['coords'][:self.t+1]   # include t
        seq = ensure_seq_of_length(vu_coords, self.seq_len)
        vu_pred = predict_next_pos_multi(seq, self.lstm_model, self.scaler_X, self.scaler_y, predict_delta=self.predict_delta)
        # CoV current and true next positions
        cov_curr = []
        cov_true_next = []
        for cid in self.cov_ids:
            coords = self.traces[cid]['coords']
            cov_curr.append(coords[self.t].tolist())
            cov_true_next.append(coords[self.t + 1].tolist())
        cov_curr_flat = np.array(cov_curr).reshape(-1)  # 3*num_cov
        delays = self.base_delays.copy()
        trusts = self.trust_scores.copy()
        obs = np.concatenate([vu_pred, cov_curr_flat, delays, trusts]).astype(np.float32)
        # cache true next positions for reward calculation
        self._cached = {
            'vu_pred': np.array(vu_pred),
            'cov_true_next': np.array(cov_true_next)
        }
        return obs

    def step(self, action):
        # Gymnasium-style step -> (obs, reward, terminated, truncated, info)
        assert 0 <= action < self.num_cov
        chosen_idx = int(action)
        vu_pred = self._cached['vu_pred']                       # (3,)
        cov_true_next = self._cached['cov_true_next']           # (num_cov,3)
        chosen_true_next = cov_true_next[chosen_idx]
        dist = float(np.linalg.norm(chosen_true_next - vu_pred))
        perception_gain = 1.0 / (1.0 + dist)
        delay = float(self.base_delays[chosen_idx])
        trust = float(self.trust_scores[chosen_idx])
        reward = perception_gain - ALPHA_DELAY * delay + BETA_TRUST * trust - GAMMA_COST * 0.0

        # advance time
        self.t += 1
        terminated = bool(self.t >= self.max_t)
        truncated = False

        if not terminated:
            obs = self._get_obs()
        else:
            # return a valid observation even if terminal (Gymnasium prefers returning final obs)
            obs = self._get_obs()

        info = {
            'distance': dist,
            'perception_gain': perception_gain,
            'delay': delay,
            'trust': trust,
            'chosen_cov': self.cov_ids[chosen_idx]
        }
        return obs, float(reward), terminated, truncated, info

# ---------------- Training & Eval ----------------
def build_envs_and_train(traces, vu_id, lstm_model, scaler_X, scaler_y, seq_len=SEQ_LEN, num_cov=NUM_COV):
    # choose candidate CoVs (other vehicles)
    all_vids = [v for v in traces.keys() if v != str(vu_id)]
    if len(all_vids) < num_cov:
        raise ValueError("Not enough candidate CoVs in dataset.")
    random.shuffle(all_vids)
    cov_ids = all_vids[:num_cov]

    def make_env():
        return VuSelectEnv(traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y, seq_len=seq_len, predict_delta=True)

    # vectorize (single env) for SB3
    env = DummyVecEnv([make_env])
    model = PPO("MlpPolicy", env, verbose=VERBOSE, seed=RANDOM_SEED)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    model.save(OUT_PPO)
    return model, cov_ids

def evaluate_policy(model, traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y, episodes=8):
    env = VuSelectEnv(traces, vu_id, cov_ids, lstm_model, scaler_X, scaler_y)
    results = []
    for ep in range(episodes):
        obs, info = env.reset(seed=RANDOM_SEED + ep)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            # record
            results.append({
                'chosen_cov': info['chosen_cov'],
                'distance': info['distance'],
                'perception_gain': info['perception_gain'],
                'delay': info['delay'],
                'trust': info['trust'],
                'reward': reward
            })
            done = terminated or truncated
    df = pd.DataFrame(results)
    df.to_csv(EVAL_CSV, index=False)

    sep = "=" * 80
    print("\n" + sep)
    print("EVAL RESULTS (CSV rows only) — saved to:", EVAL_CSV)
    print("chosen_cov,distance,perception_gain,delay,trust,reward")
    for _, r in df.iterrows():
        print(f"{r['chosen_cov']},{r['distance']:.6f},{r['perception_gain']:.6f},{r['delay']:.6f},{r['trust']:.6f},{r['reward']:.6f}")
    print(sep + "\n")
    print("Mean reward:", df['reward'].mean())
    print("Mean distance:", df['distance'].mean())

# ---------------- Main -----------------
def main():
    set_seed(RANDOM_SEED)
    print("Using Gymnasium + Stable-Baselines3 (PPO).")
    # load traces
    traces = load_traces(CSV_PATH)
    if str(VU_ID) not in traces:
        raise SystemExit(f"VU id {VU_ID} not found in dataset. Available ids: {list(traces.keys())[:10]}")
    # load LSTM predictor + scalers
    lstm_model, scaler_X, scaler_y = load_lstm_predictor(LSTM_MODEL_DIR)
    print("Loaded LSTM predictor and scalers.")

    # train
    print("Starting PPO training to learn CoV selection policy...")
    model, cov_ids = build_envs_and_train(traces, VU_ID, lstm_model, scaler_X, scaler_y)
    print("PPO training complete. Saved policy to:", OUT_PPO + ".zip")

    # evaluate
    print("Evaluating policy for a few episodes...")
    evaluate_policy(model, traces, VU_ID, cov_ids, lstm_model, scaler_X, scaler_y, episodes=8)

if __name__ == "__main__":
    main()
