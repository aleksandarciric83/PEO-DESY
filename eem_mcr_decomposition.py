#!/usr/bin/env python3
"""
eem_mcr_decomposition.py
------------------------
Noise-weighted multivariate curve resolution (MCR) of excitation-emission matrices (EEMs)
by hierarchical alternating least squares (HALS), as used in the manuscript on
Sm2+/Sm3+/Eu2+/Eu3+ emission of PEO Al2O3 coatings (beamline P66, PETRA III).

No language model, neural network or pretrained parameter is used. The program is classical
numerical linear algebra (NumPy/SciPy). Every number it prints is computed from the input file.
The algorithm is deterministic: with start 0 (default) the result is identical on every run.

Model
-----
The three EEMs (emission 330-880 nm x excitation 120-332 nm) are placed side by side:

    D (2201 x 639) = S (2201 x 5) @ C.T (5 x 639) + E

S  : emission spectra, one column per component (shared by the three coatings)
C  : excitation profiles, one row per excitation step of each coating
E  : noise
Components: 0 Sm3+, 1 Sm2+, 2 Eu2+ (sub-band a), 3 Eu2+ (sub-band b), 4 Eu3+.

Constraints
-----------
* non-negativity of S and C
* composition: Eu2+ components zero in S1; Eu3+ zero in S1 and S2
* emission windows: Sm3+ zero <545, 678-697, >830 nm; Sm2+ zero <600 nm;
  Eu3+ zero <570 nm; both Eu2+ components zero >=620 nm
* smoothing of the two Eu2+ spectra (median 41 points + moving average 5 points;
  from 500 nm blended into median 121 points + moving average 21 points)
* the unmeasured corner of the data (emission < 500.7 nm at excitation > 290 nm; NaN in the data
  file) is treated as missing and filled with the model values at every iteration

Weights
-------
Each point is divided by its estimated standard deviation sigma_i(emission) * sigma_j(excitation):
sigma_j from the blank emission range 830-880 nm (smoothed over 5 excitation steps),
sigma_i from the robust second difference along excitation (smoothed over 41 emission points).

Usage
-----
    python eem_mcr_decomposition.py [--data eem_stitched_S1_S3.npz] [--out results] [--starts 1]

Requires: numpy, scipy.   Run time: a few minutes per start on a laptop CPU.
"""
import argparse, hashlib, os, sys, time
import numpy as np
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.optimize import nnls

SAMPLES = ["S1", "S2", "S3"]
ROLES = ["Sm3+", "Sm2+", "Eu2+ (a)", "Eu2+ (b)", "Eu3+"]
EM_LO, EM_HI = 330.0, 880.0
# emission windows in which a component is forced to zero (nm); index = component
WINDOWS = {0: [(0, 545), (678, 697), (830, 1000)], 1: [(0, 600)],
           2: [(620, 1000)], 3: [(620, 1000)], 4: [(0, 570)]}
TAIL = (500, 121, 21)        # (start nm, median window, moving-average window) for the Eu2+ tail
# sample index (0 = S1) in which a component is absent
ABSENT = {2: [0], 3: [0], 4: [0, 1]}


def sigma_rows(D, M):
    """robust noise level of every emission row from the second difference along excitation"""
    d2 = D[:, 2:] - 2 * D[:, 1:-1] + D[:, :-2]
    ok = M[:, 2:] & M[:, 1:-1] & M[:, :-2]
    out = np.full(D.shape[0], np.nan)
    for i in range(D.shape[0]):
        v = d2[i][ok[i]]
        if len(v) > 10:
            out[i] = 1.4826 * np.median(np.abs(v - np.median(v))) / np.sqrt(6)
    return out


class Problem:
    def __init__(self, path):
        z = np.load(path)
        self.em_full, self.ex = z["emission_nm"], z["excitation_nm"]
        iem = (self.em_full >= EM_LO) & (self.em_full <= EM_HI)
        self.em = self.em_full[iem]
        self.D = {s: z[s][iem, :].astype(float) for s in SAMPLES}
        self.M = {s: np.isfinite(self.D[s]) for s in SAMPLES}
        for s in SAMPLES:
            self.D[s] = np.where(self.M[s], self.D[s], 0.0)
        self.nex = len(self.ex)
        self.Daug = np.hstack([self.D[s] for s in SAMPLES])
        self.Maug = np.hstack([self.M[s] for s in SAMPLES])
        self.sample_of_col = np.repeat(np.arange(len(SAMPLES)), self.nex)
        # ---- noise model ----
        rowsb = (self.em >= 830) & (self.em <= 880)
        self.sig_j = np.concatenate([uniform_filter1d(
            np.array([np.std(self.D[s][rowsb, j]) for j in range(self.nex)]), 5, mode="nearest") for s in SAMPLES])
        sr = np.mean([sigma_rows(self.D[s], self.M[s]) for s in SAMPLES], axis=0)
        self.sig_i = uniform_filter1d(np.where(np.isfinite(sr), sr, np.nanmedian(sr)), 41, mode="nearest")
        self.sj = self.sig_j / np.mean(self.sig_j)
        self.si = self.sig_i / np.mean(self.sig_i)
        self.Dw = self.Daug / (self.si[:, None] * self.sj[None, :])
        self.Dwn2 = float((self.Dw[self.Maug] ** 2).sum())
        self.sigma_ij = self.sig_i[:, None] * (self.sig_j / np.mean(self.sig_j))[None, :]
        # ---- constraints ----
        self.Zc = np.zeros((self.Daug.shape[1], 5), bool)           # composition zeros in C
        for comp, absent in ABSENT.items():
            for s in absent:
                self.Zc[self.sample_of_col == s, comp] = True
        self.Zs = np.zeros((len(self.em), 5), bool)                 # emission-window zeros in S
        for comp, wl in WINDOWS.items():
            for a, b in wl:
                self.Zs[(self.em >= a) & (self.em < b), comp] = True

    def start_spectra(self):
        """initial emission spectra from places where one ion dominates (excitation fingerprints)"""
        def sl(s, a, b):
            m = (self.ex >= a) & (self.ex <= b)
            return np.mean(self.D[s][:, m], axis=1)
        pos = lambda x: np.maximum(x, 0)
        unit = lambda x: x / np.max(x)
        sm3, sm2 = sl("S1", 163, 168), sl("S1", 324, 332)
        eu2, eu3 = sl("S2", 252, 258), sl("S3", 205, 209)
        eu2_base = median_filter(pos(eu2 - 0.5 * sm2), size=41, mode="nearest")
        wblue = 1 / (1 + np.exp((self.em - 440) / 15))
        return np.column_stack([unit(pos(sm3)), unit(pos(sm2)), unit(eu2_base * wblue),
                                unit(eu2_base * (1 - wblue) + 1e-3), unit(pos(eu3))])

    # ---------------- goodness of fit ----------------
    def lof_w(self, Ct, St):
        R = (self.Dw - St @ Ct.T)[self.Maug]
        return float(np.sqrt((R ** 2).sum() / self.Dwn2))

    def chi2_red(self, S, C, cols=None):
        R = (self.Daug - S @ C.T)
        sel = self.Maug if cols is None else (self.Maug & cols[None, :])
        return float(np.mean((R[sel] / self.sigma_ij[sel]) ** 2))

    def explained_variance(self, S, C):
        R = (self.Daug - S @ C.T)[self.Maug]
        return float(1 - (R ** 2).sum() / (self.Daug[self.Maug] ** 2).sum())

    # ---------------- HALS ----------------
    def hals(self, S0, win=41, max_iter=4000, tol=1e-9, check=25, verbose=True):
        k = S0.shape[1]
        St = S0 / self.si[:, None]
        St[self.Zs] = 0
        Ct = np.zeros((self.Daug.shape[1], k))
        for j in range(self.Daug.shape[1]):
            v = self.Maug[:, j]
            Ct[j], _ = nnls(St[v], self.Dw[v, j])
        Ct[self.Zc] = 0
        smooth = (2, 3)
        wt = 1 / (1 + np.exp(-(self.em - TAIL[0]) / 4.0))
        prev = np.inf
        for it in range(max_iter):
            Dh = St @ Ct.T
            Df = np.where(self.Maug, self.Dw, Dh)            # missing corner filled with the model
            StS = St.T @ St; DtS = Df.T @ St
            for r in range(k):                               # update excitation profiles
                num = DtS[:, r] - Ct @ StS[:, r] + Ct[:, r] * StS[r, r]
                Ct[:, r] = np.maximum(0, num / max(StS[r, r], 1e-18))
                Ct[self.Zc[:, r], r] = 0
            CtC = Ct.T @ Ct; DC = Df @ Ct
            for r in range(k):                               # update emission spectra
                num = DC[:, r] - St @ CtC[:, r] + St[:, r] * CtC[r, r]
                St[:, r] = np.maximum(0, num / max(CtC[r, r], 1e-18))
                St[self.Zs[:, r], r] = 0
                if r in smooth:
                    ph = St[:, r] * self.si
                    p1 = uniform_filter1d(median_filter(ph, size=win, mode="nearest"), 5, mode="nearest")
                    p2 = uniform_filter1d(median_filter(ph, size=TAIL[1], mode="nearest"), TAIL[2], mode="nearest")
                    St[:, r] = ((1 - wt) * p1 + wt * p2) / self.si
                    St[self.Zs[:, r], r] = 0                 # zeros are re-imposed after smoothing
            mx = (St * self.si[:, None]).max(axis=0); mx[mx == 0] = 1
            St /= mx; Ct *= mx
            if it % check == 0:
                cur = self.lof_w(Ct, St)
                if verbose and it % 250 == 0:
                    print(f"  iteration {it:5d}  weighted lack of fit {cur:.6f}", flush=True)
                if abs(prev - cur) < tol * max(cur, 1e-12):
                    break
                prev = cur
        S = St * self.si[:, None]; C = Ct * self.sj[:, None]
        return S, C, Ct, St, self.lof_w(Ct, St), it


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="eem_stitched_S1_S3.npz")
    ap.add_argument("--out", default="mcr_results")
    ap.add_argument("--starts", type=int, default=1, help="number of starts (start 0 = fingerprint, others jittered)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    print("numpy", np.__version__, "| python", sys.version.split()[0])
    print("input sha256:", hashlib.sha256(open(a.data, "rb").read()).hexdigest())
    P = Problem(a.data)
    print(f"matrix {P.Daug.shape}, measured points {int(P.Maug.sum())}, unmeasured {100*(1-P.Maug.mean()):.1f} %")
    S_init = P.start_spectra()
    fits = []
    for sd in range(a.starts):
        S0 = S_init.copy()
        if sd > 0:   # jittered start of the two Eu2+ sub-bands
            r = np.random.default_rng(sd)
            S0[:, 2:4] = np.maximum(S0[:, 2:4] * (1 + 1.5 * (r.random(S0[:, 2:4].shape) - 0.5))
                                    + 1.5 * 0.02 * r.random(S0[:, 2:4].shape), 0)
        t = time.time()
        out = P.hals(S0)
        fits.append(out)
        print(f"start {sd}: weighted lack of fit {out[4]:.5f} | {out[5]} iterations | {time.time()-t:.0f} s", flush=True)
    S, C, Ct, St, lof, it = min(fits, key=lambda o: o[4])
    print("\n=== result ===")
    print(f"weighted lack of fit : {lof:.4f}")
    print(f"reduced chi-square   : {P.chi2_red(S, C):.2f}  (S1/S2/S3: " +
          "/".join(f"{P.chi2_red(S, C, P.sample_of_col == j):.2f}" for j in range(3)) + ")")
    print(f"explained variance   : {100*P.explained_variance(S, C):.1f} %")
    print(f"Eu2+ emission above 620 nm (max of components a,b): {S[P.em >= 620][:, 2:4].max():.2e}")
    np.savez_compressed(os.path.join(a.out, "mcr_components.npz"), emission_nm=P.em, excitation_nm=P.ex,
                        S=S, C=C, roles=np.array(ROLES))
    print("saved", os.path.join(a.out, "mcr_components.npz"))


if __name__ == "__main__":
    main()
