# PEO-DESY: curve resolution of synchrotron excitation-emission matrices

Code and data for the decomposition of the excitation-emission matrices (EEMs) of three coatings grown on aluminium by plasma electrolytic oxidation (PEO): Al<sub>2</sub>O<sub>3</sub>:Sm<sup>2+/3+</sup> (S1), Al<sub>2</sub>O<sub>3</sub>:Eu<sup>2+</sup>,Sm<sup>2+/3+</sup> (S2) and the same coating with Y<sub>2</sub>O<sub>3</sub>:Eu<sup>3+</sup> (S3). The EEMs were recorded at room temperature at beamline P66 of PETRA III (DESY, Hamburg) with excitation from 120 to 332 nm.

The program separates the emission of Sm<sup>3+</sup>, Sm<sup>2+</sup>, Eu<sup>2+</sup> and Eu<sup>3+</sup> by noise-weighted multivariate curve resolution (MCR) with hierarchical alternating least squares (HALS). Paper: TODO (reference and DOI to be added).

## Files

| File | Content |
|---|---|
| `eem_mcr_decomposition.py` | Decomposition program (Python, NumPy and SciPy) |
| `eem_stitched_S1_S3.npz` | Stitched EEMs of S1, S2 and S3 (input of the program) |
| `mcr_components_reference.npz` | Output of the reference run (emission spectra S and excitation profiles C) |

## Data

`eem_stitched_S1_S3.npz` contains five arrays:

- `emission_nm`: 2631 emission wavelengths, 299.75-957.25 nm in steps of 0.25 nm.
- `excitation_nm`: 213 excitation wavelengths, 120-332 nm in steps of 1 nm.
- `S1`, `S2`, `S3`: corrected intensity of each coating (2631 emission points x 213 excitation points) in the same arbitrary units. Intensities of different coatings are not comparable in absolute terms, because coating thickness and measurement geometry differ.

The raw scans were processed in this order: removal of 45 fixed detector rows with an instrumental excess, Savitzky-Golay smoothing along emission (window 7, polynomial order 4), scaling of the three emission windows to the middle window and joining by linear cross-fading. `NaN` marks the unmeasured corner (emission below 500.7 nm at excitation above 290 nm, 6.1% of the points of the decomposed matrix). The processing steps are described in the paper and are not part of this repository.

## Method in short

The three EEMs are joined side by side into one matrix D (2201 emission points between 330 and 880 nm x 639 excitation points) and described as

D = S C<sup>T</sup> + noise,

where the columns of S are the emission spectra of five components (Sm<sup>3+</sup>, Sm<sup>2+</sup>, two Eu<sup>2+</sup> sub-bands and Eu<sup>3+</sup>) shared by the three coatings, and the columns of C are their excitation profiles. Every point is divided by its estimated standard deviation. The solution is restricted by non-negativity, by the composition of the coatings (Eu<sup>2+</sup> only in S2 and S3, Eu<sup>3+</sup> only in S3), by emission windows for each component and by smoothing of the two Eu<sup>2+</sup> spectra. Starting spectra are taken from the data at excitation wavelengths where one ion dominates. No reference spectra are used. The algorithm is deterministic.

## Requirements and use

Python 3 with NumPy and SciPy (tested with Python 3.11.16, NumPy 2.4.6 and SciPy 1.17.1).

```
python eem_mcr_decomposition.py --data eem_stitched_S1_S3.npz --out mcr_results
```

A run takes about one minute on a laptop CPU (4000 cycles) and writes `mcr_results/mcr_components.npz` with the emission spectra `S` (2201 x 5), the excitation profiles `C` (639 x 5), the wavelength axes and the component names.

## Expected output

```
input sha256: 8b2b6149636186dd26a85a498bbd59afbfe22e264961b2df0add6ef4d98aea1c
matrix (2201, 639), measured points 1320381, unmeasured 6.1 %
weighted lack of fit : 0.1096
reduced chi-square   : 8.22  (S1/S2/S3: 8.09/8.85/7.73)
explained variance   : 97.0 %
```

The saved `S` and `C` are identical to those in `mcr_components_reference.npz` (maximum difference 0.0 in a verification run).

## Code generation

The program was generated with a large language model (Claude, Anthropic). It consists of deterministic numerical code (NumPy, SciPy) and uses no neural network and no pretrained parameter. TODO: the authors confirm that they reviewed and tested the code.

## Licence and contact

TODO: licence (to be chosen by the authors) and contact.
