# mmwave-drone

Detecting a palm-sized drone with a 60 GHz FMCW radar (TI IWR6843AOPEVM) from its
propeller micro-Doppler, and telling it apart from a person, a fan and an empty room.

The full write-up, with physics, design choices, results and limits, is in
[`report/report.pdf`](report/report.pdf).

## Idea

A drone held in place has a static body, so the radar's static-clutter removal deletes it.
Its propellers are not static: the blade tips move at tens of m/s, close to or above the
largest speed the radar can measure without aliasing, so their echo spreads over a wide band
of Doppler velocities at the drone's range, at close range over almost the whole axis. A walking person stays within a few m/s, and a fan's blades are slower and
larger. A few physical features of that spread are enough to separate the classes.

## Results in short

| | |
|---|---|
| Training data | 28 labelled recordings in one room: empty, person, fan, drone |
| Cross-validation (recording-wise) | 90.7 % of detected one-second windows correct; drone 79 % of detected windows (228 of 289), 61 % of all drone windows (228 of 373) |
| Independent test, frozen model | 20 new recordings in two rooms, plus one with the radar 18 cm lower |
| False drone labels in the test | none in 1110 frames from 10 recordings of empty rooms, people and fans |
| Drone recognised in the test | 668 of 700 frames (95 %) in 6 recordings at 0.8 to 1.4 m, two of them free flights; 661 of 674 (98 %) without the part after the drone fell off its pole in one; 668 of 817 (82 %) with the missed recording at 1.64 m |
| Range of the simple detection gate | detected at 1.41 m (on a pole), missed at 1.64 m (on a sofa arm); the radar equation puts the limit at about 1.6 to 1.7 m |
| Radar cross section of the propellers | of the order of 1 cm² (spread of estimates 0.4 to 1.7 cm²), model-based, not calibrated |

Known weaknesses, all measured and described in the report:

- With the radar 18 cm lower, the fan was never recognised (100 % "unsure").
- With two moving targets at once (a person and a fan or drone), the answer is mostly "unsure".
- In one room the empty-room level comes close to the detection threshold.
- One drone, one person and one fan were used; other drones, rooms and geometries are untested.
- The 1110 frames are strongly correlated, so zero false alarms there is not a general false-alarm rate.

## Hardware

- TI IWR6843AOPEVM with the mmWave SDK 3.x out-of-box demo firmware, USB to a laptop
- Chirp: [`configs/md_1tx.cfg`](configs/md_1tx.cfg) (1 TX, 4 RX, 64 samples, 128 chirps,
  4 frames/s); `python3 -m mmwd.config configs/md_1tx.cfg` prints the derived numbers
- Target: a palm-sized toy quadcopter

## Code

| | |
|---|---|
| `mmwd/tlv.py` | stream parser for the demo's UART packets |
| `mmwd/config.py` | chirp parameters to range/velocity axes and limits |
| `mmwd/radar.py` | serial ports, configuration, frame reader |
| `mmwd/features.py` | physical features of one range-Doppler map |
| `mmwd/classify.py` | detection gate, classifier, live smoothing |
| `mmwd/sim.py` | point-scatterer FMCW simulator (rotors, walker, fan) |
| `mmwd/phase.py` | phase retrieval (Gauss-Newton) for the breathing follow-up |
| `mmwd/freeze.py` | checks the frozen files and model against the manifest |
| `scripts/check_board.py` | find the board, configure it, check frames |
| `scripts/record.py` | labelled recordings (parsed maps + raw bytes) |
| `scripts/train.py` | classifier with recording-wise cross-validation |
| `scripts/live.py` | live range-Doppler view with labels |
| `scripts/eval_test.py` | independent test: checks the frozen files, then replays recordings |
| `report/*.py` | the numbers and figures of the report (Appendix C lists which script makes what, and the few exceptions) |

## Data

| folder | content |
|---|---|
| `data/` | R01-R28 labelled training recordings (IDs as in the report), B breathing, M metal pot |
| `data_prop/` | P01-P07 propeller recordings (intact and cut blades) |
| `data_test/` | T01-T13 (room 1) and U01-U07 (room 2), the independent test |
| `data_test_excluded/` | test recordings set aside for set-up reasons (see the report) |
| `data_excluded/` | X01-X04 training-session recordings with a set-up fault |
| `models/frozen/` | the frozen model and checksums: `FROZEN.txt` for the published files, `FROZEN_original.txt` for the files at test time |

The checksum check (`mmwd/freeze.py`) covers `mmwd/classify.py`, `mmwd/features.py`,
`mmwd/config.py`, `configs/md_1tx.cfg` and the model, and confirms that they match the
manifest. It does not cover the recordings or the loader that reads them, and it cannot prove
which files were used before the test results were seen; that rests on my account (see the
report).

Each `.npz` holds the range-Doppler maps, range profiles, detected points, frame times
(relative to the first frame) and a JSON header with the label, distance, note and the
full chirp configuration.

## Running it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m pytest
python3 scripts/eval_test.py data_test/T09_drone_0p5m_0deg.npz   # one measured drone recording
python3 scripts/eval_test.py data_test/*.npz   # the whole independent test
python3 report/compute_results.py              # numbers of Section 6
```

Appendix C of the report lists the script behind each section, and the few figures that are
not produced by these scripts. The frozen model was saved with scikit-learn 1.9.1, which
`requirements.txt` pins; other versions may warn when loading it.

The first command replays a measured recording (the drone on a pole, about 0.8 m from the
radar) through the frozen classifier; 113 of its 115 scored frames are labelled drone.

With the board connected, `scripts/check_board.py` finds the ports and `scripts/live.py`
shows the live view. These two and `scripts/record.py` also run without hardware with
`--sim drone` (or `human`, `fan`, `empty`). The simulator tests the software chain, not the
classifier: its drone is not a calibrated copy of the real one, and the frozen model mostly
answers "unsure" on it.

## Credits

- `configs/ti_default_68xx.cfg` is the TI mmWave Demo Visualizer's default 68xx profile
  (SDK 3.5), from https://github.com/kirkster96/IWR6843-Read-Data-Python-MMWAVE-SDK, kept
  as a known-good reference.
- Packet format: TI mmWave SDK 3.x out-of-box demo documentation.
- I designed and ran the experiments: the radar setup, all recordings and the independent
  test in two rooms. For the software I worked with an AI coding assistant (Claude).
