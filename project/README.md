# Final Project — Panda Tasks

Baseline code for four Franka Panda (7-DOF) tasks. Fill in the `Controller` class in each `mainX.py` and run.

## Tasks

| File | Scene | Description |
|---|---|---|
| `main1.py` | `project1.xml` | Pick & place from a conveyor belt into a basket (suction gripper). |
| `main2.py` | `project2.xml` | Throw a ball at a target. |
| `main3.py` | `project3.xml` | Spot welding on a V-shaped workpiece (9 spots, force + hold time). |
| `main_free.py` | `project_free.xml` | Empty scene for a self-chosen task. |

## Run

```bash
python project/main1.py     # or main2.py / main3.py / main_free.py
```

A MuJoCo viewer + Tk GUI opens. The default `Controller` returns gravity compensation only — the arm hangs at home. Edit the `Controller` class in each `mainX.py` (the rest — environment classes, App wiring, logging — is handled for you).

## Logging

Use the **Start Recording** button in the GUI to toggle CSV logging. Files are saved to `project/data/` on stop or exit.

## Files

```
project/
├── main1.py / main2.py / main3.py / main_free.py
├── models/panda/
│   ├── panda_torque.xml           # Panda arm (do not edit)
│   ├── project1.xml ... project_free.xml
│   └── assets/                    # meshes (do not edit)
└── data/                          # log output
```
