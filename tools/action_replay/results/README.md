# Action replay results

Each run of `run_replay.sh` writes `<sample>_<gripper>_<time>.csv` (per-step sim vs real TCP
displacement) and a matching `.html` report (plots + summary table).

| Run | Result |
|---|---|
| `demo14_recorded_20261009-082823` | **Sim follows the recorded actions.** Demo 14, 589 actions up to the gripper close: final error 1.9 mm, direction within 0.7°, sim moved 99.5% of the real net motion, rotation error 0.5°, tracking error mean 4.6 mm / max 8.5 mm. |

Earlier runs on 2026-10-09 (demo 12 against the sim before its action handling was fixed: 155–409 mm
final error, up to 142° off in direction) and an identical repeat of the demo 14 run were removed.
