#!/usr/bin/env python3
"""
Replay a recorded real-robot demonstration's actions into Isaac Sim and compare
the sim's TCP trajectory with the one the real robot followed.

Checks whether the sim applies /il/grasp/action the way the real robot did
(world frame, rotation-vector left multiplication, scale, every action applied,
TCP point). The policy is not involved: the actions come from the sample file.

  1. In Isaac Sim: Fan Grasp Mode IL, Publish Policy Observations on, press Play.
     Do NOT run a policy node: two publishers on /il/grasp/action would mix.
  2. Start this script; it waits for the sim's IL hand-off (/il/grasp/handoff).
  3. Start the task in the Delta Extension panel. At the hand-off the script
     publishes the sample's actions at 30 Hz, records /left_arm/tcp_pose_world,
     and writes <out>.csv and <out>.html (comparison report).

Both trajectories are compared as displacement from their own start, in the
robot base frame, so the start pose does not need to match the demo's.
"""
import argparse
import html
import json
import os
import time

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from scipy.spatial.transform import Rotation as Rot
from std_msgs.msg import Float64MultiArray, Int32

ACTION_TOPIC = "/il/grasp/action"
HANDOFF_TOPIC = "/il/grasp/handoff"
TCP_TOPIC = "/left_arm/tcp_pose_world"       # Link_6 pose, world frame (sim_observation_publisher.py)
TCP_OFFSET = np.array([0.0, 0.0, 0.2204])     # Link_6 -> TCP, as in config/params.yaml and the training data
# Sim world <- left arm base, from Scene-MCHD-20.usd (/world/WorkSpace/RS_M90E7A_Left)
SIM_BASE_POS = np.array([-0.2136, 0.0004, 0.5325])
SIM_BASE_QUAT_XYZW = [-0.6118, -0.3545, 0.3538, 0.6122]


class Replayer(Node):
    def __init__(self, sample, args):
        super().__init__("action_replayer")
        self.args = args
        self.close_index = None
        self.actions = sample["actions"].copy()
        if args.max_actions:
            self.actions = self.actions[: args.max_actions]
        if args.gripper == "open":
            self.actions[:, 6] = 1.0            # keep the sim's IL step running for the whole replay
        else:
            closes = np.where(self.actions[:, 6] < 0.5)[0]
            if len(closes):                     # the sim ends its IL step at the first close: stop there
                self.actions = self.actions[: closes[0] + 1]
                self.close_index = int(closes[0])
        self.pub = self.create_publisher(Float64MultiArray, ACTION_TOPIC, 10)
        self.create_subscription(PoseStamped, TCP_TOPIC, self._on_tcp, 10)
        self.create_subscription(Int32, HANDOFF_TOPIC, self._on_handoff, 10)
        self.latest = None
        self.started = None
        self.i = 0
        self.rows = []                          # (t, action index sent so far, link6 pos, link6 quat xyzw)
        self.done_at = None
        self.timer = None
        if args.start == "now":
            self.create_timer(2.0, self._start_once)

    def _on_tcp(self, msg):
        p, o = msg.pose.position, msg.pose.orientation
        self.latest = (np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w]))
        # record from the start of the replay until `settle` seconds after the last action
        if self.started is not None and (self.done_at is None or time.time() - self.done_at < self.args.settle):
            self.rows.append((time.time() - self.started, self.i, *self.latest))

    def _on_handoff(self, msg):
        if self.args.start == "handoff" and self.started is None:
            self.get_logger().info(f"IL hand-off #{msg.data}: replaying {len(self.actions)} actions at {self.args.rate} Hz")
            self._begin()

    def _start_once(self):
        if self.started is None:
            self.get_logger().info(f"--start now: replaying {len(self.actions)} actions at {self.args.rate} Hz")
            self._begin()

    def _begin(self):
        others = [i for i in self.get_publishers_info_by_topic(ACTION_TOPIC) if i.node_name != self.get_name()]
        if others:
            self.get_logger().error(f"another node publishes {ACTION_TOPIC} ({others[0].node_name}); stop it first")
            raise SystemExit(1)
        if self.latest is None:
            self.get_logger().error(f"no {TCP_TOPIC} yet: is 'Publish Policy Observations' on and the sim playing?")
            raise SystemExit(1)
        self.started = time.time()
        self.rows.append((0.0, 0, *self.latest))
        self.timer = self.create_timer(1.0 / self.args.rate, self._tick)

    def _tick(self):
        if self.i >= len(self.actions):
            if self.done_at is None:
                self.done_at = time.time()
                self.get_logger().info(f"all actions sent; recording {self.args.settle:.0f} s more for the arm to settle")
            elif time.time() - self.done_at > self.args.settle:
                raise SystemExit(0)
            return
        self.pub.publish(Float64MultiArray(data=[float(v) for v in self.actions[self.i]]))
        self.i += 1


# ---------------- analysis ----------------
def tcp_in_base(pos_w, quat_w, T_world_base):
    """World-frame Link_6/TCP pose -> base frame."""
    R_bw = T_world_base[:3, :3].T
    return (pos_w - T_world_base[:3, 3]) @ R_bw.T, Rot.from_matrix(R_bw) * Rot.from_quat(quat_w)


def analyse(sample, actions, rows, out):
    T_sim = np.eye(4); T_sim[:3, :3] = Rot.from_quat(SIM_BASE_QUAT_XYZW).as_matrix(); T_sim[:3, 3] = SIM_BASE_POS
    T_real = sample["T_realworld_base"]
    n = len(actions)
    # real: recorded TCP after each action k (k = 0..n)
    rp, rr = tcp_in_base(sample["tcp_pos_world"][: n + 1], sample["tcp_quat_xyzw_world"][: n + 1], T_real)
    real_d = (rp - rp[0]) * 1000
    real_ang = np.degrees((rr * rr[0].inv()).magnitude())
    # sim: Link_6 pose -> TCP, then base frame
    t = np.array([r[0] for r in rows]); k = np.array([r[1] for r in rows])
    l6p = np.array([r[2] for r in rows]); l6q = np.array([r[3] for r in rows])
    tcp_w = l6p + Rot.from_quat(l6q).apply(TCP_OFFSET)
    sp, sr = tcp_in_base(tcp_w, l6q, T_sim)
    sim_d = (sp - sp[0]) * 1000
    sim_ang = np.degrees((sr * sr[0].inv()).magnitude())
    # real trajectory at the action count the sim had received when each sample was taken
    real_at = real_d[np.minimum(k, n)]
    err = np.linalg.norm(sim_d - real_at, axis=1)
    # "final" = when the last action had been sent: after a close the sim's scripted grasp moves the arm
    # (lift, retreat), which is not part of the replay; that motion is reported separately.
    last = int(np.argmax(k >= n)) if (k >= n).any() else len(k) - 1
    moved_after = float(np.linalg.norm(sim_d[-1] - sim_d[last]))
    sim_d_rep, sim_ang_rep, err_rep = sim_d[: last + 1], sim_ang[: last + 1], err[: last + 1]
    fin_r, fin_s = real_d[n], sim_d[last]
    path_r = np.linalg.norm(np.diff(real_d, axis=0), axis=1).sum()
    path_s = np.linalg.norm(np.diff(sim_d_rep, axis=0), axis=1).sum()
    cosang = float(fin_r @ fin_s / (np.linalg.norm(fin_r) * np.linalg.norm(fin_s) + 1e-12))
    summary = {
        "actions_sent": int(k.max()), "replay_seconds": round(float(t[last]), 2),
        "final_displacement_real_mm": np.round(fin_r, 2).tolist(), "final_displacement_sim_mm": np.round(fin_s, 2).tolist(),
        "final_error_mm": round(float(np.linalg.norm(fin_s - fin_r)), 2),
        "net_motion_ratio_sim_over_real": round(float(np.linalg.norm(fin_s) / (np.linalg.norm(fin_r) + 1e-12)), 3),
        "path_length_real_mm": round(float(path_r), 1), "path_length_sim_mm": round(float(path_s), 1),
        "direction_agreement_deg": round(float(np.degrees(np.arccos(np.clip(cosang, -1, 1)))), 2),
        "tracking_error_mean_mm": round(float(err_rep.mean()), 2), "tracking_error_max_mm": round(float(err_rep.max()), 2),
        "final_rotation_real_deg": round(float(real_ang[n]), 2), "final_rotation_sim_deg": round(float(sim_ang_rep[-1]), 2),
        "final_rotation_error_deg": round(float(np.degrees((sr[last] * sr[0].inv() * (rr[n] * rr[0].inv()).inv()).magnitude())), 2),
        "moved_after_last_action_mm": round(moved_after, 1),
    }
    with open(out + ".csv", "w") as f:
        f.write("t_s,actions_sent,sim_dx_mm,sim_dy_mm,sim_dz_mm,sim_rot_deg,real_dx_mm,real_dy_mm,real_dz_mm,real_rot_deg,error_mm\n")
        for i in range(len(t)):
            kk = min(k[i], n)
            f.write(f"{t[i]:.3f},{k[i]},{sim_d[i,0]:.3f},{sim_d[i,1]:.3f},{sim_d[i,2]:.3f},{sim_ang[i]:.3f},"
                    f"{real_d[kk,0]:.3f},{real_d[kk,1]:.3f},{real_d[kk,2]:.3f},{real_ang[kk]:.3f},{err[i]:.3f}\n")
    write_report(out + ".html", summary, k, sim_d, sim_ang, real_d, real_ang, n)
    return summary


def write_report(path, summary, k, sim_d, sim_ang, real_d, real_ang, n):
    def panel(title, unit, real_y, sim_x, sim_y):
        W, H, L, R, T, B = 460, 210, 48, 12, 10, 26
        xs = np.arange(len(real_y)); lo = min(real_y.min(), sim_y.min()); hi = max(real_y.max(), sim_y.max())
        pad = (hi - lo) * 0.08 or 1.0; lo -= pad; hi += pad
        X = lambda v: L + v / max(n, 1) * (W - L - R)
        Y = lambda v: T + (1 - (v - lo) / (hi - lo)) * (H - T - B)
        def line(x, y, cls):
            step = max(1, len(x) // 600)
            pts = " ".join(f"{X(a):.1f},{Y(b):.1f}" for a, b in zip(x[::step], y[::step]))
            return f'<polyline class="{cls}" points="{pts}"/>'
        ticks = np.linspace(lo, hi, 5)
        grid = "".join(f'<line class="grid" x1="{L}" x2="{W-R}" y1="{Y(v):.1f}" y2="{Y(v):.1f}"/>'
                       f'<text class="tick" x="{L-6}" y="{Y(v)+4:.1f}" text-anchor="end">{v:.1f}</text>' for v in ticks)
        xt = "".join(f'<text class="tick" x="{X(v):.1f}" y="{H-8}" text-anchor="middle">{int(v)}</text>' for v in np.linspace(0, n, 5))
        return (f'<figure><figcaption><b>{html.escape(title)}</b> <span>({unit})</span></figcaption>'
                f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{html.escape(title)}">{grid}{xt}'
                f'{line(xs, real_y, "real")}{line(sim_x, sim_y, "sim")}</svg></figure>')
    panels = "".join(panel(f"{ax} displacement from start", "mm, base frame", real_d[:, i], k, sim_d[:, i]) for i, ax in enumerate("xyz"))
    panels += panel("rotation from start", "deg", real_ang, k, sim_ang)
    rows = "".join(f"<tr><td>{html.escape(a.replace('_', ' '))}</td><td>{html.escape(json.dumps(b))}</td></tr>" for a, b in summary.items())
    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>Action Replay</title>
<style>
:root {{ --bg:#f9f9f7; --surface:#fcfcfb; --ink:#0b0b0b; --ink2:#52514e; --grid:#e1e0d9; --real:#2a78d6; --sim:#eb6834; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#0d0d0d; --surface:#1a1a19; --ink:#fff; --ink2:#c3c2b7; --grid:#2c2c2a; --real:#3987e5; --sim:#d95926; }} }}
body {{ margin:0; background:var(--bg); color:var(--ink); font:14px/1.4 system-ui,-apple-system,"Segoe UI",sans-serif; }}
main {{ max-width:980px; margin:0 auto; padding:20px 16px; }}
.grid2 {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(min(100%,440px),1fr)); gap:12px; }}
figure {{ margin:0; background:var(--surface); border:1px solid rgba(127,127,127,.2); border-radius:10px; padding:10px 12px; }}
figcaption span, .legend {{ color:var(--ink2); }} svg {{ width:100%; height:auto; }}
.grid {{ stroke:var(--grid); }} .tick {{ fill:var(--ink2); font-size:10px; }}
.real, .sim {{ fill:none; stroke-width:2; stroke-linejoin:round; }} .real {{ stroke:var(--real); }} .sim {{ stroke:var(--sim); }}
.legend i {{ display:inline-block; width:14px; height:2px; vertical-align:middle; margin:0 6px 0 14px; }}
table {{ border-collapse:collapse; margin-top:12px; font-variant-numeric:tabular-nums; }} td {{ padding:3px 12px 3px 0; border-bottom:1px solid var(--grid); }}
</style></head><body><main><h1 style="font-size:20px">Action Replay: sim vs recorded real robot</h1>
<p class="legend"><i style="background:var(--real)"></i>real robot (recorded)<i style="background:var(--sim)"></i>Isaac Sim (replayed). x axis: actions sent.</p>
<div class="grid2">{panels}</div><table>{rows}</table></main></body></html>"""
    with open(path, "w") as f:
        f.write(doc)


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sample", default="demo12",
                    help="sample to replay: a name from samples/ (e.g. demo14, 14, demo14_fan_grasp.npz) or a path")
    ap.add_argument("--list", action="store_true", help="list the samples in samples/ and exit")
    ap.add_argument("--start", choices=["handoff", "now"], default="handoff",
                    help="handoff: wait for the sim's IL hand-off (default); now: start 2 s after launch")
    ap.add_argument("--gripper", choices=["open", "recorded"], default="open",
                    help="open: send gripper=1 throughout so the IL step runs the whole replay (default); "
                         "recorded: send the demo's gripper and stop at its first close (action 809 for demo 12), "
                         "where the sim ends its IL step and closes the gripper")
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--max-actions", type=int, default=0, help="replay only the first N actions (0 = all)")
    ap.add_argument("--settle", type=float, default=2.0, help="seconds to keep recording after the last action")
    ap.add_argument("--out", default=None, help="output path without extension (default: results/<sample>_<time>)")
    args = ap.parse_args()
    sample_dir = os.path.join(here, "samples")
    if args.list:
        for f in sorted(os.listdir(sample_dir)):
            if f.endswith(".npz"):
                s = np.load(os.path.join(sample_dir, f))
                print(f"{f:26s} {len(s['actions']):5d} actions  gripper close at action {int(s['close_frame']) - 1:5d}  "
                      f"{str(s['source']).split(' ', 1)[1]}")
        return
    if not os.path.exists(args.sample):
        name = args.sample if args.sample.startswith("demo") else f"demo{args.sample}"
        for cand in (name, f"{name}.npz", f"{name}_fan_grasp.npz"):
            if os.path.exists(os.path.join(sample_dir, cand)):
                args.sample = os.path.join(sample_dir, cand)
                break
        else:
            ap.error(f"sample '{args.sample}' not found in {sample_dir} (see --list)")
    if args.out is None:
        stem = os.path.basename(args.sample).replace("_fan_grasp.npz", "").replace(".npz", "")
        args.out = os.path.join(here, "results", f"{stem}_{args.gripper}_" + time.strftime("%Y%m%d-%H%M%S"))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    sample = dict(np.load(args.sample))
    rclpy.init()
    node = Replayer(sample, args)
    if node.close_index is not None:
        node.get_logger().info(f"--gripper recorded: replaying up to and including the gripper close at action {node.close_index}")
    node.get_logger().info(f"loaded {len(node.actions)} actions from {args.sample}; "
                           + ("waiting for the IL hand-off: start the task in the Delta Extension panel"
                              if args.start == "handoff" else "starting in 2 s"))
    try:
        rclpy.spin(node)
    except (SystemExit, KeyboardInterrupt, ExternalShutdownException):
        pass
    rows = node.rows
    node.destroy_node()
    rclpy.try_shutdown()
    if len(rows) < 10:
        print("too few TCP samples recorded; nothing to compare")
        return
    summary = analyse(sample, node.actions, rows, args.out)
    print(json.dumps(summary, indent=1))
    print(f"\nwrote {args.out}.csv and {args.out}.html")


if __name__ == "__main__":
    main()
