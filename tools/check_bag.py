#!/usr/bin/env python3
"""Run the inference node on a ros2 bag inside the container and check what it publishes.

Only the node's four input topics (config/params.yaml) are played, on ROS_DOMAIN_ID 42 by default,
so nothing reaches a robot on the network even with network_mode: host.

  ./run_docker.sh --policy-name omnireset python3 /workspace/src/ros2_policy_inference/tools/check_bag.py \\
      /checkpoints/test_IL_polciy ckpt_path:=/checkpoints/omnireset/d19b.pth

key:=value args go to inference.launch.py after policy_name:=omnireset inference_rate_hz:=30.0
(later ones win, e.g. policy_name:=dummy).
"""
import argparse
import os
import re
import signal
import subprocess
import sys
import time

import numpy as np
import yaml

LOG_PATH = "/tmp/check_bag_node.log"


def _stop(proc):
    # SIGINT to the leader only: ros2 launch / bag play forward it once (a second one interrupts the node's cleanup)
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(10)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()


def _node_log():
    with open(LOG_PATH) as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("bag")
    ap.add_argument("launch_args", nargs="*", metavar="key:=value")
    ap.add_argument("--duration", type=float, default=15.0, help="seconds of bag playback (default 15)")
    ap.add_argument("--domain", type=int, default=42, help="ROS_DOMAIN_ID of the test (default 42)")
    args = ap.parse_args()
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)   # before rclpy and the subprocesses start

    import rclpy
    from ament_index_python.packages import get_package_share_directory
    from std_msgs.msg import Float64MultiArray

    launch_args = ["policy_name:=omnireset", "inference_rate_hz:=30.0", *args.launch_args]
    largs = dict(a.split(":=", 1) for a in launch_args)   # later ones win, as in ros2 launch
    with open(os.path.join(get_package_share_directory("ros2_policy_inference"), "config", "params.yaml")) as f:
        params = yaml.safe_load(f)["policy_inference_node"]["ros__parameters"]
    inputs = [params[k] for k in ("joint_state_topic", "rgb_topic", "depth_topic", "ft_sensor_topic")]
    rate = float(largs["inference_rate_hz"])
    topic = largs.get("action_topic", params["action_topic"])
    print(f"[check_bag] ROS_DOMAIN_ID {args.domain}, launch {' '.join(launch_args)}", flush=True)

    with open(LOG_PATH, "w") as log:
        node_proc = subprocess.Popen(["ros2", "launch", "ros2_policy_inference", "inference.launch.py", *launch_args],
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    # the node warns "waiting for all observation topics" on its first tick, i.e. once the policy is loaded
    t0 = time.monotonic()
    while "waiting for all observation topics" not in _node_log():
        if node_proc.poll() is not None or time.monotonic() - t0 > 180:
            _stop(node_proc)
            print(_node_log()[-3000:])
            print("FAIL  the node did not start (log above)")
            return 1
        time.sleep(0.5)

    rclpy.init()
    node = rclpy.create_node("check_bag")
    stamps, msgs = [], []

    def on_action(msg):
        stamps.append(time.monotonic())
        msgs.append(np.asarray(msg.data, dtype=np.float64))

    node.create_subscription(Float64MultiArray, topic, on_action, 10)
    print(f"[check_bag] playing {args.bag} for up to {args.duration:g} s: {' '.join(inputs)}", flush=True)
    bag = subprocess.Popen(["ros2", "bag", "play", args.bag, "--topics", *inputs],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    end = time.monotonic() + args.duration
    while time.monotonic() < end and bag.poll() is None:
        rclpy.spin_once(node, timeout_sec=0.05)
    _stop(bag)
    log_text = _node_log()   # checked up to here: stopping the node below is not part of the test
    _stop(node_proc)
    node.destroy_node()
    rclpy.shutdown()

    for line in log_text.splitlines():
        if re.search(r"loaded from|done \(|Traceback|Error|Exception|stale", line):
            m = re.search(r"\]: (.*)", line)
            print(f"  node | {(m.group(1) if m else line)[:200]}")
    checks = [("Traceback" not in log_text, "no Python exception in the node log"),
              (len(msgs) > 0, f"received {len(msgs)} actions on {topic}")]
    if msgs:
        shape_ok = all(m.shape == (7,) for m in msgs)
        checks.append((shape_ok, "every action has 7 values [dx dy dz drx dry drz gripper]"))
        if shape_ok:
            a = np.stack(msgs)
            checks.append((bool(np.isfinite(a).all()), "all values finite"))
            print(f"  max |dpos| {np.abs(a[:, :3]).max():.4f} m, max |drot| {np.abs(a[:, 3:6]).max():.4f} rad, "
                  f"gripper values {sorted(set(np.round(a[:, 6], 3).tolist()))}")
        d = np.diff(stamps[5:])   # skip the first ticks
        if len(d) > 10:
            hz = 1.0 / d.mean()
            checks.append((hz >= 0.95 * rate, f"rate {hz:.1f} Hz (target {rate:g} Hz)"))
            checks.append((d.max() < 3.0 / rate, f"longest gap {d.max() * 1000:.0f} ms < 3 periods "
                                                 f"(p95 {np.percentile(d, 95) * 1000:.0f} ms)"))
    for ok, text in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {text}")
    return 0 if all(ok for ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
