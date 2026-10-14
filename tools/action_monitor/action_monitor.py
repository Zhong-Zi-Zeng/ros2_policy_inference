#!/usr/bin/env python3
"""
Live monitor for the policy's action output: subscribes to the action topic
(default /il/grasp/action, Float64MultiArray [dx, dy, dz, drx, dry, drz, gripper])
and serves a web page that plots it, at http://localhost:<port>.

Read-only: it never publishes, so it can run alongside the policy and the sim.
Besides the actions it reports which node publishes the topic and, if that is
policy_inference_node, its policy_name parameter.

The page's Save PNG / Save CSV buttons write into --save-dir as <policy>_NN.png/.csv,
NN being the next number not yet used by that policy (one number per run, shared by
its PNG and CSV).

Usage (inside any ROS 2 Jazzy environment; see run_action_monitor.sh for Docker):
    python3 action_monitor.py [--topic /il/grasp/action] [--port 8765] [--save-dir DIR]
"""
import argparse
import collections
import json
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import rclpy
from rcl_interfaces.srv import GetParameters
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Int32

ACTION_DIM = 7
INDEX_HTML = Path(__file__).with_name("index.html")
POLICY_RE = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")
SAVE_NAME_RE = re.compile(r"^([A-Za-z0-9_.-]{1,64})_(\d{2,})\.(png|csv)$")
MAX_SAVE_BYTES = 50 * 1024 * 1024


class RecordingStore:
    """Numbered <policy>_NN.png/.csv files in one directory."""

    def __init__(self, directory: Path, label: str):
        self.dir = directory
        self.label = label                      # how the directory is shown to the user
        self._lock = threading.Lock()
        self._reserved = set()                  # (policy, index) handed out but not yet written

    def next_index(self, policy: str) -> int:
        with self._lock:
            used = {int(m.group(2)) for p in self.dir.iterdir()
                    if (m := SAVE_NAME_RE.match(p.name)) and m.group(1) == policy}
            used |= {i for pol, i in self._reserved if pol == policy}
            index = max(used, default=0) + 1
            self._reserved.add((policy, index))
            return index

    def save(self, name: str, data: bytes) -> str:
        m = SAVE_NAME_RE.match(name)
        if not m:
            raise ValueError(f"bad file name {name!r}")
        tmp = self.dir / f".{name}.part"
        tmp.write_bytes(data)
        tmp.replace(self.dir / name)
        return f"{self.label.rstrip('/')}/{name}"


class ActionMonitor(Node):
    def __init__(self, topic: str, policy_node: str, history: int, handoff_topic: str = "/il/grasp/handoff"):
        super().__init__("action_monitor")
        self.topic = topic
        self._lock = threading.Lock()
        self._samples = collections.deque(maxlen=history)  # (seq, wall time, [7 floats])
        self._seq = 0
        self._bad_count = 0
        self._last_bad_len = None
        self._publishers = []
        self._policy_name = None
        self._events = collections.deque(maxlen=500)          # (wall time, kind, text), for the Timeline panel

        self.create_subscription(Float64MultiArray, topic, self._on_action, 50)
        # The sim announces the IL hand-off here (delta_extension); shown as a timeline event
        self.create_subscription(Int32, handoff_topic, self._on_handoff, 10)
        self._param_client = self.create_client(GetParameters, f"{policy_node}/get_parameters")
        self.create_timer(1.0, self._refresh_status)
        self.get_logger().info(f"monitoring {topic}")

    def _on_action(self, msg: Float64MultiArray) -> None:
        data = [float(v) for v in msg.data]
        with self._lock:
            if len(data) != ACTION_DIM:
                self._bad_count += 1
                self._last_bad_len = len(data)
                return
            self._seq += 1
            self._samples.append((self._seq, time.time(), data))

    def _on_handoff(self, msg: Int32) -> None:
        self._add_event("handoff", f"IL hand-off #{msg.data}: the sim's IL step starts")

    def _add_event(self, kind: str, text: str) -> None:
        with self._lock:
            self._events.append((time.time(), kind, text))

    def _refresh_status(self) -> None:
        infos = self.get_publishers_info_by_topic(self.topic)
        publishers = sorted(
            f"{i.node_namespace.rstrip('/')}/{i.node_name}" for i in infos
        )
        with self._lock:
            before = set(self._publishers)
            self._publishers = publishers
        for p in sorted(set(publishers) - before):
            self._add_event("publisher", f"publisher appeared: {p}")
        for p in sorted(before - set(publishers)):
            self._add_event("publisher", f"publisher gone: {p}")

        if not self._param_client.service_is_ready():
            with self._lock:
                self._policy_name = None
            return
        future = self._param_client.call_async(GetParameters.Request(names=["policy_name"]))
        future.add_done_callback(self._on_policy_name)

    def _on_policy_name(self, future) -> None:
        result = future.result()
        name = result.values[0].string_value if result and result.values else None
        with self._lock:
            self._policy_name = name or None

    def snapshot(self, since: int) -> dict:
        with self._lock:
            samples = [[s, round(t, 4), d] for s, t, d in self._samples if s > since]
            return {
                "now": time.time(),
                "seq": self._seq,
                "samples": samples,
                "events": [[round(t, 4), k, txt] for t, k, txt in self._events],
                "status": {
                    "topic": self.topic,
                    "policy": self._policy_name,
                    "publishers": self._publishers,
                    "bad_count": self._bad_count,
                    "last_bad_len": self._last_bad_len,
                },
            }


def make_handler(monitor: ActionMonitor, store: RecordingStore | None):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            url = urlparse(self.path)
            query = parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                self._send(200, "text/html; charset=utf-8", INDEX_HTML.read_bytes())
            elif url.path == "/data":
                try:
                    since = int(query.get("since", ["0"])[0])
                except ValueError:
                    since = 0
                snapshot = monitor.snapshot(since)
                snapshot["save_dir"] = store.label if store else None
                self._send_json(200, snapshot)
            elif url.path == "/next-index":
                policy = query.get("policy", [""])[0]
                if store is None:
                    self._send_json(503, {"error": "saving is disabled (no --save-dir)"})
                elif not POLICY_RE.match(policy):
                    self._send_json(400, {"error": f"bad policy name {policy!r}"})
                else:
                    self._send_json(200, {"index": store.next_index(policy)})
            else:
                self._send(404, "text/plain", b"not found")

        def do_POST(self):
            url = urlparse(self.path)
            if url.path != "/save":
                self._send(404, "text/plain", b"not found")
                return
            if store is None:
                self._send_json(503, {"error": "saving is disabled (no --save-dir)"})
                return
            length = int(self.headers.get("Content-Length", 0))
            if not 0 < length <= MAX_SAVE_BYTES:
                self._send_json(400, {"error": f"bad body size {length}"})
                return
            data = self.rfile.read(length)
            try:
                path = store.save(parse_qs(url.query).get("name", [""])[0], data)
            except (ValueError, OSError) as e:
                self._send_json(400 if isinstance(e, ValueError) else 500, {"error": str(e)})
                return
            monitor.get_logger().info(f"saved {path}")
            self._send_json(200, {"path": path})

        def _send_json(self, code, obj):
            self._send(code, "application/json", json.dumps(obj).encode())

        def _send(self, code, content_type, body):
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):  # keep the terminal quiet; one line per poll is noise
            pass

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--topic", default="/il/grasp/action")
    parser.add_argument("--handoff-topic", default="/il/grasp/handoff",
                        help="IL hand-off events from the sim, shown on the Timeline")
    parser.add_argument("--policy-node", default="/policy_inference_node",
                        help="node whose policy_name parameter is shown")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--history", type=int, default=36000,
                        help="actions kept for a newly opened page (36000 = 20 min at 30 Hz)")
    parser.add_argument("--save-dir", type=Path, default=None,
                        help="where Save PNG / Save CSV write <policy>_NN files (saving off if unset)")
    parser.add_argument("--save-dir-label", default=None,
                        help="how --save-dir is shown in the page (e.g. its host path when run in Docker)")
    args = parser.parse_args()

    store = None
    if args.save_dir is not None:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        store = RecordingStore(args.save_dir, args.save_dir_label or str(args.save_dir))

    rclpy.init()
    monitor = ActionMonitor(args.topic, args.policy_node, args.history, args.handoff_topic)
    spin_thread = threading.Thread(target=rclpy.spin, args=(monitor,), daemon=True)
    spin_thread.start()
    if store:
        monitor.get_logger().info(f"Save PNG / Save CSV write to {store.label}")

    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor, store))
    server.daemon_threads = True
    monitor.get_logger().info(f"open http://localhost:{args.port} in a browser")
    try:
        server.serve_forever()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        server.server_close()
        monitor.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
