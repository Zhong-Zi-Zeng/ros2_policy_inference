"""Registry mapping policy_name (ROS param) -> PolicyModel implementation.

Add a branch here when you add a new policies/<x>.py module. Imports are
local to each branch so unrelated policies' dependencies (e.g. a heavy ML
framework) don't need to be installed unless that policy is selected.
"""
from .base_policy import PolicyModel


def create_policy(name: str) -> PolicyModel:
    if name == "dummy":
        from .dummy_policy import DummyPolicyModel
        return DummyPolicyModel()
    if name == "act":
        from .act.act_policy import ACTPolicyModel
        return ACTPolicyModel()
    if name == "pi":
        from .pi_policy import PIPolicyModel
        return PIPolicyModel()
    if name == "liteflow":
        from .liteflow.liteflow_policy import LiteFlowPolicyModel
        return LiteFlowPolicyModel()
    if name in ("diffusion", "diffusion_rtc"):
        # Same checkpoint and backend; diffusion_rtc adds guided Real-Time Chunking at inference
        # (policies/diffusion/rtc.py), diffusion keeps lerobot's synchronous chunking.
        from .diffusion.diffusion_policy import DiffusionPolicyModel
        return DiffusionPolicyModel(use_rtc=(name == "diffusion_rtc"))
    raise ValueError(
        f"unknown policy_name '{name}' (options: dummy, act, pi, liteflow, diffusion, diffusion_rtc)")
