"""Registry mapping policy_name (ROS param) -> PolicyModel implementation.

Add a branch here when you add a new policies/<x>.py module. Imports are
local to each branch so unrelated policies' dependencies (e.g. a heavy ML
framework) don't need to be installed unless that policy is selected.
"""
from .base_policy import PolicyModel


# policies whose constructor takes the ckpt_path parameter (overrides ckpt_path in their config.yaml)
_CKPT_PATH_POLICIES = ("omnireset",)


def create_policy(name: str, ckpt_path: str = "") -> PolicyModel:
    if ckpt_path and name not in _CKPT_PATH_POLICIES:
        raise ValueError(f"policy_name '{name}' reads ckpt_path from its config.yaml only "
                         f"(the ckpt_path parameter works for: {', '.join(_CKPT_PATH_POLICIES)})")
    if name == "dummy":
        from .dummy_policy import DummyPolicyModel
        return DummyPolicyModel()
    if name == "act":
        from .act.act_policy import ACTPolicyModel
        return ACTPolicyModel()
    if name == "pi":
        from .pi.pi_policy import PIPolicyModel
        return PIPolicyModel()
    if name == "liteflow":
        from .liteflow.liteflow_policy import LiteFlowPolicyModel
        return LiteFlowPolicyModel()
    if name == "groot":
        from .groot.groot_policy import GrootPolicyModel
        return GrootPolicyModel()
    if name == "diffusion":
        from .diffusion.diffusion_policy import DiffusionPolicyModel
        return DiffusionPolicyModel()
    if name == "omnireset":
        from .omnireset.omnireset_policy import OmniResetPolicyModel
        return OmniResetPolicyModel(ckpt_path)
    raise ValueError(f"unknown policy_name '{name}' (options: dummy, act, pi, liteflow, groot, diffusion, omnireset)")
