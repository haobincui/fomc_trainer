import importlib


def _normalize_trl_availability_flags():
    trl_import_utils = importlib.import_module("trl.import_utils")
    for name, value in vars(trl_import_utils).items():
        if name.endswith("_available") and isinstance(value, tuple):
            setattr(trl_import_utils, name, value[0])
    return trl_import_utils


def _force_optional_dependency_unavailable(module_name: str) -> bool:
    trl_import_utils = importlib.import_module("trl.import_utils")
    optional_flags = {
        "vllm_ascend": "_vllm_ascend_available",
        "mergekit": "_mergekit_available",
    }
    flag_name = optional_flags.get(module_name)
    if flag_name is None or not hasattr(trl_import_utils, flag_name):
        return False
    setattr(trl_import_utils, flag_name, False)
    return True


def import_trl_grpo_symbols():
    _normalize_trl_availability_flags()
    trl = importlib.import_module("trl")
    for _ in range(3):
        try:
            return trl.GRPOTrainer, trl.ModelConfig
        except RuntimeError as exc:
            message = str(exc)
            recovered = False
            for module_name in ("vllm_ascend", "mergekit"):
                if f"No module named '{module_name}'" in message:
                    recovered = _force_optional_dependency_unavailable(module_name)
                    break
            if not recovered:
                raise
    return trl.GRPOTrainer, trl.ModelConfig
