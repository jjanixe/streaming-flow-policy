"""Reproducible oracle ablations for mode-aligned grouped steering."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from env.artifacts import load_demonstration_bank, train_data_digest
from env.config import DEFAULT_CONFIG
from env.grouped_preference import (
    GroupedPreference, classify_modes, synthetic_grouped_users)
from env.run_grouped_preference import _closed_loop, _json_values, _path_metrics
from env.run_preference import _digest, _seed
from env.run_stage_b import load_trained_sfps


_MODE_NAMES = (
    "upper_narrow", "upper_wide", "lower_narrow", "lower_wide", "other")
_ENV_DIR = Path(__file__).parent
_SOURCE_PATHS = tuple(_ENV_DIR / name for name in (
    "run_mode_ablation.py", "run_grouped_preference.py", "grouped_rollout.py",
    "grouped_preference.py", "preference_rollout.py", "environment.py",
    "config.py", "sfp_policies.py"))


def _canonical_digest(value) -> str:
    encoded = json.dumps(
        _json_values(value), sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=f".{path.name}.", delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(_json_values(value), stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _atomic_npz(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
                "wb", dir=path.parent, prefix=f".{path.name}.",
                suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
            np.savez_compressed(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_json(path: Path) -> dict:
    try:
        with path.open(encoding="utf-8") as stream:
            value = json.load(stream)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON record: {path}") from error
    if not isinstance(value, dict):
        raise ValueError(f"JSON record must contain an object: {path}")
    return value


def _wilson(successes: int, count: int) -> list[float]:
    rate = successes / count
    z = 1.96
    center = (rate + z**2 / (2 * count)) / (1 + z**2 / count)
    half = z * np.sqrt(
        rate * (1 - rate) / count + z**2 / (4 * count**2)
    ) / (1 + z**2 / count)
    return [float(center - half), float(center + half)]


def _target_mode_summary(output: dict, target_mode: str) -> dict:
    if target_mode not in _MODE_NAMES[:-1]:
        raise ValueError("target_mode must be one of the four designed modes")
    positions = output["positions"]
    lengths = output["lengths"]
    success = output["success"]
    count = len(positions)
    if count <= 0:
        raise ValueError("output must contain at least one episode")
    reached = lengths > 32
    complete = (
        (lengths == 65) & ~output["numerical_failure"] &
        ~output["action_limit_failure"])
    modes = classify_modes(positions[reached, 32, 1])
    successful = complete & success
    successful_modes = classify_modes(positions[successful, 32, 1])
    target_id = _MODE_NAMES.index(target_mode)
    target_count = int(np.sum(successful_modes == target_id))
    return {
        "target_mode": target_mode,
        "successful_target_count": target_count,
        "successful_target_rate": target_count / count,
        "successful_target_wilson_95": _wilson(target_count, count),
        "mode_denominator": int(reached.sum()),
        "mode_counts": {
            name: int(np.sum(modes == index))
            for index, name in enumerate(_MODE_NAMES)},
        "successful_mode_denominator": int(successful.sum()),
        "successful_mode_counts": {
            name: int(np.sum(successful_modes == index))
            for index, name in enumerate(_MODE_NAMES)},
    }


def _validate_positive_int_sequence(values, name: str) -> tuple[int, ...]:
    result = tuple(values)
    if (not result or any(type(value) is not int or value <= 0 for value in result)
            or len(set(result)) != len(result)):
        raise ValueError(f"{name} must contain distinct positive integers")
    return result


def _validate_marker(
        marker_path: Path, artifact_path: Path, expected: dict) -> dict:
    marker = _read_json(marker_path)
    for name, value in expected.items():
        if marker.get(name) != value:
            raise ValueError(
                f"completed condition metadata mismatch for {marker_path.name}: {name}")
    if not artifact_path.is_file():
        raise ValueError(f"completed condition artifact is missing: {artifact_path.name}")
    actual_digest = _digest(artifact_path)
    if marker.get("artifact_digest") != actual_digest:
        raise ValueError(f"completed condition artifact digest mismatch: {artifact_path.name}")
    try:
        with np.load(artifact_path, allow_pickle=False) as payload:
            if sorted(payload.files) != marker.get("artifact_fields"):
                raise ValueError("artifact field list mismatch")
            for name, value in expected.items():
                if name not in payload:
                    raise ValueError(f"artifact metadata missing: {name}")
                if str(payload[name].item()) != str(value):
                    raise ValueError(f"artifact metadata mismatch: {name}")
    except (OSError, ValueError, KeyError) as error:
        if isinstance(error, ValueError) and str(error).startswith("artifact metadata"):
            raise
        raise ValueError(f"invalid completed condition artifact: {artifact_path.name}") from error
    record = marker.get("record")
    if not isinstance(record, dict):
        raise ValueError(f"completed condition marker lacks record: {marker_path.name}")
    if marker.get("record_digest") != _canonical_digest(record):
        raise ValueError(f"completed condition record digest mismatch: {marker_path.name}")
    return record


def run_mode_ablation(
        output_dir, *, checkpoint_path, stats_path, demonstrations_path,
        seed=0, device="cpu", candidates=(8, 32, 128), continuations=4,
        rollout_count=16, integration_steps_per_action=6,
        regimes=("centered", "gaussian"), users=None,
        methods=("soft", "best"), beta=16., proposal_std=1.0,
        feature_kind="mode",
        progress=None):
    """Run or resume paired oracle conditions without retaining banks in RAM."""
    candidates = _validate_positive_int_sequence(candidates, "candidates")
    for name, value in (
            ("continuations", continuations), ("rollout_count", rollout_count),
            ("integration_steps_per_action", integration_steps_per_action)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    regimes = tuple(regimes)
    if not regimes or len(set(regimes)) != len(regimes) or any(
            value not in ("centered", "gaussian") for value in regimes):
        raise ValueError("regimes must be a distinct subset of centered and gaussian")
    methods = tuple(methods)
    if not methods or len(set(methods)) != len(methods) or any(
            value not in ("soft", "best") for value in methods):
        raise ValueError("methods must be a distinct subset of soft and best")
    if feature_kind not in ("continuous", "mode"):
        raise ValueError("feature_kind must be continuous or mode")
    if (isinstance(beta, (bool, np.bool_)) or
            not isinstance(beta, (int, float, np.integer, np.floating)) or
            not np.isfinite(beta) or float(beta) < 0):
        raise ValueError("beta must be finite and nonnegative")
    if (isinstance(proposal_std, (bool, np.bool_)) or
            not isinstance(proposal_std, (int, float, np.integer, np.floating)) or
            not np.isfinite(proposal_std) or float(proposal_std) <= 0):
        raise ValueError("proposal_std must be finite and positive")
    beta, proposal_std = float(beta), float(proposal_std)
    selected_users = dict(synthetic_grouped_users() if users is None else users)
    known_users = synthetic_grouped_users()
    if (not selected_users or any(
            name not in known_users or not isinstance(preference, GroupedPreference)
            for name, preference in selected_users.items())):
        raise ValueError("users must be a nonempty subset of synthetic grouped users")

    destination = Path(output_dir)
    if destination.exists() and not destination.is_dir():
        raise ValueError("output directory must be a directory")
    destination.mkdir(parents=True, exist_ok=True)
    notify = progress or (lambda message: None)
    started = time.perf_counter()

    checkpoint_path = Path(checkpoint_path).resolve()
    stats_path = Path(stats_path).resolve()
    demonstrations_path = Path(demonstrations_path).resolve()
    checkpoint_digest = _digest(checkpoint_path)
    stats_digest = _digest(stats_path)
    demonstrations_digest = _digest(demonstrations_path)
    bank = load_demonstration_bank(demonstrations_path)
    train_digest = train_data_digest(bank)
    del bank
    source_digests = {
        f"env/{path.name}": _digest(path) for path in _SOURCE_PATHS}
    source_digest = _canonical_digest(source_digests)
    policy, stats, metadata = load_trained_sfps(
        checkpoint_path, stats_path, expected_train_data_digest=train_digest,
        device=device)
    policy.requires_grad_(False)
    policy.eval()
    state_before = {
        name: value.detach().cpu().clone()
        for name, value in policy.state_dict().items()}
    stats_before = {
        field.name: getattr(stats, field.name).copy() for field in fields(stats)}

    config = {
        "seed": seed,
        "device": str(device),
        "candidates": list(candidates),
        "continuations": continuations,
        "rollout_count": rollout_count,
        "integration_steps_per_action": integration_steps_per_action,
        "regimes": list(regimes),
        "users": {name: asdict(value) for name, value in selected_users.items()},
        "methods": list(methods),
        "beta": beta,
        "proposal_std": proposal_std,
        "feature_kind": feature_kind,
        "checkpoint_path": str(checkpoint_path),
        "stats_path": str(stats_path),
        "demonstrations_path": str(demonstrations_path),
        "checkpoint_digest": checkpoint_digest,
        "stats_digest": stats_digest,
        "demonstrations_digest": demonstrations_digest,
        "train_data_digest": train_digest,
        "source_digest": source_digest,
        "source_digests": source_digests,
        "torch_threads": torch.get_num_threads(),
        "environment": asdict(DEFAULT_CONFIG.environment),
        "model_horizon": 16,
        "prediction_points": 9,
        "executed_commands": 8,
        "goal_cost": "(error/.1)^2 + 25*(not success)",
        "proposal_distribution": (
            "guided current latents use proposal_std * N(0,I); independent "
            "future latents use N(0,I)"),
    }
    config = _json_values(config)
    config_digest = _canonical_digest(config)
    resolved_path = destination / "resolved_config.json"
    if resolved_path.exists():
        if _read_json(resolved_path) != config:
            raise ValueError("resolved config mismatch; use a separate output directory")
    else:
        _atomic_json(resolved_path, config)

    result = {
        "synthetic_users": True,
        "checkpoint_digest": checkpoint_digest,
        "stats_digest": stats_digest,
        "train_data_digest": train_digest,
        "source_digest": source_digest,
        "config_digest": config_digest,
        "checkpoint_architecture": metadata["architecture"],
        "config": config,
        "conditions": {},
        "complete": False,
        "source_unchanged": False,
        "policy_unchanged": False,
        "stats_unchanged": False,
        "limitations": [
            "This is an oracle experiment over a finite candidate bank.",
            "A finite bank cannot establish mathematical unreachability.",
            "proposal_std changes the guided current-latent proposal distribution; "
            "it is a coverage probe, not unchanged-prior Gibbs sampling.",
            "Future-sequence search is outside this experiment.",
        ],
    }
    environment_seeds = [_seed(seed, 9, index) for index in range(rollout_count)]
    rollout_seeds = [_seed(seed, 10, index) for index in range(rollout_count)]
    condition_specs = []
    for regime in regimes:
        for profile, preference in selected_users.items():
            for method in methods:
                for candidate_count in candidates:
                    key = f"{regime}__{profile}__{method}__m{candidate_count}"
                    condition_specs.append((
                        regime, profile, preference, method, candidate_count, key))
    completed = {}
    for _, _, _, _, _, key in condition_specs:
        expected = {
            "condition_key": key,
            "config_digest": config_digest,
            "source_digest": source_digest,
            "checkpoint_digest": checkpoint_digest,
            "stats_digest": stats_digest,
            "train_data_digest": train_digest,
        }
        marker_path = destination / f"{key}.complete.json"
        if marker_path.exists():
            completed[key] = _validate_marker(
                marker_path, destination / f"{key}.npz", expected)
    result["conditions"].update(completed)
    diagnostics_path = destination / "diagnostics.json"
    _atomic_json(diagnostics_path, result)

    for regime, profile, preference, method, candidate_count, key in condition_specs:
        target_mode = profile.rsplit("_", 1)[0]
        artifact_name = f"{key}.npz"
        marker_name = f"{key}.complete.json"
        artifact_path = destination / artifact_name
        marker_path = destination / marker_name
        expected = {
            "condition_key": key,
            "config_digest": config_digest,
            "source_digest": source_digest,
            "checkpoint_digest": checkpoint_digest,
            "stats_digest": stats_digest,
            "train_data_digest": train_digest,
        }
        if key in completed:
            notify(f"{key}: validated completed artifact")
            continue

        condition_started = time.perf_counter()
        output = _closed_loop(
            policy, stats, preference,
            environment_seeds=environment_seeds,
            rollout_seeds=rollout_seeds,
            centered=regime == "centered",
            candidates=candidate_count,
            continuations=continuations,
            method=method,
            beta=beta,
            integration_steps=integration_steps_per_action,
            feature_kind=feature_kind,
            proposal_std=proposal_std,
        )
        condition_elapsed = time.perf_counter() - condition_started
        record = _path_metrics(
            output, selected_users, feature_kind=feature_kind)
        record.update(_target_mode_summary(output, target_mode))
        record.update({
            "artifact": artifact_name,
            "completion_marker": marker_name,
            "regime": regime,
            "profile": profile,
            "method": method,
            "M": candidate_count,
            "preference": asdict(preference),
            "proposal_std": proposal_std,
            "task_failure_count": rollout_count - record["success_count"],
            "elapsed_seconds": condition_elapsed,
        })
        payload = dict(output)
        payload.update({
            name: np.asarray(value, dtype=np.str_)
            for name, value in expected.items()})
        _atomic_npz(artifact_path, payload)
        marker = {
            **expected,
            "artifact": artifact_name,
            "artifact_digest": _digest(artifact_path),
            "artifact_fields": sorted(payload),
            "record_digest": _canonical_digest(record),
            "record": record,
        }
        _atomic_json(marker_path, marker)
        result["conditions"][key] = _json_values(record)
        result["elapsed_seconds"] = time.perf_counter() - started
        _atomic_json(diagnostics_path, result)
        notify(
            f"{key}: target {record['successful_target_count']}/"
            f"{rollout_count}, task failures {record['task_failure_count']}, "
            f"fallback {record['fallback_count']}")
        del output, payload

    result["policy_unchanged"] = (
        checkpoint_digest == _digest(checkpoint_path) and all(
            torch.equal(state_before[name], value.detach().cpu())
            for name, value in policy.state_dict().items()))
    result["stats_unchanged"] = (
        stats_digest == _digest(stats_path) and all(
            np.array_equal(before, getattr(stats, name))
            for name, before in stats_before.items()))
    final_source_digests = {
        f"env/{path.name}": _digest(path) for path in _SOURCE_PATHS}
    result["source_unchanged"] = (
        source_digest == _canonical_digest(final_source_digests))
    if (not result["source_unchanged"] or not result["policy_unchanged"] or
            not result["stats_unchanged"]):
        raise RuntimeError(
            "source, frozen checkpoint, policy, or statistics changed during evaluation")
    result["complete"] = True
    result["elapsed_seconds"] = time.perf_counter() - started
    result = _json_values(result)
    _atomic_json(diagnostics_path, result)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-dir", "checkpoint-path", "stats-path", "demonstrations-path"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidates", nargs="+", type=int, default=[8, 32, 128])
    parser.add_argument("--continuations", type=int, default=4)
    parser.add_argument("--rollout-count", type=int, default=16)
    parser.add_argument("--integration-steps-per-action", type=int, default=6)
    parser.add_argument(
        "--regimes", nargs="+", choices=("centered", "gaussian"),
        default=["centered", "gaussian"])
    parser.add_argument(
        "--users", nargs="+", choices=list(synthetic_grouped_users()))
    parser.add_argument(
        "--methods", nargs="+", choices=("soft", "best"),
        default=["soft", "best"])
    parser.add_argument("--beta", type=float, default=16.)
    parser.add_argument("--proposal-std", type=float, default=1.)
    parser.add_argument(
        "--feature-kind", choices=("continuous", "mode"), default="mode")
    parser.add_argument("--torch-threads", type=int, default=1)
    return parser


def main() -> None:
    parser = _parser()
    args = vars(parser.parse_args())
    threads = args.pop("torch_threads")
    if threads <= 0:
        parser.error("--torch-threads must be positive")
    torch.set_num_threads(threads)
    if args["users"] is not None:
        available = synthetic_grouped_users()
        args["users"] = {name: available[name] for name in args["users"]}
    run_mode_ablation(
        **args, progress=lambda message: print(message, flush=True))


if __name__ == "__main__":
    main()
