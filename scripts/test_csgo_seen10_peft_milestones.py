#!/usr/bin/env python3
"""CPU-only checks for the formal PEFT checkpoint schedule and aliases."""

from __future__ import annotations

import copy
import hashlib
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from csgo_seen10.peft_artifact_contract import (
    EXPERIMENT, FORMAT, PEFT_CHECKPOINT_STEPS, validate_peft_checkpoint,
)
from csgo_seen10.source_compat import check_resume_identity
from scripts.validate_csgo_seen10_aligned import ContractError, check_checkpoint_contract
from scripts.validate_csgo_seen10_peft import load_config
from train_seen10_peft import MILESTONES, _update_index


CONFIG_PATH = ROOT / "configs" / f"{EXPERIMENT}.json"
OLD_STEPS = (3_900, 7_800, 11_700, 15_600, 19_500)


def expect_error(error_type: type[Exception], message: str, action) -> None:
    try:
        action()
    except error_type as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected {error_type.__name__}: {message}")


def payload_for_step(step: int, best_step: int, config: dict, identity: dict) -> dict:
    return {
        "format": FORMAT,
        "args": {"experiment": EXPERIMENT, "seed": config["seed"], "smoke": False},
        "identity": identity,
        "training_config": {
            "seed": config["seed"], "world_size": config["world_size"],
            "batch_size": config["batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
            "effective_batch_size": config["effective_batch_size"],
            "max_optimizer_steps": config["max_optimizer_steps"],
            "checkpoint_steps": list(PEFT_CHECKPOINT_STEPS),
        },
        "sampler_state": {
            "seed": config["seed"], "num_replicas": config["world_size"],
            "micro_batch_per_device": config["batch_size"],
            "gradient_accumulation_steps": config["gradient_accumulation_steps"],
        },
        "steps": step, "global_optimizer_step": step,
        "consumed_samples": step * config["effective_batch_size"],
        "best_step": best_step,
        "model_config": {"peft": {
            "rank": config["lora_rank"], "alpha": config["lora_alpha"],
            "dropout": config["lora_dropout"], "qkv_independent": True,
        }},
    }


def main() -> None:
    config = load_config(CONFIG_PATH, verify_weights=False)
    assert tuple(config["checkpoint_steps"]) == MILESTONES == PEFT_CHECKPOINT_STEPS
    with tempfile.TemporaryDirectory(prefix="peft-milestones-") as temporary:
        root = Path(temporary)
        old_config = copy.deepcopy(config)
        old_config["checkpoint_steps"] = list(OLD_STEPS)
        old_config_path = root / "old_config.json"
        old_config_path.write_text(json.dumps(old_config), encoding="utf-8")
        expect_error(ContractError, "checkpoint milestones changed",
                     lambda: load_config(old_config_path, verify_weights=False))

        data_contract = {
            "benchmark_manifest_sha256": "manifest-test",
            "minimal_dataset_report_sha256": "report-test",
        }
        files = {
            "config": hashlib.sha256(CONFIG_PATH.read_bytes()).hexdigest(),
            "manifest": data_contract["benchmark_manifest_sha256"],
            "report": data_contract["minimal_dataset_report_sha256"],
            "official_gpt": config["official_gpt_sha256"],
            "vq": config["vq_sha256"],
            "code": {"csgo_seen10/peft.py": hashlib.sha256(
                (ROOT / "csgo_seen10/peft.py").read_bytes()).hexdigest()},
        }
        identity = {"files": files, "data_root": str(root), "experiment": EXPERIMENT,
                    "benchmark_data_contract": data_contract}
        identity["identity_sha256"] = hashlib.sha256(json.dumps(
            identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()
        old_identity = copy.deepcopy(identity)
        old_identity["files"]["config"] = hashlib.sha256(
            old_config_path.read_bytes()).hexdigest()
        old_identity.pop("identity_sha256")
        old_identity["identity_sha256"] = hashlib.sha256(json.dumps(
            old_identity, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")).hexdigest()
        expect_error(ValueError, "Resume identity mismatch",
                     lambda: check_resume_identity(
                         old_identity, identity, profile="peft", allow_legacy=False))
        ledger_path = root / "source_ledger.json"
        ledger_path.write_text(json.dumps({"profiles": {"peft": {
            "source": old_identity["files"]["code"],
            "target": identity["files"]["code"],
        }}}), encoding="utf-8")
        expect_error(ValueError, "config, weights, or split identity differs",
                     lambda: check_resume_identity(
                         old_identity, identity, profile="peft", allow_legacy=True,
                         ledger_path=ledger_path))

        cases = {
            "best_before_final": (0.9, 0.8, 0.6, 0.7, 0.75),
            "best_at_final": (0.9, 0.8, 0.7, 0.6, 0.5),
        }
        for label, losses in cases.items():
            run_root = root / label
            checkpoint_dir = run_root / "checkpoints"
            checkpoint_dir.mkdir(parents=True)
            best_step = None
            best_loss = float("inf")
            payloads = {}
            for step, loss in zip(PEFT_CHECKPOINT_STEPS, losses):
                if loss < best_loss:
                    best_step, best_loss = step, loss
                payload = payload_for_step(step, best_step, config, identity)
                payloads[step] = payload
                path = checkpoint_dir / f"step_{step:06d}.pt"
                path.write_text(json.dumps(payload), encoding="utf-8")
                indexed_best, indexed_loss = _update_index(
                    checkpoint_dir, path, step, loss, final_step=PEFT_CHECKPOINT_STEPS[-1]
                )
                assert (indexed_best, indexed_loss) == (best_step, best_loss)
                if step != PEFT_CHECKPOINT_STEPS[-1]:
                    assert not (checkpoint_dir / "late.pt").exists()

            for role, selected_step in (("late", PEFT_CHECKPOINT_STEPS[-1]), ("best", best_step)):
                contract = check_checkpoint_contract(
                    run_root, role, checkpoint_steps=PEFT_CHECKPOINT_STEPS,
                    verify_all_sha256=True,
                )
                assert tuple(contract["steps"]) == PEFT_CHECKPOINT_STEPS
                alias = checkpoint_dir / f"{role}.pt"
                assert alias.samefile(checkpoint_dir / f"step_{selected_step:06d}.pt")
                validate_peft_checkpoint(
                    alias, payloads[selected_step], checkpoint_role=role,
                    config_path=CONFIG_PATH, config=config, data_root=root,
                    data_contract=data_contract, smoke=False,
                )
            expect_error(ContractError, "Checkpoint steps=",
                         lambda: check_checkpoint_contract(run_root, "late"))
            old_payload = copy.deepcopy(payloads[PEFT_CHECKPOINT_STEPS[-1]])
            old_payload["training_config"]["checkpoint_steps"] = list(OLD_STEPS)
            expect_error(ValueError, "milestone schedule mismatch",
                         lambda: validate_peft_checkpoint(
                             checkpoint_dir / "late.pt", old_payload, checkpoint_role="late",
                             config_path=CONFIG_PATH, config=config, data_root=root,
                             data_contract=data_contract, smoke=False,
                         ))
    print(json.dumps({"passed": True, "device": "cpu", "steps": PEFT_CHECKPOINT_STEPS,
                      "cases": list(cases), "old_schedule_rejected": True,
                      "old_config_resume_rejected": True,
                      "aligned_default_unchanged": True}))


if __name__ == "__main__":
    main()
