"""Stage 10 contracts for parameter-matched Transformer and GRU baselines."""

from dataclasses import asdict
import json
from pathlib import Path

import pytest
import torch

from encoder import Encoder
from task import BOS
from train import TrainConfig, build_model, count_parameters, load_config, train


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
RLT_FIELDS = (
    "dim", "num_encoder_layers", "num_decoder_layers", "num_heads",
    "head_dim", "window_size", "alpha", "rms_eps",
)
UNUSED_FIELDS = {
    "transformer": ("num_decoder_layers", "window_size", "alpha"),
    "gru": (
        "num_decoder_layers", "num_heads", "head_dim", "window_size",
        "alpha", "rms_eps",
    ),
}


@pytest.fixture(autouse=True)
def deterministic_float64():
    old_dtype = torch.get_default_dtype()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_default_dtype(torch.float64)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(2030)
    yield
    torch.set_default_dtype(old_dtype)
    torch.use_deterministic_algorithms(old_determinism)


def _config(model_type: str, task: str = "parity") -> TrainConfig:
    fields = dict(
        name=f"{task}_{model_type}_test", task=task, model_type=model_type,
        seed=17, eval_seed=701, dim=8, num_encoder_layers=1,
        num_decoder_layers=1, num_heads=2, head_dim=4, window_size=2,
        alpha=0.75, rms_eps=1e-6, train_length=1, batch_size=16,
        train_programs=16, steps=30, learning_rate=0.03,
        min_learning_rate=0.03, warmup_steps=1, weight_decay=0.0,
        grad_clip_norm=1.0, eval_lengths=(1,), eval_programs=4,
        eval_interval=30,
    )
    for field in UNUSED_FIELDS.get(model_type, ()):
        fields[field] = None
    return TrainConfig(**fields)


@pytest.mark.parametrize("model_type", ("transformer", "gru"))
@pytest.mark.parametrize("task,classes", (("parity", 3), ("five_state", 5)))
def test_baseline_shapes_and_transformer_reuses_encoder(model_type, task, classes):
    model = build_model(_config(model_type, task))
    if model_type == "transformer":
        assert isinstance(model.encoder, Encoder)
    tokens = torch.tensor([[BOS, 0, 1, 0], [BOS, 1, 0, 1]])
    assert model(tokens).shape == (2, 4, classes)


@pytest.mark.parametrize("model_type", ("transformer", "gru"))
def test_baseline_causality_and_document_isolation(model_type):
    model = build_model(_config(model_type))
    tokens = torch.tensor([[BOS, 0, 1, 0], [BOS, 1, 0, 1]])
    changed = tokens.clone()
    changed[0, 2] = 0
    original_logits = model(tokens)
    changed_logits = model(changed)
    assert torch.equal(original_logits[0, :2], changed_logits[0, :2])
    assert not torch.equal(original_logits[0, 2], changed_logits[0, 2])
    assert torch.equal(original_logits[1], changed_logits[1])


@pytest.mark.parametrize("task", ("parity", "five_state"))
def test_experiment_configs_match_rlt_parameter_count(task):
    configs = {
        model_type: load_config(CONFIG_DIR / f"{task}_{model_type}.json")
        for model_type in ("rlt", "transformer", "gru")
    }
    assert all(configs[kind].task == task and configs[kind].model_type == kind
               for kind in configs)
    counts = {kind: count_parameters(build_model(config))
              for kind, config in configs.items()}
    assert counts["rlt"] == (59040 if task == "parity" else 59168)
    expected = {
        "parity": {"transformer": 58120, "gru": 58800},
        "five_state": {"transformer": 58280, "gru": 59192},
    }
    for kind in ("transformer", "gru"):
        assert counts[kind] == expected[task][kind]
        assert abs(counts[kind] - counts["rlt"]) / counts["rlt"] <= 0.05


@pytest.mark.parametrize("model_type", ("transformer", "gru"))
def test_baseline_initialization_is_seed_deterministic(model_type):
    config = _config(model_type)
    tokens = torch.tensor([[BOS, 0, 1, 0]])

    def logits(seed):
        torch.manual_seed(seed)
        return build_model(config)(tokens)

    assert torch.equal(logits(41), logits(41))
    assert not torch.equal(logits(41), logits(42))


@pytest.mark.parametrize("model_type", ("rlt", "transformer", "gru"))
def test_each_model_trains_and_records_diagnostics(model_type, tmp_path):
    config = _config(model_type)
    result = train(config, output_dir=tmp_path / model_type,
                   checkpoint_dir=tmp_path / model_type / "checkpoints",
                   dtype=torch.float64)
    assert result.loss_history[-1] < result.loss_history[0]
    payload = json.loads(result.results_path.read_text(encoding="utf-8"))
    assert payload["config"]["model_type"] == model_type
    assert payload["parameter_count"] == count_parameters(result.model)
    assert len(payload["diagnostics"]) == config.steps
    for entry in payload["diagnostics"]:
        if model_type == "rlt":
            assert isinstance(entry["mean_gate"], float)
            assert isinstance(entry["mean_state_norm"], float)
        else:
            assert entry["mean_gate"] is None
            assert entry["mean_state_norm"] is None


@pytest.mark.parametrize("model_type,field", tuple(
    (model_type, field)
    for model_type, fields in UNUSED_FIELDS.items() for field in fields
))
def test_baseline_rejects_nonnull_unused_field(model_type, field):
    fields = asdict(_config(model_type))
    fields[field] = 0.5 if field in ("alpha", "rms_eps") else 1
    with pytest.raises(ValueError):
        TrainConfig(**fields)


@pytest.mark.parametrize("model_type,field", (
    ("transformer", "dim"),
    ("transformer", "num_encoder_layers"),
    ("transformer", "num_heads"),
    ("transformer", "head_dim"),
    ("transformer", "rms_eps"),
    ("gru", "dim"),
    ("gru", "num_encoder_layers"),
))
def test_baseline_requires_used_fields_to_be_nonnull(model_type, field):
    fields = asdict(_config(model_type))
    fields[field] = None
    with pytest.raises(ValueError):
        TrainConfig(**fields)


@pytest.mark.parametrize("field", RLT_FIELDS)
def test_rlt_requires_every_model_field_to_be_nonnull(field):
    fields = asdict(_config("rlt"))
    fields[field] = None
    with pytest.raises(ValueError):
        TrainConfig(**fields)


def test_unknown_model_type_raises_value_error():
    fields = asdict(_config("rlt"))
    fields["model_type"] = "unknown"
    with pytest.raises(ValueError):
        TrainConfig(**fields)
