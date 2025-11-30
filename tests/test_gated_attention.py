import torch
import pytest
from nanochat.gpt import GPTConfig, GPT, GatedAttention


def test_gated_attention_shapes():
    """Verify correct output shapes from GatedAttention module."""
    gate = GatedAttention(n_head=6, head_dim=128, n_embd=768)
    sdpa_out = torch.randn(2, 10, 768)
    pre_norm = torch.randn(2, 10, 768)
    output = gate(sdpa_out, pre_norm)
    assert output.shape == (2, 10, 768), f"Expected (2, 10, 768), got {output.shape}"


def test_gate_values_in_range():
    """Verify sigmoid values are in [0, 1] range."""
    gate = GatedAttention(n_head=6, head_dim=128, n_embd=768, activation="sigmoid")
    x = torch.randn(1, 10, 768)
    with torch.no_grad():
        scores = torch.sigmoid(gate.gate_proj(x))
    assert (scores >= 0).all() and (scores <= 1).all(), "Gate scores must be in [0, 1]"


def test_gpt_with_gating():
    """Test full forward pass with gating enabled."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       use_gated_attention=True)
    model = GPT(config)
    model.init_weights()
    idx = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))
    loss = model(idx, targets)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_gpt_without_gating():
    """Test that model works with gating disabled."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       use_gated_attention=False)
    model = GPT(config)
    model.init_weights()
    idx = torch.randint(0, 1000, (2, 32))
    targets = torch.randint(0, 1000, (2, 32))
    loss = model(idx, targets)
    assert loss.shape == (), f"Loss should be scalar, got shape {loss.shape}"
    assert not torch.isnan(loss), "Loss should not be NaN"


def test_parameter_count():
    """Verify expected parameter increase with gating."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256)

    # Count params without gating
    config.use_gated_attention = False
    model_no_gate = GPT(config)
    params_no_gate = sum(p.numel() for p in model_no_gate.parameters())

    # Count params with gating
    config.use_gated_attention = True
    model_with_gate = GPT(config)
    params_with_gate = sum(p.numel() for p in model_with_gate.parameters())

    # Each layer adds n_embd * (n_head * head_dim) = n_embd^2 parameters
    expected_add = config.n_layer * (config.n_embd ** 2)
    actual_add = params_with_gate - params_no_gate

    assert actual_add == expected_add, \
        f"Expected {expected_add} additional params, got {actual_add}"


def test_backward_compatibility():
    """Test loading old checkpoint into new model with gating."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256)

    # Create old model without gating
    config.use_gated_attention = False
    old_model = GPT(config)
    old_model.init_weights()
    old_state = old_model.state_dict()

    # Load into new model with gating
    config.use_gated_attention = True
    new_model = GPT(config)
    new_model.load_state_dict(old_state, strict=False)  # Should work with warning

    # Verify forward pass still works
    idx = torch.randint(0, 1000, (1, 16))
    targets = torch.randint(0, 1000, (1, 16))
    loss = new_model(idx, targets)
    assert not torch.isnan(loss), "Loss should not be NaN after loading old checkpoint"


def test_gradient_flow():
    """Test that gate parameters receive gradients."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       use_gated_attention=True)
    model = GPT(config)
    model.init_weights()

    idx = torch.randint(0, 1000, (2, 16))
    targets = torch.randint(0, 1000, (2, 16))

    loss = model(idx, targets)
    loss.backward()

    # Check that gate parameters have gradients
    for name, param in model.named_parameters():
        if 'gate' in name:
            assert param.grad is not None, f"Gate parameter {name} has no gradient"
            assert not torch.isnan(param.grad).any(), f"Gate parameter {name} has NaN gradient"


def test_silu_activation():
    """Test that SiLU activation works."""
    gate = GatedAttention(n_head=6, head_dim=128, n_embd=768, activation="silu")
    sdpa_out = torch.randn(1, 10, 768)
    pre_norm = torch.randn(1, 10, 768)
    output = gate(sdpa_out, pre_norm)
    assert output.shape == (1, 10, 768), "Output shape mismatch with SiLU activation"


def test_inference_mode():
    """Test model can generate text (inference mode)."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       use_gated_attention=True)
    model = GPT(config)
    model.init_weights()
    model.eval()

    idx = torch.randint(0, 1000, (1, 16))
    with torch.no_grad():
        logits = model(idx)  # No targets = inference mode

    assert logits.shape == (1, 16, config.vocab_size), \
        f"Expected logits shape (1, 16, {config.vocab_size}), got {logits.shape}"


def test_gate_initialization():
    """Test that gate weights are initialized with small std."""
    config = GPTConfig(n_layer=2, n_head=4, n_kv_head=4, n_embd=256,
                       use_gated_attention=True)
    model = GPT(config)
    model.init_weights()

    # Check gate weight initialization
    for name, module in model.named_modules():
        if isinstance(module, GatedAttention):
            weight_std = module.gate_proj.weight.std().item()
            # Should be around 0.02 (some variance is expected)
            assert 0.01 < weight_std < 0.05, \
                f"Gate weight std {weight_std} outside expected range [0.01, 0.05]"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
