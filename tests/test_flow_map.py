import pytest
import torch

from discrete_flow_maps import flow_map, sample_sorted_times


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flow_map_identity_shape_and_finite(dtype):
    x = torch.randn(2, 20, 4, 5).to(dtype)
    endpoint = torch.softmax(torch.randn(2, 20, 4, 5), dim=1).to(dtype)
    s = torch.tensor([0.2, 0.9])
    output = flow_map(x, endpoint, s, s, time_eps=1.0e-5)
    assert output.shape == x.shape
    assert torch.equal(output, x)
    assert torch.isfinite(output.float()).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_flow_map_reaches_endpoint_at_t_one(dtype):
    x = torch.randn(2, 20, 3, 4).to(dtype)
    endpoint = torch.softmax(torch.randn(2, 20, 3, 4), dim=1).to(dtype)
    s = torch.tensor([0.0, 0.75])
    t = torch.ones_like(s)
    output = flow_map(x, endpoint, s, t, time_eps=1.0e-5)
    torch.testing.assert_close(output, endpoint, rtol=0.02, atol=0.02)


def test_sorted_time_sampling_enforces_gap_and_keeps_endpoint_one_available():
    s, t = sample_sorted_times(
        1024, torch.device("cpu"), min_time=0.0, max_time=1.0, min_gap=1.0e-5
    )
    assert (s >= 0).all() and (t <= 1).all()
    assert ((t - s) >= 0.9999e-5).all()

