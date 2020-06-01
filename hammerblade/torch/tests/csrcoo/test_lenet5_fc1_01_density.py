import torch

def test_lenet5_fc1_01_density():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(1, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)