import sys, pathlib
sys.path.append(str(pathlib.Path(__file__).resolve().parents[1]))

import torch
from src.dataset.transforms.ComposeTransform import ComposeTransform
from src.dataset.transforms.FarthestPointSampling import FarthestPointSampling
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

def make_cloud(center, scale, n=12000, seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((n, 3), generator=g) * float(scale) + torch.tensor(center, dtype=torch.float32)

def make_syms(center_point, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    nxny = torch.nn.functional.normalize(torch.randn((n, 3), generator=g), dim=-1)
    cxcy = torch.tensor(center_point, dtype=torch.float32).repeat(n, 1)
    conf = torch.rand((n, 1), generator=g)
    return torch.cat([nxny, cxcy, conf], dim=1)

def stats(points):
    return points.mean(0), points.norm(dim=1).max()

def main():
    N_TARGET = 4096
    fps = FarthestPointSampling(n_points=N_TARGET, start="centroid", keep_cache=True)
    unit = UnitSphereNormalization()
    compose = ComposeTransform([fps, unit])

    pts0 = make_cloud([ 3.0, -5.0, 10.0], scale=2.0, n=8000,  seed=1)
    pts1 = make_cloud([-7.0,  4.0, -2.0], scale=5.0, n=20000, seed=2)
    syms0 = make_syms([ 3.0, -5.0, 10.0], n=5, seed=3)
    syms1 = make_syms([-7.0,  4.0, -2.0], n=6, seed=4)

    # forward
    _, p0n, s0n, *_ = compose.transform(0, pts0.clone(), syms0.clone(), None, None)
    _, p1n, s1n, *_ = compose.transform(1, pts1.clone(), syms1.clone(), None, None)

    print("[FPS] sizes:", p0n.shape[0], p1n.shape[0], " (esperado:", N_TARGET, ")")
    assert p0n.shape[0] == min(N_TARGET, pts0.shape[0])
    assert p1n.shape[0] == min(N_TARGET, pts1.shape[0])

    m0, r0 = stats(p0n); m1, r1 = stats(p1n)
    print("[Norm] mean0≈0:", m0, " rmax0≈1:", float(r0))
    print("[Norm] mean1≈0:", m1, " rmax1≈1:", float(r1))
    assert torch.allclose(m0, torch.zeros(3), atol=2e-3)
    assert torch.allclose(m1, torch.zeros(3), atol=2e-3)
    assert abs(float(r0) - 1.0) < 2e-3
    assert abs(float(r1) - 1.0) < 2e-3

    # inverse (des-normaliza)
    _, p0r, s0r, *_ = compose.inverse_transform(0, p0n, s0n, None, None)
    _, p1r, s1r, *_ = compose.inverse_transform(1, p1n, s1n, None, None)

    # **AQUÍ** usamos los índices reales de FPS para alinear comparación
    idxs0 = fps.get_indices(0)
    idxs1 = fps.get_indices(1)
    pts0_sel = pts0.index_select(0, idxs0)
    pts1_sel = pts1.index_select(0, idxs1)

    e_pts0 = (p0r - pts0_sel).pow(2).sum().sqrt().item()
    e_pts1 = (p1r - pts1_sel).pow(2).sum().sqrt().item()
    e_s0   = (s0r[:, 3:6] - syms0[:, 3:6]).abs().max().item()
    e_s1   = (s1r[:, 3:6] - syms1[:, 3:6]).abs().max().item()

    print("[Inv] recon error pts0 (L2):", e_pts0)
    print("[Inv] recon error pts1 (L2):", e_pts1)
    print("[Inv] recon error syms0 centers (∞-norm):", e_s0)
    print("[Inv] recon error syms1 centers (∞-norm):", e_s1)

    assert e_pts0 < 1e-2
    assert e_pts1 < 1e-2
    assert e_s0   < 1e-6
    assert e_s1   < 1e-6

    print("OK: Compose(FPS -> UnitSphereNormalization) pasó todos los checks.")
    # opcional
    fps._sel.clear()


if __name__ == "__main__":
    main()
