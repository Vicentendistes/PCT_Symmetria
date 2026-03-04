import torch
from src.dataset.transforms.UnitSphereNormalization import UnitSphereNormalization

def make_cloud(center, scale, n=8192, seed=0):
    g = torch.Generator().manual_seed(seed)
    pts = torch.randn((n, 3), generator=g) * scale + torch.tensor(center, dtype=torch.float32)
    return pts

def make_syms(center_point, n=4, seed=0):
    """
    simetrías dummy: [nx,ny,nz, cx,cy,cz, conf]
    sólo nos importa que columnas 3:6 sean un punto y que se revierta bien.
    """
    g = torch.Generator().manual_seed(seed)
    nxny = torch.nn.functional.normalize(torch.randn((n, 3), generator=g), dim=-1)
    cxcy = torch.tensor(center_point, dtype=torch.float32).repeat(n, 1)
    conf = torch.rand((n, 1), generator=g)
    return torch.cat([nxny, cxcy, conf], dim=1)

def check_stats(points):
    mean = points.mean(0)
    rmax = points.norm(dim=1).max()
    return mean, rmax

def main():
    tr = UnitSphereNormalization()

    # Crea dos nubes MUY distintas
    pts0 = make_cloud(center=[ 3.0, -5.0, 10.0], scale=2.0, n=5000, seed=1)
    pts1 = make_cloud(center=[-7.0,  4.0, -2.0], scale=5.0, n=7000, seed=2)

    syms0 = make_syms(center_point=[ 3.0, -5.0, 10.0], n=5, seed=3)
    syms1 = make_syms(center_point=[-7.0,  4.0, -2.0], n=6, seed=4)

    # 1) Normaliza AMBAS (para comprobar que no se pisan)
    idx0, p0n, s0n, _, _ = tr.transform(0, pts0.clone(), syms0.clone(), None, None)
    idx1, p1n, s1n, _, _ = tr.transform(1, pts1.clone(), syms1.clone(), None, None)

    # Chequeos de esfera unitaria
    m0, r0 = check_stats(p0n)
    m1, r1 = check_stats(p1n)
    print("mean0≈0:", m0, " max_r0≈1:", r0.item())
    print("mean1≈0:", m1, " max_r1≈1:", r1.item())

    # 2) Revierte cada una por separado → deben coincidir con los originales
    _, p0r, s0r, _, _ = tr.inverse_transform(0, p0n, s0n, None, None)
    _, p1r, s1r, _, _ = tr.inverse_transform(1, p1n, s1n, None, None)

    e_pts0 = (p0r - pts0).pow(2).sum().sqrt().item()
    e_pts1 = (p1r - pts1).pow(2).sum().sqrt().item()
    e_s0   = (s0r[:, 3:6] - syms0[:, 3:6]).abs().max().item()
    e_s1   = (s1r[:, 3:6] - syms1[:, 3:6]).abs().max().item()

    print("recon error pts0:", e_pts0)
    print("recon error pts1:", e_pts1)
    print("recon error syms0 centers (∞-norm):", e_s0)
    print("recon error syms1 centers (∞-norm):", e_s1)

    # 3) Asegura que el cache se limpia (por defecto keep_cache=False)
    print("has stats idx0?", tr.has_stats(0))
    print("has stats idx1?", tr.has_stats(1))

    # 4) Comprobación de error si intentas invertir sin stats
    try:
        tr.inverse_transform(2, p0n, s0n, None, None)
    except RuntimeError as e:
        print("ok, levantó error esperado:", e)

if __name__ == "__main__":
    main()
