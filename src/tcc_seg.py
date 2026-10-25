# tcc_seg.py | TCC 2 (Felipe Augusto Sousa de Almada)
# Segmentação 3D de fumaça sem ground truth: preparação dos dados, pseudo-rótulos,
# 4 arquiteturas (3D U-Net, V-Net, PointNet++, Point Transformer), métricas sem GT, tabelas e figuras.
import os, json, time, math, traceback, warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
from pathlib import Path
import numpy as np
from scipy import ndimage

GRADE = (64, 112, 64)  # x, y (vertical), z: divisível por 16 (exigência da V-Net)
BASE = Path(os.environ.get('TCC_BASE', '/content/drive/MyDrive/TCC2_FluidNexus'))
SEG = BASE / 'segmentacao'
DADOS, MODELOS, RESULT = SEG / 'dados', SEG / 'modelos', SEG / 'resultados'
FIGS = BASE / 'figuras'
SF_SEQS = [f'sim_{i:06d}' for i in range(int(os.environ.get('TCC_N_SF', 10)))]
N_TREINO = int(os.environ.get('TCC_N_TREINO', 6))
N_VAL = int(os.environ.get('TCC_N_VAL', 2))
SPLIT = {'treino': SF_SEQS[:N_TREINO], 'validacao': SF_SEQS[N_TREINO:N_TREINO + N_VAL],
         'teste': SF_SEQS[N_TREINO + N_VAL:]}
FN_FRAMES = {'ScalarReal': 141, 'Smoke': 71, 'Ball': 71}
DOMINIO = {'ScalarReal': 'FluidNexus ScalarReal', 'Smoke': 'FluidNexus Smoke', 'Ball': 'FluidNexus Ball (esparsa)'}
LIMIAR_CAND = 0.02  # voxels com densidade normalizada acima disso são "candidatos" a fumaça
MINUTOS = float(os.environ.get('TCC_MINUTOS', 20))  # tempo máximo de treino por modelo
MODELOS_VOL = ['unet3d', 'vnet']
MODELOS_PTS = ['pointnet2', 'point_transformer']
NOMES = {'limiar': 'Limiar (pseudo-rótulo)', 'unet3d': '3D U-Net', 'vnet': 'V-Net',
         'pointnet2': 'PointNet++', 'point_transformer': 'Point Transformer'}


def log(*a):
    print(*a, flush=True)


def salvar_json(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False))


# ---------------------------------------------------------------------------
# 1) PREPARAÇÃO: grade comum 64x112x64, densidade normalizada [0,1] e pseudo-rótulos
# ---------------------------------------------------------------------------
def normalizar(d):
    pos = d[d > 0]
    esc = float(np.percentile(pos, 99.9)) if pos.size > 100 else float(d.max() or 1.0)
    return np.clip(d / max(esc, 1e-8), 0, 1).astype(np.float32)


def pseudo_rotulo(d):
    """Limiar de Otsu (por sequência) sobre os voxels com fumaça + remoção de partes < 20 voxels."""
    from skimage.filters import threshold_otsu
    vals = d[d > LIMIAR_CAND]
    thr = float(threshold_otsu(vals)) if vals.size > 100 else 0.1
    thr = float(np.clip(thr, 0.05, 0.6))
    m = d > thr
    for t in range(m.shape[0]):
        lab, n = ndimage.label(m[t])
        if n > 1:
            tam = ndimage.sum(np.ones_like(lab), lab, index=np.arange(1, n + 1))
            manter = np.zeros(n + 1, bool); manter[1:] = tam >= 20
            m[t] = manter[lab]
    return m, thr


def reamostrar(vol, forma):
    fat = [a / b for a, b in zip(forma, vol.shape[:3])]
    return ndimage.zoom(vol, fat, order=1)


def _salvar_seq(nome, dens, rot, thr, dominio, split, vel=None):
    DADOS.mkdir(parents=True, exist_ok=True)
    extra = {} if vel is None else {'vel': vel.astype(np.float16)}
    np.savez_compressed(DADOS / f'{nome}.npz', dens=dens.astype(np.float16), rot=np.packbits(rot.reshape(-1)),
                        forma=np.array(rot.shape), **extra)
    salvar_json(DADOS / f'{nome}.json', {'nome': nome, 'dominio': dominio, 'split': split, 'limiar_otsu': thr,
                                         'frames': int(dens.shape[0]), 'grade': list(GRADE),
                                         'ocupacao_rotulo_pct': float(100 * rot.mean())})
    log(f'  {nome}: {dens.shape[0]} frames | limiar {thr:.3f} | rótulo ocupa {100 * rot.mean():.2f}% | '
        f'{"com" if vel is not None else "sem"} velocidade')


def carregar_seq(nome, com_vel=False):
    z = np.load(DADOS / f'{nome}.npz')
    forma = tuple(z['forma'])
    rot = np.unpackbits(z['rot'])[:int(np.prod(forma))].reshape(forma).astype(bool)
    meta = json.loads((DADOS / f'{nome}.json').read_text())
    vel = z['vel'] if (com_vel and 'vel' in z.files) else None
    return z['dens'], rot, meta, vel


def preparar_scalarflow():
    pasta = BASE / 'dados' / 'scalarflow'
    for seq in SF_SEQS:
        if (DADOS / f'{seq}.npz').exists():
            log(f'  {seq}: já preparado.'); continue
        arq = pasta / seq / 'densidade.npz'
        if not arq.exists():
            log(f'  {seq}: AUSENTE em {arq} (pulado)'); continue
        D = np.load(arq)['densidade'].astype(np.float32)
        D = np.stack([reamostrar(d, GRADE) for d in D])
        D = normalizar(D)
        rot, thr = pseudo_rotulo(D)
        split = next(k for k, v in SPLIT.items() if seq in v)
        vel = None
        if split == 'teste' and (pasta / seq / 'velocidade_meia_res.npz').exists():
            V = np.load(pasta / seq / 'velocidade_meia_res.npz')['velocidade'].astype(np.float32)
            fator = GRADE[0] / 100.0  # velocidade em células da grade original (100) por frame -> grade 64
            vel = np.stack([np.stack([reamostrar(v[..., c], GRADE) for c in range(3)], -1) * fator for v in V])
        _salvar_seq(seq, D, rot, thr, 'ScalarFlow', split, vel)


def _nuvem(fluido, i):
    for sub in ['fisico', 'visual']:
        p = BASE / 'modelos' / f'{fluido}_20_160_alta' / sub / 'quantities' / f'frame_{i:03d}_visual_xyz.npy'
        if p.exists():
            return np.load(p)
    return None


def voxelizar(nuvens):
    """Nuvens de pontos (FluidNexus) -> volumes de densidade na grade comum (mesma escala em x, y e z)."""
    amostra = np.concatenate(nuvens[::max(1, len(nuvens) // 12)])
    lo, hi = np.percentile(amostra, 0.5, axis=0), np.percentile(amostra, 99.5, axis=0)
    centro = (lo + hi) / 2
    esc = max((hi - lo)[k] / (GRADE[k] * 0.9) for k in range(3))
    # eixo vertical do FluidNexus = coluna 1; a fumaça sobe -> a fonte deve ficar embaixo
    sobe = np.median(nuvens[-1][:, 1]) - np.median(nuvens[0][:, 1])
    sinal = 1.0 if sobe >= 0 else -1.0
    vols = []
    for p in nuvens:
        q = (p - centro) / esc
        q[:, 1] *= sinal
        idx = np.floor(q + np.array(GRADE) / 2).astype(int)
        ok = np.all((idx >= 0) & (idx < np.array(GRADE)), axis=1)
        v = np.zeros(GRADE, np.float32)
        np.add.at(v, tuple(idx[ok].T), 1.0)
        vols.append(ndimage.gaussian_filter(v, 1.0))
    return np.stack(vols)


def preparar_fluidnexus():
    for fluido, n in FN_FRAMES.items():
        if (DADOS / f'{fluido}.npz').exists():
            log(f'  {fluido}: já preparado.'); continue
        nuvens = [_nuvem(fluido, i) for i in range(n)]
        nuvens = [p for p in nuvens if p is not None and len(p) > 0]
        if len(nuvens) < 5:
            log(f'  {fluido}: nuvens não encontradas (pulado)'); continue
        D = normalizar(np.log1p(voxelizar(nuvens)))  # log1p: evita que um aglomerado denso domine
        rot, thr = pseudo_rotulo(D)
        _salvar_seq(fluido, D, rot, thr, DOMINIO[fluido], 'teste')


def etapa_preparar():
    log('=== ETAPA 1: padronizar dados e gerar pseudo-rótulos ===')
    t0 = time.time()
    preparar_scalarflow()
    preparar_fluidnexus()
    log(f'Etapa 1 concluída em {(time.time() - t0) / 60:.1f} min')


# ---------------------------------------------------------------------------
# 2) MODELOS DE PONTOS (PyTorch puro: sem compilar nada)
# ---------------------------------------------------------------------------
def _torch():
    import torch, torch.nn as nn, torch.nn.functional as F
    return torch, nn, F


def construir_modelo(tipo):
    torch, nn, F = _torch()

    def knn(q, p, k):
        return torch.cdist(q, p).topk(k, dim=-1, largest=False).indices

    def juntar(x, idx):  # x (B,N,C), idx (B,M,k) -> (B,M,k,C)
        b = torch.arange(x.shape[0], device=x.device).view(-1, 1, 1)
        return x[b, idx]

    def fps(p, m):  # amostragem por ponto mais distante
        B, N, _ = p.shape
        idx = torch.zeros(B, m, dtype=torch.long, device=p.device)
        dist = torch.full((B, N), float('inf'), device=p.device)
        far = torch.randint(0, N, (B,), device=p.device)
        ar = torch.arange(B, device=p.device)
        for i in range(m):
            idx[:, i] = far
            d = ((p - p[ar, far][:, None]) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            far = dist.argmax(-1)
        return idx

    def interpolar(xyz_d, xyz_s, f_s):  # 3 vizinhos, peso pelo inverso da distância
        d = torch.cdist(xyz_d, xyz_s)
        dist, idx = d.topk(min(3, xyz_s.shape[1]), dim=-1, largest=False)
        w = 1.0 / (dist + 1e-8); w = w / w.sum(-1, keepdim=True)
        return (juntar(f_s, idx) * w[..., None]).sum(2)

    class MLP(nn.Module):
        def __init__(s, dims):
            super().__init__()
            camadas = []
            for a, b in zip(dims[:-1], dims[1:]):
                camadas += [nn.Linear(a, b), nn.LayerNorm(b), nn.ReLU(inplace=True)]
            s.net = nn.Sequential(*camadas)

        def forward(s, x):
            return s.net(x)

    if tipo == 'unet3d':
        from monai.networks.nets import UNet
        return UNet(spatial_dims=3, in_channels=1, out_channels=1, channels=(16, 32, 64, 128),
                    strides=(2, 2, 2), num_res_units=2)
    if tipo == 'vnet':
        from monai.networks.nets import VNet
        return VNet(spatial_dims=3, in_channels=1, out_channels=1)

    if tipo == 'pointnet2':
        class SA(nn.Module):
            def __init__(s, m, k, dims):
                super().__init__(); s.m, s.k, s.mlp = m, k, MLP(dims)

            def forward(s, xyz, f):
                i = fps(xyz, s.m)
                nxyz = juntar(xyz, i[..., None])[:, :, 0]
                idx = knn(nxyz, xyz, s.k)
                g = torch.cat([juntar(xyz, idx) - nxyz[:, :, None], juntar(f, idx)], -1)
                return nxyz, s.mlp(g).max(2).values

        class FP(nn.Module):
            def __init__(s, dims):
                super().__init__(); s.mlp = MLP(dims)

            def forward(s, xyz_d, f_d, xyz_s, f_s):
                return s.mlp(torch.cat([interpolar(xyz_d, xyz_s, f_s), f_d], -1))

        class PointNet2(nn.Module):
            def __init__(s):
                super().__init__()
                s.ent = MLP([4, 32])
                s.sa1 = SA(1024, 24, [3 + 32, 32, 64])
                s.sa2 = SA(256, 24, [3 + 64, 64, 128])
                s.sa3 = SA(64, 24, [3 + 128, 128, 256])
                s.fp3 = FP([256 + 128, 128])
                s.fp2 = FP([128 + 64, 64])
                s.fp1 = FP([64 + 32, 64])
                s.saida = nn.Linear(64, 1)

            def forward(s, x):  # x (B,N,4) = xyz + densidade
                xyz, f0 = x[..., :3], s.ent(x)
                x1, f1 = s.sa1(xyz, f0); x2, f2 = s.sa2(x1, f1); x3, f3 = s.sa3(x2, f2)
                f2 = s.fp3(x2, f2, x3, f3); f1 = s.fp2(x1, f1, x2, f2); f0 = s.fp1(xyz, f0, x1, f1)
                return s.saida(f0)[..., 0]
        return PointNet2()

    if tipo == 'point_transformer':
        class Bloco(nn.Module):  # atenção vetorial com codificação de posição (Zhao et al., 2021)
            def __init__(s, c, k=16):
                super().__init__(); s.k = k
                s.q, s.kk, s.v = nn.Linear(c, c), nn.Linear(c, c), nn.Linear(c, c)
                s.pos = nn.Sequential(nn.Linear(3, c), nn.ReLU(inplace=True), nn.Linear(c, c))
                s.att = nn.Sequential(nn.Linear(c, c), nn.ReLU(inplace=True), nn.Linear(c, c))
                s.out, s.norm = nn.Linear(c, c), nn.LayerNorm(c)

            def forward(s, xyz, f):
                idx = knn(xyz, xyz, s.k)
                pe = s.pos(xyz[:, :, None] - juntar(xyz, idx))
                w = s.att(s.q(f)[:, :, None] - juntar(s.kk(f), idx) + pe).softmax(dim=2)
                y = (w * (juntar(s.v(f), idx) + pe)).sum(2)
                return s.norm(f + s.out(y))

        class Desce(nn.Module):
            def __init__(s, ci, co, fator=4, k=16):
                super().__init__(); s.fator, s.k, s.mlp = fator, k, MLP([3 + ci, co])

            def forward(s, xyz, f):
                i = fps(xyz, xyz.shape[1] // s.fator)
                nxyz = juntar(xyz, i[..., None])[:, :, 0]
                idx = knn(nxyz, xyz, s.k)
                g = torch.cat([juntar(xyz, idx) - nxyz[:, :, None], juntar(f, idx)], -1)
                return nxyz, s.mlp(g).max(2).values

        class Sobe(nn.Module):
            def __init__(s, ci, co):
                super().__init__(); s.a, s.b = MLP([ci, co]), MLP([co, co])

            def forward(s, xyz_d, f_d, xyz_s, f_s):
                return s.b(f_d) + interpolar(xyz_d, xyz_s, s.a(f_s))

        class PointTransformer(nn.Module):
            def __init__(s):
                super().__init__()
                s.ent = MLP([4, 32]); s.b1 = Bloco(32)
                s.d1 = Desce(32, 64); s.b2 = Bloco(64)
                s.d2 = Desce(64, 128); s.b3 = Bloco(128)
                s.u2 = Sobe(128, 64); s.b4 = Bloco(64)
                s.u1 = Sobe(64, 32); s.b5 = Bloco(32)
                s.saida = nn.Linear(32, 1)

            def forward(s, x):
                xyz = x[..., :3]
                f1 = s.b1(xyz, s.ent(x))
                x2, f2 = s.d1(xyz, f1); f2 = s.b2(x2, f2)
                x3, f3 = s.d2(x2, f2); f3 = s.b3(x3, f3)
                f2 = s.b4(x2, s.u2(x2, f2, x3, f3))
                f1 = s.b5(xyz, s.u1(xyz, f1, x2, f2))
                return s.saida(f1)[..., 0]
        return PointTransformer()
    raise ValueError(tipo)


# ---------------------------------------------------------------------------
# 3) TREINO (mesma supervisão para todos: pseudo-rótulo do ScalarFlow de treino)
# ---------------------------------------------------------------------------
N_PONTOS = int(os.environ.get('TCC_N_PONTOS', 4096))
_COORD = None


def coords_norm():
    global _COORD
    if _COORD is None:
        g = np.stack(np.meshgrid(*[np.linspace(-1, 1, n) for n in GRADE], indexing='ij'), -1)
        _COORD = g.reshape(-1, 3).astype(np.float32)
    return _COORD


def amostrar_pontos(dens, rot, rng, n=None):
    n = n or N_PONTOS
    d, r = dens.reshape(-1), rot.reshape(-1)
    cand = np.flatnonzero(d > LIMIAR_CAND)
    k = min(len(cand), n // 2)
    sel = np.concatenate([rng.choice(cand, k, replace=False) if k else np.empty(0, int),
                          rng.integers(0, d.size, n - k)])
    x = np.concatenate([coords_norm()[sel], d[sel, None].astype(np.float32)], 1)
    return x, r[sel].astype(np.float32)


def _dados_split(nome_split):
    X, Y = [], []
    for seq in SPLIT[nome_split]:
        if (DADOS / f'{seq}.npz').exists():
            d, r, _, _ = carregar_seq(seq)
            X.append(d); Y.append(r)
    if not X:
        raise RuntimeError(f'Nenhuma sequência de {nome_split} preparada.')
    return np.concatenate(X), np.concatenate(Y)


def _dice(p, y, eps=1.0):
    inter = (p * y).sum()
    return float((2 * inter + eps) / (p.sum() + y.sum() + eps))


def treinar(tipo):
    torch, nn, F = _torch()
    MODELOS.mkdir(parents=True, exist_ok=True)
    if (MODELOS / f'{tipo}.json').exists() and (MODELOS / f'{tipo}.pt').exists():
        log(f'{NOMES[tipo]}: já treinado. Pulando.'); return
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(0); rng = np.random.default_rng(0)
    Xtr, Ytr = _dados_split('treino'); Xva, Yva = _dados_split('validacao')
    volumetrico = tipo in MODELOS_VOL
    modelo = construir_modelo(tipo).to(dev)
    n_param = sum(p.numel() for p in modelo.parameters())
    opt = torch.optim.AdamW(modelo.parameters(), lr=1e-3, weight_decay=1e-4)
    usar_amp = dev == 'cuda' and volumetrico
    scaler = torch.amp.GradScaler('cuda', enabled=usar_amp)
    bce = nn.BCEWithLogitsLoss()
    lote = 4 if volumetrico else 8
    iters = int(os.environ.get('TCC_ITERS', 150))
    if dev == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    log(f'\n=== Treinando {NOMES[tipo]} | {n_param:,} parâmetros | {len(Xtr)} volumes de treino | '
        f'até {MINUTOS:.0f} min | {dev} ==='.replace(',', '.'))

    def lote_vol(X, Y, ids, aug):
        x = torch.from_numpy(X[ids].astype(np.float32))[:, None]
        y = torch.from_numpy(Y[ids].astype(np.float32))[:, None]
        if aug:
            if rng.random() < 0.5: x, y = x.flip(2), y.flip(2)
            if rng.random() < 0.5: x, y = x.flip(4), y.flip(4)
            x = (x * float(rng.uniform(0.8, 1.2)) + 0.03 * torch.randn_like(x)).clamp(0, 1)
        return x.to(dev), y.to(dev)

    def lote_pts(X, Y, ids, aug):
        xs, ys = zip(*[amostrar_pontos(X[i], Y[i], rng) for i in ids])
        x, y = torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys))
        if aug:
            if rng.random() < 0.5: x[..., 0] = -x[..., 0]
            if rng.random() < 0.5: x[..., 2] = -x[..., 2]
            x[..., 3] = (x[..., 3] * float(rng.uniform(0.8, 1.2)) + 0.03 * torch.randn_like(x[..., 3])).clamp(0, 1)
        return x.to(dev), y.to(dev)

    def perda(logit, y):
        p = torch.sigmoid(logit)
        dims = tuple(range(1, p.dim()))
        dice = 1 - (2 * (p * y).sum(dims) + 1) / (p.sum(dims) + y.sum(dims) + 1)
        return bce(logit, y) + dice.mean()

    val_ids = np.arange(0, len(Xva), 3)
    melhor, hist, t0, ep = -1.0, [], time.time(), 0
    while time.time() - t0 < MINUTOS * 60 and ep < int(os.environ.get('TCC_MAX_EPOCAS', 60)):
        ep += 1; modelo.train(); soma = 0.0
        for _ in range(iters):
            ids = rng.integers(0, len(Xtr), lote)
            x, y = (lote_vol if volumetrico else lote_pts)(Xtr, Ytr, ids, True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', enabled=usar_amp):
                logit = modelo(x)
            l = perda(logit.float(), y)
            scaler.scale(l).backward(); scaler.step(opt); scaler.update()
            soma += float(l)
            if time.time() - t0 > MINUTOS * 60:
                break
        modelo.eval(); dices = []
        with torch.no_grad():
            for j in range(0, len(val_ids), lote):
                ids = val_ids[j:j + lote]
                x, y = (lote_vol if volumetrico else lote_pts)(Xva, Yva, ids, False)
                with torch.autocast('cuda', enabled=usar_amp):
                    p = (torch.sigmoid(modelo(x).float()) > 0.5).float()
                dices += [_dice(p[i].cpu().numpy(), y[i].cpu().numpy()) for i in range(len(ids))]
        dv = float(np.mean(dices))
        hist.append({'epoca': ep, 'perda': soma / iters, 'dice_val': dv, 'min': (time.time() - t0) / 60})
        log(f'  época {ep}: perda {soma / iters:.4f} | Dice validação (vs pseudo-rótulo) {dv:.4f} | '
            f'{(time.time() - t0) / 60:.1f} min')
        if dv > melhor:
            melhor = dv; torch.save(modelo.state_dict(), MODELOS / f'{tipo}.pt')
    mem = torch.cuda.max_memory_allocated() / 1e9 if dev == 'cuda' else 0.0
    salvar_json(MODELOS / f'{tipo}.json', {'modelo': NOMES[tipo], 'parametros': n_param, 'epocas': ep,
                                           'melhor_dice_val': melhor, 'tempo_treino_min': (time.time() - t0) / 60,
                                           'memoria_gpu_gb': mem, 'historico': hist})
    log(f'{NOMES[tipo]}: concluído | melhor Dice validação {melhor:.4f} | {(time.time() - t0) / 60:.1f} min | '
        f'memória GPU {mem:.2f} GB')


def etapa_treinar(tipos):
    for t in tipos:
        try:
            treinar(t)
        except Exception:
            log(f'ERRO ao treinar {t}:\n{traceback.format_exc()}')
        finally:
            try:
                import torch; torch.cuda.empty_cache()
            except Exception:
                pass


# ---------------------------------------------------------------------------
# 4) INFERÊNCIA + MÉTRICAS SEM GROUND TRUTH
# ---------------------------------------------------------------------------
def prever(tipo, modelo, dens, dev):
    """Máscara 3D (T, 64,112,64) de um modelo. Só voxels candidatos (densidade > 0,02) podem ser fumaça."""
    torch, _, _ = _torch()
    T = dens.shape[0]; cand = dens.astype(np.float32) > LIMIAR_CAND
    saida = np.zeros(dens.shape, bool)
    t0 = time.time()
    with torch.no_grad():
        if tipo in MODELOS_VOL:
            for i in range(0, T, 4):
                x = torch.from_numpy(dens[i:i + 4].astype(np.float32))[:, None].to(dev)
                with torch.autocast('cuda', enabled=dev == 'cuda'):
                    p = torch.sigmoid(modelo(x).float())[:, 0].cpu().numpy() > 0.5
                saida[i:i + 4] = p
        else:
            rng = np.random.default_rng(0); C = coords_norm()
            for t in range(T):
                idx = np.flatnonzero(cand[t].reshape(-1))
                if len(idx) == 0:
                    continue
                idx = rng.permutation(idx)
                pred = np.zeros(len(idx), bool)
                for j in range(0, len(idx), N_PONTOS):
                    parte = idx[j:j + N_PONTOS]
                    pad = np.resize(parte, N_PONTOS)  # completa repetindo pontos
                    x = np.concatenate([C[pad], dens[t].reshape(-1)[pad, None].astype(np.float32)], 1)
                    lg = modelo(torch.from_numpy(x)[None].to(dev))[0].float().cpu().numpy()
                    pred[j:j + len(parte)] = lg[:len(parte)] > 0
                m = np.zeros(cand[t].size, bool); m[idx] = pred
                saida[t] = m.reshape(GRADE)
    ms = 1000 * (time.time() - t0) / T
    return saida & cand, ms


def _iou(a, b):
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else np.nan  # dois frames vazios: não conta


def _borda(m):
    return m & ~ndimage.binary_erosion(m)


def _hd95(a, b):
    ba, bb = _borda(a), _borda(b)
    if not ba.any() or not bb.any():
        return np.nan
    da = ndimage.distance_transform_edt(~bb)[ba]
    db = ndimage.distance_transform_edt(~ba)[bb]
    return float(max(np.percentile(da, 95), np.percentile(db, 95)))


def _advectar(m, v):
    """Leva a máscara do frame t para t+1 com o campo de velocidade (semi-Lagrangiano, para trás)."""
    g = np.indices(m.shape).astype(np.float32)
    orig = g - np.moveaxis(v.astype(np.float32), -1, 0)
    return ndimage.map_coordinates(m.astype(np.float32), orig, order=1, mode='constant') > 0.5


def metricas(M, rot, vel=None):
    """Métricas sem GT (estabilidade temporal, forma e física) + extras. M e rot: (T, X, Y, Z) bool."""
    T = M.shape[0]
    vol = M.reshape(T, -1).sum(1).astype(float)
    iou_t, var_v, desl, hd, flick, fis, comp = [], [], [], [], [], [], []
    cent = [np.array(ndimage.center_of_mass(M[t])) if vol[t] > 0 else None for t in range(T)]
    for t in range(T):
        lab, n = ndimage.label(M[t]); comp.append(n)
        if t + 1 < T:
            iou_t.append(_iou(M[t], M[t + 1]))
            if vol[t] > 0: var_v.append(abs(vol[t + 1] - vol[t]) / vol[t])
            if cent[t] is not None and cent[t + 1] is not None:
                desl.append(float(np.linalg.norm(cent[t + 1] - cent[t])))
            if t % 2 == 0: hd.append(_hd95(M[t], M[t + 1]))
            if vel is not None and t < len(vel):
                fis.append(_iou(_advectar(M[t], vel[t]), M[t + 1]))
        if 0 < t < T - 1:
            troca = (M[t - 1] == M[t + 1]) & (M[t] != M[t - 1])
            flick.append(float(troca.sum() / max(vol[t], 1)))
    fina = rot & ~ndimage.binary_opening(rot, structure=np.ones((1, 3, 3, 3), bool))
    return {
        'consistencia_temporal_iou': float(np.nanmean(iou_t)) if np.isfinite(iou_t).any() else np.nan,
        'variacao_volume': float(np.mean(var_v)) if var_v else np.nan,
        'deslocamento_centroide': float(np.mean(desl)) if desl else np.nan,
        'estabilidade_borda_hd95': float(np.nanmean(hd)) if hd else np.nan,
        'flicker': float(np.mean(flick)) if flick else np.nan,
        'consistencia_fisica_iou': float(np.nanmean(fis)) if np.isfinite(fis).any() else np.nan,
        'frames_vazios_pct': float(100 * (vol == 0).mean()),
        'fracao_volume_pct': float(100 * M.mean()),
        'componentes': float(np.mean(comp)),
        'retencao_estrutura_fina': float((M & fina).sum() / max(fina.sum(), 1)),
        'dice_vs_pseudo': _dice(M.astype(np.float32), rot.astype(np.float32)),
    }


def etapa_avaliar():
    torch, _, _ = _torch()
    import pandas as pd
    RESULT.mkdir(parents=True, exist_ok=True)
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    seqs_teste = [s for s in SPLIT['teste'] if (DADOS / f'{s}.npz').exists()]
    seqs_teste += [f for f in FN_FRAMES if (DADOS / f'{f}.npz').exists()]
    csv = RESULT / 'metricas_por_sequencia.csv'
    linhas = pd.read_csv(csv).to_dict('records') if csv.exists() else []
    feitos = {(l['modelo'], l['sequencia']) for l in linhas}
    exemplos = {}
    for tipo in ['limiar'] + MODELOS_VOL + MODELOS_PTS:
        modelo, info = None, {}
        if tipo != 'limiar':
            if not (MODELOS / f'{tipo}.pt').exists():
                log(f'{NOMES[tipo]}: sem modelo treinado (pulado)'); continue
            try:
                modelo = construir_modelo(tipo).to(dev)
                modelo.load_state_dict(torch.load(MODELOS / f'{tipo}.pt', map_location=dev)); modelo.eval()
                info = json.loads((MODELOS / f'{tipo}.json').read_text())
            except Exception:
                log(f'ERRO ao carregar {tipo}:\n{traceback.format_exc()}'); continue
        for seq in seqs_teste:
            if (NOMES[tipo], seq) in feitos:
                continue
            try:
                t0 = time.time()
                dens, rot, meta, vel = carregar_seq(seq, com_vel=True)
                if tipo == 'limiar':
                    M, ms = rot, 0.0
                else:
                    M, ms = prever(tipo, modelo, dens, dev)
                met = metricas(M, rot, vel)
                linha = {'modelo': NOMES[tipo], 'dominio': meta['dominio'], 'sequencia': seq, **met,
                         'inferencia_ms_por_frame': ms, 'parametros': info.get('parametros', 0),
                         'tempo_treino_min': info.get('tempo_treino_min', 0.0),
                         'memoria_gpu_gb': info.get('memoria_gpu_gb', 0.0)}
                linhas = [l for l in linhas if not (l['modelo'] == NOMES[tipo] and l['sequencia'] == seq)] + [linha]
                pd.DataFrame(linhas).to_csv(csv, index=False)
                t_meio = M.shape[0] // 2
                exemplos[(NOMES[tipo], seq)] = (M[t_meio], dens[t_meio].astype(np.float32))
                log(f'  {NOMES[tipo]} | {seq}: IoU temporal {met["consistencia_temporal_iou"]:.3f} | '
                    f'flicker {met["flicker"]:.3f} | Dice vs pseudo {met["dice_vs_pseudo"]:.3f} | '
                    f'{time.time() - t0:.0f} s')
            except Exception:
                log(f'ERRO avaliando {tipo} em {seq}:\n{traceback.format_exc()}')
        del modelo
        try: torch.cuda.empty_cache()
        except Exception: pass
    if linhas:
        gerar_tabelas_e_figuras(pd.DataFrame(linhas), exemplos)


def gerar_tabelas_e_figuras(df, exemplos):
    import pandas as pd
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    FIGS.mkdir(parents=True, exist_ok=True)
    df = df.copy()
    df['grupo'] = df['dominio'].replace({'ScalarFlow': 'ScalarFlow (teste)'})
    cols = ['consistencia_temporal_iou', 'variacao_volume', 'deslocamento_centroide', 'estabilidade_borda_hd95',
            'flicker', 'consistencia_fisica_iou', 'frames_vazios_pct', 'fracao_volume_pct', 'componentes', 'retencao_estrutura_fina',
            'dice_vs_pseudo', 'inferencia_ms_por_frame']
    media = df.groupby(['modelo', 'grupo'])[cols].mean().round(4)
    desvio = df.groupby(['modelo', 'grupo'])[cols].std().round(4)
    media.to_csv(RESULT / 'tabela2_metricas_sem_gt_media.csv')
    desvio.to_csv(RESULT / 'tabela2_metricas_sem_gt_desvio.csv')
    custo = df.groupby('modelo')[['parametros', 'tempo_treino_min', 'memoria_gpu_gb', 'inferencia_ms_por_frame']].mean()
    custo.to_csv(RESULT / 'tabela3_custo_computacional.csv')
    log('\nTABELA 2 (média por modelo e domínio) salva em', RESULT / 'tabela2_metricas_sem_gt_media.csv')
    try:
        from IPython.display import display
        display(media); display(custo.round(2))
    except Exception:
        log(media.to_string())

    # Figura: métricas principais (barras por domínio)
    ordem_m = [NOMES[k] for k in ['limiar'] + MODELOS_VOL + MODELOS_PTS if NOMES[k] in set(df.modelo)]
    grupos = list(dict.fromkeys(df['grupo']))
    principais = [('consistencia_temporal_iou', 'Consistência temporal (IoU) ↑'), ('flicker', 'Flicker ↓'),
                  ('estabilidade_borda_hd95', 'Estabilidade da borda (HD95, voxels) ↓'),
                  ('retencao_estrutura_fina', 'Retenção de estrutura fina ↑')]
    fig, ax = plt.subplots(1, 4, figsize=(22, 4.8), facecolor='white')
    w = 0.8 / max(len(ordem_m), 1)
    for k, (c, tit) in enumerate(principais):
        for j, m in enumerate(ordem_m):
            vals = [media.loc[(m, g), c] if (m, g) in media.index else np.nan for g in grupos]
            ax[k].bar(np.arange(len(grupos)) + (j - (len(ordem_m) - 1) / 2) * w, vals, w, label=m)
        ax[k].set_xticks(range(len(grupos))); ax[k].set_xticklabels(grupos, rotation=20, ha='right', fontsize=8)
        ax[k].set_title(tit); ax[k].grid(axis='y', alpha=0.3)
    ax[0].legend(fontsize=8)
    fig.suptitle('Segmentação 3D sem ground truth: métricas por modelo e domínio', fontsize=13)
    fig.tight_layout(); fig.savefig(FIGS / 'segmentacao_metricas_sem_gt.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

    # Figura: máscaras projetadas (frame do meio) por modelo x sequência
    seqs = list(dict.fromkeys(s for (_, s) in exemplos))
    if seqs and ordem_m:
        fig, ax = plt.subplots(len(ordem_m) + 1, len(seqs), figsize=(2.6 * len(seqs), 3.6 * (len(ordem_m) + 1)),
                               facecolor='white', squeeze=False)
        for c, s in enumerate(seqs):
            dens = next(v[1] for (m, ss), v in exemplos.items() if ss == s)
            ax[0, c].imshow(dens.sum(2).T, origin='lower', cmap='inferno', interpolation='bilinear')
            ax[0, c].set_title(s, fontsize=9)
            for r, m in enumerate(ordem_m, start=1):
                if (m, s) in exemplos:
                    ax[r, c].imshow(exemplos[(m, s)][0].sum(2).T > 0, origin='lower', cmap='gray')
        for r, rot_ in enumerate(['Densidade'] + ordem_m):
            ax[r, 0].set_ylabel(rot_, fontsize=9)
        for a in ax.ravel():
            a.set_xticks([]); a.set_yticks([])
        fig.suptitle('Máscaras de fumaça (projeção de frente, frame do meio) por modelo', fontsize=13)
        fig.tight_layout(); fig.savefig(FIGS / 'segmentacao_mascaras_por_modelo.png', dpi=130, bbox_inches='tight')
        plt.close(fig)
    log('Figuras salvas em', FIGS)


def executar(etapa, *args):
    """Roda uma etapa sem nunca levantar erro (para a fila de células do Colab não ser cancelada)."""
    t0 = time.time()
    try:
        etapa(*args)
        log(f'OK: {etapa.__name__} em {(time.time() - t0) / 60:.1f} min')
    except Exception:
        log(f'ERRO em {etapa.__name__}:\n{traceback.format_exc()}')
