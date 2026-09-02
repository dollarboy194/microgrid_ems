"""Faithful Python port of dataviz/scripts/validate_palette.js (categorical checks).

Same thresholds, same Machado-2009 matrices, same CIE76 dE on adjacent pairs.
Used because this machine has no Node runtime.
"""
import math
import sys

BAND = {"light": (0.43, 0.77), "dark": (0.48, 0.67)}
CHROMA_FLOOR = 0.10
CVD_TARGET, CVD_FLOOR = 12.0, 8.0
CONTRAST_MIN = 3.0
DEFAULT_SURFACE = {"light": "#fcfcfb", "dark": "#1a1a19"}

MACHADO = {
    "protan": [[0.152286, 1.052583, -0.204868],
               [0.114503, 0.786281, 0.099216],
               [-0.003882, -0.048116, 1.051998]],
    "deutan": [[0.367322, 0.860646, -0.227968],
               [0.280085, 0.672501, 0.047413],
               [-0.011820, 0.042940, 0.968881]],
    "tritan": [[1.255528, -0.076749, -0.178779],
               [-0.078411, 0.930809, 0.147602],
               [0.004733, 0.691367, 0.303900]],
}


def hex2srgb(h):
    h = h.strip().lstrip("#")
    return [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]


def s2lin(c):
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def lin(h):
    return [s2lin(c) for c in hex2srgb(h)]


def rel_lum(h):
    r, g, b = lin(h)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def contrast(a, b):
    hi, lo = sorted([rel_lum(a), rel_lum(b)], reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def oklab(h):
    r, g, b = lin(h)
    l = (0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b) ** (1 / 3)
    m = (0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b) ** (1 / 3)
    s = (0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b) ** (1 / 3)
    return (0.2104542553 * l + 0.7936177850 * m - 0.0040720468 * s,
            1.9779984951 * l - 2.4285922050 * m + 0.4505937099 * s,
            0.0259040371 * l + 0.7827717662 * m - 0.8086757660 * s)


def oklch(h):
    L, a, b = oklab(h)
    return L, math.hypot(a, b)


def lin2lab(r, g, b):
    X = 0.4124564 * r + 0.3575761 * g + 0.1804375 * b
    Y = 0.2126729 * r + 0.7151522 * g + 0.0721750 * b
    Z = 0.0193339 * r + 0.1191920 * g + 0.9503041 * b
    f = lambda t: t ** (1 / 3) if t > 0.008856 else 7.787 * t + 16 / 116
    fx, fy, fz = f(X / 0.95047), f(Y / 1.0), f(Z / 1.08883)
    return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)]


def simulate(h, kind):
    r, g, b = lin(h)
    M = MACHADO[kind]
    clamp = lambda c: max(0.0, min(1.0, c))
    return [clamp(M[i][0] * r + M[i][1] * g + M[i][2] * b) for i in range(3)]


def delta_e(h1, h2, kind=None):
    a = lin2lab(*(simulate(h1, kind) if kind else lin(h1)))
    b = lin2lab(*(simulate(h2, kind) if kind else lin(h2)))
    return math.dist(a, b)


def validate(palette, mode="light", surface=None, pairs="adjacent"):
    surface = surface or DEFAULT_SURFACE[mode]
    lo, hi = BAND[mode]
    ok = True
    print(f"\nPalette ({mode}, surface {surface}, categorical): {len(palette)} slots")

    offband = [(c, round(oklch(c)[0], 3)) for c in palette if not (lo <= oklch(c)[0] <= hi)]
    ok &= not offband
    print(f"  [{'PASS' if not offband else 'FAIL'}] Lightness band       "
          f"{offband if offband else f'all inside L {lo}-{hi}'}")

    lowc = [(c, round(oklch(c)[1], 3)) for c in palette if oklch(c)[1] < CHROMA_FLOOR]
    ok &= not lowc
    print(f"  [{'PASS' if not lowc else 'FAIL'}] Chroma floor         "
          f"{lowc if lowc else f'all >= {CHROMA_FLOOR}'}")

    n = len(palette)
    if pairs == "all":
        pairlist = [(i, j) for i in range(n) for j in range(i + 1, n)]
    else:
        pairlist = [(i, i + 1) for i in range(n - 1)]

    worst = None
    for kind in ("protan", "deutan"):
        for i, j in pairlist:
            d = delta_e(palette[i], palette[j], kind)
            if worst is None or d < worst[0]:
                worst = (d, kind, palette[i], palette[j])
    tri = min(delta_e(palette[i], palette[j], "tritan") for i, j in pairlist) if pairlist else 99
    nor = min(delta_e(palette[i], palette[j]) for i, j in pairlist) if pairlist else 99
    wd = worst[0] if worst else 99
    state = "PASS" if wd >= CVD_TARGET else ("WARN" if wd >= CVD_FLOOR else "FAIL")
    ok &= state != "FAIL"
    print(f"  [{state}] CVD separation       worst {pairs} {worst[3]}<->{worst[2]} "
          f"dE {wd:.1f} ({worst[1]}) - tritan {tri:.1f} - normal {nor:.1f}")

    low = [(c, round(contrast(c, surface), 2)) for c in palette if contrast(c, surface) < CONTRAST_MIN]
    print(f"  [{'PASS' if not low else 'WARN'}] Contrast vs surface  "
          f"{low if low else f'all >= {CONTRAST_MIN}:1'}")
    if low:
        print("         -> relief required: visible direct labels or table view")

    print(f"\n  => {'ALL CHECKS PASS' if ok else 'FAILED'}\n")
    return ok


if __name__ == "__main__":
    pal = [s.strip() for s in sys.argv[1].split(",") if s.strip()]
    mode = sys.argv[2] if len(sys.argv) > 2 else "light"
    pairs = sys.argv[3] if len(sys.argv) > 3 else "adjacent"
    sys.exit(0 if validate(pal, mode=mode, pairs=pairs) else 1)
