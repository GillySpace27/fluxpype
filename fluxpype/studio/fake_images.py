"""Procedural placeholder images for the demo (stdlib only: zlib + struct)."""

import math
import random
import struct
import zlib


def _png(width, height, rgb):
    raw = bytearray()
    for y in range(height):
        raw.append(0)
        raw += rgb[y * width * 3 : (y + 1) * width * 3]

    def chunk(tag, data):
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw), 6))
        + chunk(b"IEND", b"")
    )


class Canvas:
    def __init__(self, w, h, bg=(8, 10, 16)):
        self.w, self.h = w, h
        self.px = bytearray(bytes(bg) * (w * h))

    def add(self, x, y, c, k=1.0):
        x, y = int(x), int(y)
        if 0 <= x < self.w and 0 <= y < self.h:
            i = (y * self.w + x) * 3
            for j in range(3):
                self.px[i + j] = min(255, self.px[i + j] + int(c[j] * k))

    def png(self):
        return _png(self.w, self.h, self.px)


def _disk(cv, cx, cy, r, inner, outer):
    for y in range(int(cy - r), int(cy + r) + 1):
        for x in range(int(cx - r), int(cx + r) + 1):
            d = math.hypot(x - cx, y - cy) / r
            if d <= 1:
                t = d**2
                c = tuple(int(inner[k] * (1 - t) + outer[k] * t) for k in range(3))
                cv.add(x, y, c)


def world(rng, small):
    w, h = (96, 64) if small else (720, 480)
    cv = Canvas(w, h)
    cx, cy, R = w / 2, h / 2, h * 0.27
    for _ in range(260 if not small else 40):
        phi = rng.uniform(0, 2 * math.pi)
        lat0 = rng.uniform(-1.35, 1.35)
        reach = rng.uniform(1.15, 2.6)
        tilt = rng.uniform(-0.5, 0.5)
        hue = (255, 150, 60) if rng.random() < 0.55 else (110, 170, 255)
        for i in range(260 if not small else 14):
            s = i / (259 if not small else 13)
            lam = lat0 + (s - 0.5) * 2 * math.acos(max(-1, min(1, 1 / reach)))
            r = R * (1 + (reach - 1) * math.sin(math.pi * s) ** 1.1)
            x = cx + r * math.cos(lam + tilt) * math.cos(phi) * 1.0
            y = cy - r * math.sin(lam + tilt)
            cv.add(x, y, hue, 0.55)
    _disk(cv, cx, cy, R, (96, 52, 22), (40, 18, 8))
    return cv.png()


def fieldmap(rng, small):
    w, h = (96, 48) if small else (480, 240)
    cv = Canvas(w, h)
    blobs = [(rng.uniform(0, w), rng.uniform(0.15 * h, 0.85 * h), rng.uniform(0.02, 0.07) * w, rng.choice((-1, 1)) * rng.uniform(0.5, 1)) for _ in range(16)]
    for y in range(h):
        for x in range(w):
            v = 0.0
            for bx, by, s, a in blobs:
                v += a * math.exp(-((x - bx) ** 2 + (y - by) ** 2) / (2 * s * s))
            v = max(-1, min(1, v))
            if v > 0:
                c = (40 + int(215 * v), 30 + int(70 * v), 30)
            else:
                c = (30, 50 + int(70 * -v), 60 + int(195 * -v))
            cv.add(x, y, c)
    return cv.png()


def wind(rng, small):
    w, h = (96, 56) if small else (720, 420)
    cv = Canvas(w, h, bg=(10, 12, 20))
    for gx in range(0, w, max(w // 8, 1)):
        for y in range(h):
            cv.add(gx, y, (22, 26, 38))
    for gy in range(0, h, max(h // 5, 1)):
        for x in range(w):
            cv.add(x, gy, (22, 26, 38))
    colors = [(255, 170, 60), (90, 200, 160), (140, 150, 255), (255, 110, 140)]
    for n, col in enumerate(colors):
        base = rng.uniform(0.25, 0.5)
        amp = rng.uniform(0.08, 0.2)
        ph = rng.uniform(0, 6)
        for x in range(w):
            t = x / w
            v = base + 0.55 * (1 - math.exp(-5 * t)) * (0.6 + 0.4 * n / 4) + amp * math.sin(9 * t + ph) * t
            y = h - 6 - v * (h - 14)
            for dy in range(2 if not small else 1):
                cv.add(x, y + dy, col, 0.95)
    return cv.png()


def light(rng, small):
    n = 72 if small else 460
    cv = Canvas(n, n, bg=(0, 0, 0))
    c = n / 2
    seeds = [(rng.uniform(0, 2 * math.pi), rng.uniform(0.2, 0.8)) for _ in range(9)]
    for y in range(n):
        for x in range(n):
            dx, dy = (x - c) / c, (y - c) / c
            r = math.hypot(dx, dy)
            if r < 0.01:
                continue
            ang = math.atan2(dy, dx)
            streak = sum(a * math.exp(-8 * (1 - math.cos(ang - s)) * 4) for s, a in seeds)
            v = (0.9 / (r**2.6 + 0.04)) * (0.35 + 0.65 * streak) * (1 - math.exp(-r * 5))
            v = min(1.0, v * 0.2) ** 1.15
            if r < 0.27:
                cv.add(x, y, (0, 0, 0))
            else:
                cv.add(x, y, (int(255 * v), int(215 * v * v ** 0.15), int(170 * v ** 1.4)))
    return cv.png()


KINDS = {"world": world, "fieldmap": fieldmap, "wind": wind, "light": light}


def render(kind, seed="0", small=False) -> bytes:
    return KINDS[kind](random.Random(seed), small)
