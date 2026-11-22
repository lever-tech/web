(() => {
  'use strict';

  const canvas = document.getElementById('field');
  if (!canvas) return;

  const LADDER_HEX = [
    '#040404', '#0e0e0e', '#1e1e1e', '#343434',
    '#545454', '#7e7e7e', '#b4b4b4', '#f2f2f2',
  ];

  const BAYER = [
     0,  8,  2, 10,
    12,  4, 14,  6,
     3, 11,  1,  9,
    15,  7, 13,  5,
  ];

  const TRAIL_LEN = 12;
  const RIPPLE_LEN = 4;
  const RIPPLE_LIFE = 1.4;
  // 4 slots x 0.35s floor = 1.4s life: a new ring never lands while a slot is
  // still alive, so rapid clicking can't stack rings into a spam blob.
  const RIPPLE_COOLDOWN = RIPPLE_LIFE / RIPPLE_LEN;
  const FPS = 30;
  const FRAME_MS = 1000 / FPS;

  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

  function hexToRgb01(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
  }

  function hexToBytes(hex) {
    const n = parseInt(hex.slice(1), 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }

  const ladderFloats = new Float32Array(24);
  const ladderBytes = LADDER_HEX.map(hexToBytes);
  LADDER_HEX.forEach((h, i) => {
    const [r, g, b] = hexToRgb01(h);
    ladderFloats[i * 3] = r;
    ladderFloats[i * 3 + 1] = g;
    ladderFloats[i * 3 + 2] = b;
  });

  // Grid state (locked integer device-pixel cells)
  let vw = 0, vh = 0, dpr = 1, cellCss = 3, cols = 1, rows = 1;

  // Pointer trail ring buffer: [xCells, yCells, strength] (movement-only, no stationary stacking)
  const trail = new Float32Array(TRAIL_LEN * 3);
  let trailHead = 0;
  let ptrX = -9999, ptrY = -9999, ptrActive = 0;
  let easeX = -9999, easeY = -9999, easeA = 0;
  let lastDepositX = -9999, lastDepositY = -9999;

  // Click ripple ring buffer: [xCells, yCells, ageSec, active]
  const ripples = new Float32Array(RIPPLE_LEN * 4);
  let rippleHead = 0;
  let lastRippleAt = -Infinity;

  // [data-dim] rect in cell coords: [x0, y0, x1, y1]
  const dimRect = new Float32Array(4);
  const dimEl = document.querySelector('[data-dim]');

  function resize() {
    vw = window.innerWidth;
    vh = window.innerHeight;
    dpr = Math.min(window.devicePixelRatio || 1, 3);
    // Lock cell size to the exact integer pixel scale of the 83x7 lever-flake SVG
    cellCss = Math.max(2, Math.min(6, Math.floor((vw * 0.5) / 83)));
    const cellDev = Math.max(1, Math.round(cellCss * dpr));
    cols = Math.ceil((vw * dpr) / cellDev);
    rows = Math.ceil((vh * dpr) / cellDev);
    canvas.width = cols;
    canvas.height = rows;
    canvas.style.width = `${(cols * cellDev) / dpr}px`;
    canvas.style.height = `${(rows * cellDev) / dpr}px`;

    // Snap the 83x7 mark onto exact integer cell coordinates
    const artCol = Math.floor((cols - 83) / 2);
    const artRow = Math.floor((rows - 7) / 2);
    const root = document.documentElement.style;
    root.setProperty('--cell', `${cellDev / dpr}px`);
    root.setProperty('--art-x', `${(artCol * cellDev) / dpr}px`);
    root.setProperty('--art-y', `${(artRow * cellDev) / dpr}px`);
    root.setProperty('--art-t', 'none');
  }

  function readDimRect() {
    if (!dimEl) {
      dimRect.fill(-1);
      return;
    }
    const r = dimEl.getBoundingClientRect();
    const sx = cols / vw;
    const sy = rows / vh;
    dimRect[0] = r.left * sx;
    dimRect[1] = r.top * sy;
    dimRect[2] = r.right * sx;
    dimRect[3] = r.bottom * sy;
  }

  // --- WebGL2 renderer ------------------------------------------------------
  function createGL() {
    const gl = canvas.getContext('webgl2', {
      antialias: false,
      depth: false,
      stencil: false,
      powerPreference: 'low-power',
    });
    if (!gl) return null;

    const vsSrc = `#version 300 es
      void main() {
        vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
        gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
      }`;

    const fsSrc = `#version 300 es
      precision highp float;
      precision highp int;
      out vec4 outColor;

      uniform vec2 uRes;
      uniform float uTime;
      uniform float uIntro;
      uniform vec3 uLadder[8];
      uniform vec4 uDim;
      uniform vec3 uPtr;
      uniform vec3 uTrail[${TRAIL_LEN}];
      uniform vec4 uRipples[${RIPPLE_LEN}];

      const int BAYER[16] = int[16](
         0,  8,  2, 10,
        12,  4, 14,  6,
         3, 11,  1,  9,
        15,  7, 13,  5
      );

      uint pcg2(uvec2 v) {
        v = v * 1664525u + 1013904223u;
        v.x += v.y * 1664525u; v.y += v.x * 1664525u;
        v ^= v >> 16u;
        v.x += v.y * 1664525u; v.y += v.x * 1664525u;
        v ^= v >> 16u;
        return v.x ^ v.y;
      }

      float rnd(ivec2 p) {
        return float(pcg2(uvec2(p + 32768)) & 0x00ffffffu) / 16777216.0;
      }

      vec2 grad(ivec2 p) {
        float a = rnd(p) * 6.2831853;
        return vec2(cos(a), sin(a));
      }

      float gnoise(vec2 p) {
        ivec2 i = ivec2(floor(p));
        vec2 f = fract(p);
        vec2 u = f * f * f * (f * (f * 6.0 - 15.0) + 10.0);
        float a = dot(grad(i), f);
        float b = dot(grad(i + ivec2(1, 0)), f - vec2(1.0, 0.0));
        float c = dot(grad(i + ivec2(0, 1)), f - vec2(0.0, 1.0));
        float d = dot(grad(i + ivec2(1, 1)), f - vec2(1.0, 1.0));
        return mix(mix(a, b, u.x), mix(c, d, u.x), u.y) * 1.4;
      }

      const mat2 ROT0 = mat2(0.8746, 0.4848, -0.4848, 0.8746); // ~29 deg: breaks any vertical/horizontal axis alignment
      const mat2 OCT  = mat2(1.62, 1.18, -1.18, 1.62);

      float fbm(vec2 p) {
        float v = 0.0;
        float a = 0.5;
        for (int i = 0; i < 4; i++) {
          v += a * gnoise(p);
          p = OCT * p + vec2(17.31, 9.17);
          a *= 0.5;
        }
        return v;
      }

      void main() {
        ivec2 ic = ivec2(gl_FragCoord.xy);
        vec2 cell = vec2(float(ic.x), uRes.y - 1.0 - float(ic.y));

        // World-space cell coordinates rotated off-axis so no grid seams align vertically/horizontally
        vec2 p = ROT0 * ((cell - uRes * 0.5) / 72.0);

        // Steady subtle presence at resting pointer (no fisheye warp)
        vec2 push = vec2(0.0);
        float wake = 0.0;
        if (uPtr.z > 0.002) {
          vec2 dp = cell - uPtr.xy;
          float gp = exp(-dot(dp, dp) * 0.00075) * uPtr.z;
          wake += gp * 0.045;
        }

        // Whisper-light flow deflection + shimmer from actual pointer movement
        for (int i = 0; i < ${TRAIL_LEN}; i++) {
          vec3 tr = uTrail[i];
          if (tr.z > 0.002) {
            vec2 d = cell - tr.xy;
            float d2 = dot(d, d);
            float g = exp(-d2 * 0.0009) * tr.z;
            wake += g * 0.03;
            vec2 swirl = vec2(-d.y, d.x);
            push += (ROT0 * swirl) * (g * 0.00022);
          }
        }

        // Small, subtle click ripple rings
        float rip = 0.0;
        for (int i = 0; i < ${RIPPLE_LEN}; i++) {
          vec4 rp = uRipples[i];
          if (rp.w > 0.5) {
            float dist = length(cell - rp.xy);
            float radius = rp.z * 58.0;
            float ring = exp(-pow((dist - radius) * 0.20, 2.0));
            float fade = exp(-rp.z * 2.8);
            rip += ring * fade * 0.09;
          }
        }

        float t = uTime * 0.10;
        vec2 q = p - push;

        // Broad aperiodic macro-field for asymmetric obsidian voids vs luminous zones
        float macro = fbm(q * 0.34 + vec2(5.2 + t * 0.06, -3.7 - t * 0.05));

        // Double domain-warping (flowing silk / marble / topographic ink currents)
        vec2 w1 = vec2(
          fbm(q * 0.72 + vec2(t * 0.24, -t * 0.17)),
          fbm(q * 0.72 + vec2(5.2 - t * 0.15, 1.3 + t * 0.21))
        );
        vec2 w2 = vec2(
          fbm(q * 0.95 + w1 * 1.35 + vec2(1.7 - t * 0.12, 9.2 + t * 0.14)),
          fbm(q * 0.95 + w1 * 1.35 + vec2(8.3 + t * 0.16, 2.8 - t * 0.11))
        );

        vec2 pw = q * 0.85 + w2 * 1.05;
        float n = fbm(pw);

        // Directional rim-lighting across the sculpted currents
        float nLight = fbm(pw + vec2(-0.11, -0.13));
        float shade = clamp((n - nLight) * 3.4, -1.0, 1.0);

        // Dramatic obsidian voids + sweeping bright crests
        float cover = -0.02 + macro * 0.36;
        float form = smoothstep(cover - 0.10, cover + 0.48, n);

        // Fine luminous contour lines along the sculpted ridges
        float band = abs(fract((n + length(w2) * 0.35) * 2.2) - 0.5) * 2.0;
        float contour = smoothstep(0.86, 0.985, band) * form * 0.22;

        float lum = 0.008
          + form * (0.24 + 0.28 * max(0.0, shade) - 0.10 * max(0.0, -shade))
          + contour
          + wake * (0.6 + form * 1.2)
          + rip;

        // Organic dithered unfold from bottom-left corner on start (valid smoothstep order)
        if (uIntro < 0.999) {
          float span = max(uRes.x, uRes.y);
          vec2 fromCorner = vec2(cell.x, uRes.y - 1.0 - cell.y) / span;
          float wave = length(fromCorner) - (n * 0.24 + macro * 0.18);
          float edge = mix(-0.35, 1.65, uIntro);
          float reveal = 1.0 - smoothstep(edge - 0.42, edge, wave);
          lum *= reveal;
        }

        // Legibility by subtraction behind [data-dim] mark
        if (uDim.z > uDim.x) {
          vec2 clamped = clamp(cell, uDim.xy, uDim.zw);
          vec2 d = cell - clamped;
          float dim = exp(-dot(d, d) * 0.00075);
          lum *= 1.0 - dim * 0.78;
        }

        // Ordered Bayer 4x4 quantization onto 8-step monochrome ladder
        float th = (float(BAYER[((ic.y & 3) << 2) | (ic.x & 3)]) + 0.5) / 16.0;
        float L = clamp(lum, 0.0, 1.0) * 7.0;
        float lo = floor(L);
        int idx = min(7, int(lo) + ((L - lo) > th ? 1 : 0));

        outColor = vec4(uLadder[idx], 1.0);
      }`;

    function compile(type, src) {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        gl.deleteShader(s);
        return null;
      }
      return s;
    }

    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) return null;

    const prog = gl.createProgram();
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) return null;

    gl.useProgram(prog);
    const loc = {
      res: gl.getUniformLocation(prog, 'uRes'),
      time: gl.getUniformLocation(prog, 'uTime'),
      intro: gl.getUniformLocation(prog, 'uIntro'),
      ladder: gl.getUniformLocation(prog, 'uLadder'),
      dim: gl.getUniformLocation(prog, 'uDim'),
      ptr: gl.getUniformLocation(prog, 'uPtr'),
      trail: gl.getUniformLocation(prog, 'uTrail'),
      ripples: gl.getUniformLocation(prog, 'uRipples'),
    };
    gl.uniform3fv(loc.ladder, ladderFloats);

    return {
      draw(timeSec, intro) {
        gl.viewport(0, 0, cols, rows);
        gl.uniform2f(loc.res, cols, rows);
        gl.uniform1f(loc.time, timeSec);
        gl.uniform1f(loc.intro, intro);
        gl.uniform4fv(loc.dim, dimRect);
        gl.uniform3f(loc.ptr, easeX, easeY, easeA);
        gl.uniform3fv(loc.trail, trail);
        gl.uniform4fv(loc.ripples, ripples);
        gl.drawArrays(gl.TRIANGLES, 0, 3);
      },
    };
  }

  // --- Minimal Canvas 2D fallback -------------------------------------------
  function create2D() {
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;
    let img = null;

    return {
      draw(timeSec) {
        if (!img || img.width !== cols || img.height !== rows) {
          img = ctx.createImageData(cols, rows);
        }
        const d = img.data;
        const t = timeSec * 0.35;
        for (let y = 0; y < rows; y++) {
          const by = (y & 3) << 2;
          for (let x = 0; x < cols; x++) {
            const v =
              0.14 +
              0.14 * Math.sin(x * 0.025 + t) * Math.cos(y * 0.025 - t * 0.7) +
              0.08 * Math.sin((x + y) * 0.018 - t * 0.5);
            const th = (BAYER[by | (x & 3)] + 0.5) / 16;
            const L = Math.max(0, Math.min(1, v)) * 7;
            const lo = Math.floor(L);
            const idx = Math.min(7, lo + (L - lo > th ? 1 : 0));
            const [r, g, b] = ladderBytes[idx];
            const o = (y * cols + x) * 4;
            d[o] = r;
            d[o + 1] = g;
            d[o + 2] = b;
            d[o + 3] = 255;
          }
        }
        ctx.putImageData(img, 0, 0);
      },
    };
  }

  // --- Pointer & ripple input -----------------------------------------------
  if (!reducedMotion) {
    window.addEventListener('pointermove', (e) => {
      ptrX = (e.clientX / vw) * cols;
      ptrY = (e.clientY / vh) * rows;
      ptrActive = 1;
    }, { passive: true });

    window.addEventListener('pointerleave', () => {
      ptrActive = 0;
    }, { passive: true });

    window.addEventListener('pointerdown', (e) => {
      const now = performance.now() * 0.001;
      if (now - lastRippleAt < RIPPLE_COOLDOWN) return;
      lastRippleAt = now;

      const rx = (e.clientX / vw) * cols;
      const ry = (e.clientY / vh) * rows;
      const o = rippleHead * 4;
      ripples[o] = rx;
      ripples[o + 1] = ry;
      ripples[o + 2] = 0;
      ripples[o + 3] = 1;
      rippleHead = (rippleHead + 1) % RIPPLE_LEN;
    }, { passive: true });
  }

  // --- Main loop (single rAF, 30 fps cap) -----------------------------------
  resize();
  const renderer = createGL() || create2D();
  if (!renderer) return;

  let lastDraw = 0;
  let simTime = 12.0; // start on an interesting wave phase
  let introAge = reducedMotion ? 1.0 : 0.0;
  const INTRO_DUR = 1.1;

  function introEase() {
    if (reducedMotion || introAge >= 1) return 1;
    return 1 - Math.pow(1 - introAge, 3);
  }

  function tick(now) {
    if (now - lastDraw >= FRAME_MS) {
      const dt = Math.min(0.1, (now - lastDraw) / 1000 || 0.033);
      lastDraw = now;

      readDimRect();

      if (!reducedMotion) {
        simTime += dt;
        if (introAge < 1) introAge = Math.min(1, introAge + dt / INTRO_DUR);

        // Smooth pointer easing & movement-only trail deposit
        const ka = 1 - Math.exp(-dt * 7);
        easeA += (ptrActive - easeA) * ka;
        if (ptrActive) {
          if (easeX < -1000) {
            easeX = ptrX;
            easeY = ptrY;
            lastDepositX = ptrX;
            lastDepositY = ptrY;
          } else {
            const k = 1 - Math.exp(-dt * 12);
            easeX += (ptrX - easeX) * k;
            easeY += (ptrY - easeY) * k;
          }
          const moved = Math.hypot(easeX - lastDepositX, easeY - lastDepositY);
          if (moved > 1.5) {
            const o = trailHead * 3;
            trail[o] = easeX;
            trail[o + 1] = easeY;
            trail[o + 2] = Math.min(1.0, moved / 10.0);
            trailHead = (trailHead + 1) % TRAIL_LEN;
            lastDepositX = easeX;
            lastDepositY = easeY;
          }
        }

        // Decay trail
        const decay = Math.exp(-dt * 2.8);
        for (let i = 0; i < TRAIL_LEN; i++) {
          trail[i * 3 + 2] *= decay;
        }

        // Advance click ripples
        for (let i = 0; i < RIPPLE_LEN; i++) {
          const o = i * 4;
          if (ripples[o + 3] > 0.5) {
            ripples[o + 2] += dt;
            if (ripples[o + 2] > RIPPLE_LIFE) ripples[o + 3] = 0;
          }
        }
      }

      renderer.draw(simTime, introEase());
    }

    if (!reducedMotion) requestAnimationFrame(tick);
  }

  window.addEventListener('resize', () => {
    resize();
    readDimRect();
    renderer.draw(simTime, introEase());
  });

  readDimRect();
  renderer.draw(simTime, introEase());
  if (!reducedMotion) requestAnimationFrame(tick);
})();
