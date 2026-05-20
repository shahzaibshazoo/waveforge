/**
 * WaveForge — main.js
 * Production UI: particle system, typed text, counters, scroll reveal,
 * sticky nav, mobile menu, code copy, benchmark chart, smooth scroll.
 */

/* ─────────────────────────────────────────────────────────────────────────
   1. PARTICLE SYSTEM  — EM wave visualization in hero canvas
   ───────────────────────────────────────────────────────────────────────── */
(function initParticles() {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // Particle pool
  const PARTICLE_COUNT = 220;
  const particles = [];
  const CONNECT_DIST = 110;

  function resize() {
    canvas.width  = canvas.offsetWidth  || window.innerWidth;
    canvas.height = canvas.offsetHeight || window.innerHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  // Colour from amplitude in [-1, 1] → cyan / violet / green gradient
  function ampColor(amp, alpha) {
    // amp in [0, 1]
    const r = Math.round(80  + amp * 100);
    const g = Math.round(200 - amp * 80);
    const b = Math.round(220 - amp * 30);
    return `rgba(${r},${g},${b},${alpha})`;
  }

  // Initialise particles spread across canvas with random phase offsets
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    particles.push({
      x:     Math.random(),   // 0-1 normalised
      y:     Math.random(),
      phase: Math.random() * Math.PI * 2,
      speed: 0.00008 + Math.random() * 0.00012,
      radius: 1.5 + Math.random() * 2,
    });
  }

  let t = 0;

  function drawParticles() {
    const W = canvas.width;
    const H = canvas.height;
    const cx = 0.5, cy = 0.5;  // normalised centre

    ctx.clearRect(0, 0, W, H);

    // Compute screen positions + amplitude for this frame
    const pts = particles.map(p => {
      const dx = p.x - cx;
      const dy = p.y - cy;
      const dist = Math.sqrt(dx * dx + dy * dy);
      // EM wave: superposition of two outward-propagating rings
      const wave =  0.6 * Math.sin(dist * 18 - t * 1.4 + p.phase)
                  + 0.4 * Math.sin(dist * 32 - t * 2.1 + p.phase * 0.7);
      const amp = (wave + 1) * 0.5;   // 0-1
      return { sx: p.x * W, sy: p.y * H, amp };
    });

    // Connection lines — only draw when close
    ctx.lineWidth = 0.6;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const dx = pts[i].sx - pts[j].sx;
        const dy = pts[i].sy - pts[j].sy;
        const d  = Math.sqrt(dx * dx + dy * dy);
        if (d < CONNECT_DIST) {
          const alpha = (1 - d / CONNECT_DIST) * 0.18;
          ctx.strokeStyle = ampColor((pts[i].amp + pts[j].amp) * 0.5, alpha);
          ctx.beginPath();
          ctx.moveTo(pts[i].sx, pts[i].sy);
          ctx.lineTo(pts[j].sx, pts[j].sy);
          ctx.stroke();
        }
      }
    }

    // Dots
    pts.forEach(p => {
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, particles[pts.indexOf(p)]?.radius ?? 2, 0, Math.PI * 2);
      ctx.fillStyle = ampColor(p.amp, 0.55 + p.amp * 0.35);
      ctx.fill();
    });

    t += 0.018;
    requestAnimationFrame(drawParticles);
  }

  // Use indexed loop for dot radius (avoid indexOf inside hot path)
  // Re-implement draw with index access
  function draw() {
    const W = canvas.width;
    const H = canvas.height;
    ctx.clearRect(0, 0, W, H);

    // Compute screen coords + amplitude
    const pts = new Array(PARTICLE_COUNT);
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const p  = particles[i];
      const dx = p.x - 0.5;
      const dy = p.y - 0.5;
      const d  = Math.sqrt(dx * dx + dy * dy);
      const w  = 0.6 * Math.sin(d * 18 - t * 1.4 + p.phase)
               + 0.4 * Math.sin(d * 32 - t * 2.1 + p.phase * 0.7);
      pts[i] = { sx: p.x * W, sy: p.y * H, amp: (w + 1) * 0.5, r: p.radius };
    }

    // Lines
    ctx.lineWidth = 0.7;
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      for (let j = i + 1; j < PARTICLE_COUNT; j++) {
        const ex = pts[i].sx - pts[j].sx;
        const ey = pts[i].sy - pts[j].sy;
        const ed = ex * ex + ey * ey;
        if (ed < CONNECT_DIST * CONNECT_DIST) {
          const alpha = (1 - Math.sqrt(ed) / CONNECT_DIST) * 0.2;
          ctx.strokeStyle = ampColor((pts[i].amp + pts[j].amp) * 0.5, alpha);
          ctx.beginPath();
          ctx.moveTo(pts[i].sx, pts[i].sy);
          ctx.lineTo(pts[j].sx, pts[j].sy);
          ctx.stroke();
        }
      }
    }

    // Dots
    for (let i = 0; i < PARTICLE_COUNT; i++) {
      const p = pts[i];
      ctx.beginPath();
      ctx.arc(p.sx, p.sy, p.r, 0, Math.PI * 2);
      ctx.fillStyle = ampColor(p.amp, 0.5 + p.amp * 0.4);
      ctx.fill();
    }

    t += 0.018;
    requestAnimationFrame(draw);
  }

  draw();
})();

/* ─────────────────────────────────────────────────────────────────────────
   2. TYPED TEXT ANIMATION
   ───────────────────────────────────────────────────────────────────────── */
(function initTyped() {
  const el = document.querySelector('.hero-subtitle');
  if (!el) return;

  const phrases = [
    'GPU-Native FDTD Simulation',
    'Brain Tumor Detection',
    'Automotive Radar Imaging',
    '21.8× Faster Than Meep',
    'Breast Cancer Imaging',
  ];

  let pi = 0, ci = 0, deleting = false;
  const CURSOR = '<span class="typed-cursor">|</span>';

  // Inject cursor CSS once
  const style = document.createElement('style');
  style.textContent = `.typed-cursor{display:inline-block;animation:blink .7s step-end infinite}@keyframes blink{0%,100%{opacity:1}50%{opacity:0}}`;
  document.head.appendChild(style);

  function tick() {
    const phrase = phrases[pi];
    if (!deleting) {
      ci++;
      el.innerHTML = phrase.slice(0, ci) + CURSOR;
      if (ci === phrase.length) {
        deleting = true;
        setTimeout(tick, 1800);
        return;
      }
      setTimeout(tick, 65);
    } else {
      ci--;
      el.innerHTML = phrase.slice(0, ci) + CURSOR;
      if (ci === 0) {
        deleting = false;
        pi = (pi + 1) % phrases.length;
        setTimeout(tick, 400);
        return;
      }
      setTimeout(tick, 32);
    }
  }

  setTimeout(tick, 800);
})();

/* ─────────────────────────────────────────────────────────────────────────
   3. COUNTER ANIMATION  — [data-count] elements count up on scroll
   ───────────────────────────────────────────────────────────────────────── */
(function initCounters() {
  const els = document.querySelectorAll('[data-count]');
  if (!els.length) return;

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function animateCounter(el) {
    const target   = parseFloat(el.dataset.count);
    const isFloat  = target % 1 !== 0;
    const duration = 1600;
    const start    = performance.now();

    function step(now) {
      const prog = Math.min((now - start) / duration, 1);
      const val  = target * easeOut(prog);
      el.textContent = isFloat ? val.toFixed(1) : Math.round(val).toString();
      if (prog < 1) requestAnimationFrame(step);
      else el.textContent = isFloat ? target.toFixed(1) : target.toString();
    }
    requestAnimationFrame(step);
  }

  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        animateCounter(e.target);
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.5 });

  els.forEach(el => obs.observe(el));
})();

/* ─────────────────────────────────────────────────────────────────────────
   4. SCROLL REVEAL  — .reveal and grid children fade + slide in
   ───────────────────────────────────────────────────────────────────────── */
(function initReveal() {
  const selectors = '.reveal, .feature-card, .gallery-item, .doc-card, .roadmap-item, .bench-card, .qs-option';
  const items = document.querySelectorAll(selectors);

  const style = document.createElement('style');
  style.textContent = `
    .will-reveal { opacity: 0; transform: translateY(24px); transition: opacity .55s ease, transform .55s ease; }
    .will-reveal.revealed { opacity: 1; transform: translateY(0); }
  `;
  document.head.appendChild(style);

  items.forEach((el, i) => {
    el.classList.add('will-reveal');
    // Stagger siblings inside a grid by their index
    const delay = (i % 6) * 70;
    el.style.transitionDelay = delay + 'ms';
  });

  const obs = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.classList.add('revealed');
        obs.unobserve(e.target);
      }
    });
  }, { threshold: 0.1, rootMargin: '0px 0px -40px 0px' });

  items.forEach(el => obs.observe(el));
})();

/* ─────────────────────────────────────────────────────────────────────────
   5. STICKY NAV  — transparent → dark-blur + active section highlighting
   ───────────────────────────────────────────────────────────────────────── */
(function initNav() {
  const nav     = document.getElementById('nav');
  const links   = nav ? nav.querySelectorAll('a[href^="#"]') : [];
  const sections = Array.from(document.querySelectorAll('section[id]'));
  if (!nav) return;

  function onScroll() {
    // Background state
    if (window.scrollY > 60) {
      nav.style.background = 'rgba(10,14,26,0.97)';
      nav.style.backdropFilter = 'blur(12px)';
      nav.style.boxShadow = '0 2px 20px rgba(0,0,0,0.4)';
    } else {
      nav.style.background = 'rgba(10,14,26,0.75)';
      nav.style.backdropFilter = 'blur(8px)';
      nav.style.boxShadow = 'none';
    }

    // Active section
    const scrollMid = window.scrollY + window.innerHeight / 3;
    let active = null;
    sections.forEach(sec => {
      if (sec.offsetTop <= scrollMid) active = sec.id;
    });
    links.forEach(a => {
      const href = a.getAttribute('href').slice(1);
      a.classList.toggle('nav-active', href === active);
    });
  }

  window.addEventListener('scroll', onScroll, { passive: true });
  onScroll();
})();

/* ─────────────────────────────────────────────────────────────────────────
   6. MOBILE MENU  — hamburger toggle, slide-in, outside-click close
   ───────────────────────────────────────────────────────────────────────── */
(function initMobileMenu() {
  // Create hamburger button and mobile panel dynamically if not in HTML
  const nav = document.getElementById('nav');
  if (!nav) return;

  const btn = document.createElement('button');
  btn.className = 'nav-hamburger';
  btn.setAttribute('aria-label', 'Open menu');
  btn.innerHTML = '<span></span><span></span><span></span>';
  nav.querySelector('.nav-inner').appendChild(btn);

  const panel = document.createElement('div');
  panel.className = 'mobile-menu';
  // Clone nav links into panel
  const linksList = nav.querySelector('.nav-links');
  if (linksList) panel.innerHTML = linksList.outerHTML;
  document.body.appendChild(panel);

  const style = document.createElement('style');
  style.textContent = `
    .nav-hamburger{display:none;flex-direction:column;gap:5px;background:none;border:none;cursor:pointer;padding:6px;margin-left:auto}
    .nav-hamburger span{display:block;width:24px;height:2px;background:#e2e8f0;border-radius:2px;transition:transform .25s,opacity .25s}
    .nav-hamburger.open span:nth-child(1){transform:translateY(7px) rotate(45deg)}
    .nav-hamburger.open span:nth-child(2){opacity:0}
    .nav-hamburger.open span:nth-child(3){transform:translateY(-7px) rotate(-45deg)}
    .mobile-menu{position:fixed;top:0;right:-280px;width:260px;height:100vh;background:#0d1117;z-index:999;padding:80px 24px 24px;transition:right .3s ease;box-shadow:-4px 0 24px rgba(0,0,0,.5)}
    .mobile-menu.open{right:0}
    .mobile-menu ul{list-style:none;padding:0;margin:0;display:flex;flex-direction:column;gap:20px}
    .mobile-menu a{color:#e2e8f0;text-decoration:none;font-size:1.1rem}
    @media(max-width:768px){.nav-hamburger{display:flex}.nav-links{display:none!important}}
    .nav-active{color:#4fd1c5!important;border-bottom:2px solid #4fd1c5}
  `;
  document.head.appendChild(style);

  function open()  { btn.classList.add('open'); panel.classList.add('open'); }
  function close() { btn.classList.remove('open'); panel.classList.remove('open'); }

  btn.addEventListener('click', () => panel.classList.contains('open') ? close() : open());
  document.addEventListener('click', e => {
    if (!panel.contains(e.target) && e.target !== btn) close();
  });
  panel.querySelectorAll('a').forEach(a => a.addEventListener('click', close));
})();

/* ─────────────────────────────────────────────────────────────────────────
   7. CODE BLOCK COPY  — copy button + toast
   ───────────────────────────────────────────────────────────────────────── */
(function initCopyButtons() {
  const blocks = document.querySelectorAll('.code-block');
  if (!blocks.length) return;

  const style = document.createElement('style');
  style.textContent = `
    .code-block{position:relative}
    .copy-btn{position:absolute;top:10px;right:10px;background:rgba(79,209,197,.15);border:1px solid rgba(79,209,197,.35);color:#4fd1c5;font-size:.75rem;padding:4px 10px;border-radius:5px;cursor:pointer;transition:background .2s}
    .copy-btn:hover{background:rgba(79,209,197,.3)}
    .copy-toast{position:fixed;bottom:28px;right:28px;background:#4fd1c5;color:#0a0e1a;padding:10px 20px;border-radius:8px;font-weight:600;font-size:.9rem;opacity:0;pointer-events:none;transform:translateY(10px);transition:opacity .25s,transform .25s;z-index:9999}
    .copy-toast.show{opacity:1;transform:translateY(0)}
  `;
  document.head.appendChild(style);

  // Single shared toast element
  const toast = document.createElement('div');
  toast.className = 'copy-toast';
  toast.textContent = 'Copied!';
  document.body.appendChild(toast);

  let toastTimer = null;
  function showToast() {
    toast.classList.add('show');
    clearTimeout(toastTimer);
    toastTimer = setTimeout(() => toast.classList.remove('show'), 2000);
  }

  blocks.forEach(block => {
    const btn = document.createElement('button');
    btn.className = 'copy-btn';
    btn.textContent = 'Copy';
    block.appendChild(btn);

    btn.addEventListener('click', () => {
      const code = block.querySelector('code, pre');
      const text = code ? code.innerText : block.innerText;
      navigator.clipboard.writeText(text).then(() => {
        btn.textContent = 'Copied!';
        showToast();
        setTimeout(() => { btn.textContent = 'Copy'; }, 2000);
      }).catch(() => {
        // Fallback for non-secure contexts
        const ta = document.createElement('textarea');
        ta.value = text;
        ta.style.position = 'fixed';
        ta.style.opacity = '0';
        document.body.appendChild(ta);
        ta.select();
        document.execCommand('copy');
        document.body.removeChild(ta);
        showToast();
      });
    });
  });
})();

/* ─────────────────────────────────────────────────────────────────────────
   8. BENCHMARK CHART  — animated canvas bar chart
   ───────────────────────────────────────────────────────────────────────── */
(function initBenchChart() {
  // Inject a canvas into the benchmarks section
  const section = document.getElementById('benchmarks');
  if (!section) return;

  const wrapper = document.createElement('div');
  wrapper.style.cssText = 'margin:2rem auto 0;max-width:640px;';
  const canvas  = document.createElement('canvas');
  canvas.id     = 'bench-chart';
  canvas.width  = 640;
  canvas.height = 300;
  canvas.style.cssText = 'width:100%;height:auto;border-radius:12px;background:rgba(255,255,255,0.04);';
  wrapper.appendChild(canvas);
  // Insert after bench-numbers
  const ref = section.querySelector('.bench-card-full') || section.querySelector('.bench-numbers');
  if (ref) ref.insertAdjacentElement('afterend', wrapper);
  else section.querySelector('.container').appendChild(wrapper);

  const bars = [
    { label: '64²',  wf: 2.8,  meep: 14.2, unit: 'Mcells/s' },
    { label: '128²', wf: 14.5, meep: 14.5, unit: 'Mcells/s' },
    { label: '256²', wf: 89.0, meep: 14.8, unit: 'Mcells/s' },
    { label: '512²', wf: 350,  meep: 16.0, unit: 'Mcells/s' },
  ];

  const MAX_VAL = 350;
  let progress   = 0; // 0 → 1
  let animating  = false;

  function easeOut(t) { return 1 - Math.pow(1 - t, 3); }

  function renderChart(prog) {
    const ctx = canvas.getContext('2d');
    const W   = canvas.width;
    const H   = canvas.height;
    const pad = { top: 28, right: 28, bottom: 48, left: 52 };
    const chartW = W - pad.left - pad.right;
    const chartH = H - pad.top  - pad.bottom;

    ctx.clearRect(0, 0, W, H);

    // Grid lines
    ctx.strokeStyle = 'rgba(255,255,255,0.07)';
    ctx.lineWidth   = 1;
    [0, 100, 200, 300].forEach(v => {
      const y = pad.top + chartH * (1 - v / MAX_VAL);
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(pad.left + chartW, y); ctx.stroke();
      ctx.fillStyle = 'rgba(255,255,255,0.3)';
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'right';
      ctx.fillText(v, pad.left - 6, y + 4);
    });

    // Y-axis label
    ctx.save();
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.font = '11px Inter, sans-serif';
    ctx.translate(12, pad.top + chartH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('Mcells / s', 0, 0);
    ctx.restore();

    // Title
    ctx.fillStyle = 'rgba(255,255,255,0.7)';
    ctx.font = '600 13px Inter, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('GPU (WaveForge) vs CPU (Meep) Throughput', W / 2, 17);

    const groupW  = chartW / bars.length;
    const barW    = groupW * 0.28;
    const gap     = groupW * 0.06;

    bars.forEach((b, i) => {
      const gx = pad.left + i * groupW + groupW * 0.12;
      const ep = easeOut(prog);

      // WaveForge bar (cyan)
      const wfH  = chartH * (b.wf / MAX_VAL) * ep;
      const wfY  = pad.top + chartH - wfH;
      const grad = ctx.createLinearGradient(0, wfY, 0, wfY + wfH);
      grad.addColorStop(0, '#4fd1c5');
      grad.addColorStop(1, '#2b6cb0');
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(gx, wfY, barW, wfH, [4, 4, 0, 0])
        : ctx.rect(gx, wfY, barW, wfH);
      ctx.fill();

      // Meep bar (muted)
      const meepH = chartH * (b.meep / MAX_VAL) * ep;
      const meepY = pad.top + chartH - meepH;
      const mx    = gx + barW + gap;
      ctx.fillStyle = 'rgba(160,174,192,0.45)';
      ctx.beginPath();
      ctx.roundRect ? ctx.roundRect(mx, meepY, barW, meepH, [4, 4, 0, 0])
        : ctx.rect(mx, meepY, barW, meepH);
      ctx.fill();

      // Value labels (only when nearly done)
      if (prog > 0.85) {
        const alpha = Math.min((prog - 0.85) / 0.15, 1);
        ctx.fillStyle = `rgba(79,209,197,${alpha})`;
        ctx.font = '600 11px Inter, sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(b.wf >= 100 ? Math.round(b.wf) : b.wf.toFixed(1), gx + barW / 2, wfY - 5);
      }

      // Group label
      ctx.fillStyle = 'rgba(255,255,255,0.55)';
      ctx.font = '12px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(b.label, gx + barW + gap / 2, pad.top + chartH + 20);
    });

    // Legend
    const lx = pad.left + chartW - 160;
    const ly = pad.top + 10;
    ctx.fillStyle = '#4fd1c5';
    ctx.fillRect(lx, ly, 14, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.fillText('WaveForge GPU', lx + 18, ly + 9);
    ctx.fillStyle = 'rgba(160,174,192,0.55)';
    ctx.fillRect(lx, ly + 18, 14, 10);
    ctx.fillStyle = 'rgba(255,255,255,0.6)';
    ctx.fillText('Meep CPU', lx + 18, ly + 27);
  }

  function animateChart() {
    if (animating) return;
    animating = true;
    const start = performance.now();
    const dur   = 1400;

    function step(now) {
      progress = Math.min((now - start) / dur, 1);
      renderChart(progress);
      if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
  }

  const obs = new IntersectionObserver(entries => {
    if (entries[0].isIntersecting) {
      animateChart();
      obs.disconnect();
    }
  }, { threshold: 0.3 });
  obs.observe(canvas);

  // Initial render at zero
  renderChart(0);
})();

/* ─────────────────────────────────────────────────────────────────────────
   9. SMOOTH SCROLL  — all anchor links, offset for fixed nav
   ───────────────────────────────────────────────────────────────────────── */
(function initSmoothScroll() {
  const NAV_HEIGHT = 64;

  document.addEventListener('click', e => {
    const link = e.target.closest('a[href^="#"]');
    if (!link) return;
    const id  = link.getAttribute('href').slice(1);
    if (!id) return;
    const target = document.getElementById(id);
    if (!target) return;
    e.preventDefault();
    const top = target.getBoundingClientRect().top + window.scrollY - NAV_HEIGHT;
    window.scrollTo({ top, behavior: 'smooth' });
  });
})();
