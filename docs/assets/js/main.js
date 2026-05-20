// ── Animated wave canvas in hero ────────────────────────────────────────
(function () {
  const canvas = document.getElementById('wave-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  function resize() {
    canvas.width  = canvas.offsetWidth;
    canvas.height = canvas.offsetHeight;
  }
  resize();
  window.addEventListener('resize', resize);

  const waves = [
    { amp: 60,  freq: 0.008, speed: 0.012, phase: 0,   color: 'rgba(79,209,197,0.5)' },
    { amp: 40,  freq: 0.012, speed: 0.018, phase: 2,   color: 'rgba(99,179,237,0.4)' },
    { amp: 80,  freq: 0.005, speed: 0.008, phase: 4,   color: 'rgba(159,122,234,0.3)' },
    { amp: 30,  freq: 0.020, speed: 0.025, phase: 1,   color: 'rgba(79,209,197,0.25)' },
  ];

  let t = 0;
  function draw() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    waves.forEach(w => {
      ctx.beginPath();
      ctx.strokeStyle = w.color;
      ctx.lineWidth = 1.5;
      for (let x = 0; x <= canvas.width; x += 2) {
        const y = canvas.height / 2
          + w.amp * Math.sin(w.freq * x + t * w.speed + w.phase)
          + (w.amp * 0.4) * Math.sin(w.freq * 2.3 * x - t * w.speed * 0.7);
        x === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
      }
      ctx.stroke();
    });
    t++;
    requestAnimationFrame(draw);
  }
  draw();
})();

// ── Sticky nav highlight on scroll ──────────────────────────────────────
(function () {
  const nav = document.getElementById('nav');
  window.addEventListener('scroll', () => {
    if (window.scrollY > 60) {
      nav.style.background = 'rgba(10,14,26,0.97)';
    } else {
      nav.style.background = 'rgba(10,14,26,0.85)';
    }
  });
})();

// ── Smooth reveal on scroll ──────────────────────────────────────────────
(function () {
  const items = document.querySelectorAll(
    '.feature-card, .gallery-item, .doc-card, .roadmap-item, .bench-card, .qs-option'
  );
  const observer = new IntersectionObserver((entries) => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        e.target.style.opacity = '1';
        e.target.style.transform = e.target.style.transform.replace('translateY(20px)', 'translateY(0)');
        observer.unobserve(e.target);
      }
    });
  }, { threshold: 0.1 });

  items.forEach(el => {
    el.style.opacity = '0';
    el.style.transform = (el.style.transform || '') + ' translateY(20px)';
    el.style.transition = 'opacity 0.5s ease, transform 0.5s ease';
    observer.observe(el);
  });
})();
