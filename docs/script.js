/* ═══════════════════════════════════════════════════════════════════════
   Demo grid: filtering, lazy loading, autoplay-on-scroll.

   68 clips at ~180 KB each is 12 MB, so nothing loads until it is close to
   the viewport — `preload="none"` plus an IntersectionObserver that sets
   the src. Videos play only while visible and pause when they leave, so a
   long scroll never has more than a handful decoding at once.
   ═══════════════════════════════════════════════════════════════════════ */

const SUITE_LABEL = { spatial: 'SafeLIBERO-Spatial', object: 'SafeLIBERO-Object', goal: 'SafeLIBERO-Goal' };
const LEVEL_LABEL = { LI: 'Level I — obstacle beside the target', LII: 'Level II — obstacle blocking the path' };

const state = { suite: 'all', level: 'all', autoplay: true, expanded: false };

const listEl  = document.getElementById('demo-list');
const countEl = document.getElementById('demo-count');

/* ── lazy load + play/pause as cards enter and leave ───────────────────── */
const io = new IntersectionObserver((entries) => {
  for (const e of entries) {
    const v = e.target;
    if (e.isIntersecting) {
      if (!v.src) v.src = v.dataset.src;
      if (state.autoplay) v.play().catch(() => { /* autoplay blocked; the poster stands */ });
    } else {
      v.pause();
    }
  }
}, { rootMargin: '250px 0px', threshold: 0.15 });

/* ── build ─────────────────────────────────────────────────────────────── */
function matches(d) {
  return (state.expanded || d.hero)
      && (state.suite === 'all' || d.suite === state.suite)
      && (state.level === 'all' || d.level === state.level);
}

function appendMore() {
  const more = document.createElement('div');
  more.className = 'more-wrap';
  more.innerHTML = `<button class="more-btn" type="button">${
    state.expanded ? 'Show fewer' : `Show all ${DEMOS.length} clips`}</button>`;
  more.querySelector('.more-btn').addEventListener('click', () => {
    state.expanded = !state.expanded;
    render();
    if (!state.expanded) document.getElementById('demos').scrollIntoView({ block: 'start' });
  });
  listEl.appendChild(more);
}


function makeCard(d, withTag = false) {
  const card = document.createElement('div');
  card.className = 'vcard';
  card.innerHTML = `
    ${withTag ? `<div class="hero-tag">${SUITE_LABEL[d.suite]} &middot; ${
      d.level === 'LI' ? 'Level I' : 'Level II'}</div>` : ''}
    <video data-src="videos/${d.file}" muted loop playsinline preload="none"
           aria-label="Base pi-0.5 on the left, self-distilled on the right: ${d.instruction}, held-out initial state ${d.init}"></video>
    <div class="vcap">
      <span>${withTag ? `&ldquo;${d.instruction}&rdquo;` : `held-out initial state ${d.init}`}</span>
      <button class="replay" type="button">restart</button>
    </div>`;

  const v = card.querySelector('video');
  card.querySelector('.replay').addEventListener('click', () => {
    if (!v.src) v.src = v.dataset.src;
    v.currentTime = 0;
    v.play().catch(() => {});
  });
  v.addEventListener('click', () => (v.paused ? v.play().catch(() => {}) : v.pause()));
  io.observe(v);
  return card;
}

function render() {
  io.disconnect();
  listEl.innerHTML = '';

  const shown = DEMOS.filter(matches);
  countEl.textContent = !state.expanded
    ? `${shown.length} of ${DEMOS.length} clips \u2014 one per suite and obstacle level`
    : shown.length === DEMOS.length
      ? `${DEMOS.length} clips`
      : `${shown.length} of ${DEMOS.length} clips`;

  if (!shown.length) {
    listEl.innerHTML = '<p class="fineprint">No clips match that combination.</p>';
    return;
  }

  if (!state.expanded) {
    const grid = document.createElement('div');
    grid.className = 'vgrid';
    const order = ['spatial', 'object', 'goal'];
    shown.sort((a, b) => order.indexOf(a.suite) - order.indexOf(b.suite)
                      || a.level.length - b.level.length);
    for (const d of shown) grid.appendChild(makeCard(d, true));
    listEl.appendChild(grid);
    appendMore();
    return;
  }

  // group by suite → level → task, keeping a stable suite order
  const order = ['spatial', 'object', 'goal'];
  const groups = new Map();
  for (const d of shown) {
    const key = `${d.suite}|${d.level}|${d.task}`;
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(d);
  }

  const keys = [...groups.keys()].sort((a, b) => {
    const [sa, la, ta] = a.split('|'), [sb, lb, tb] = b.split('|');
    return order.indexOf(sa) - order.indexOf(sb)
        || la.length - lb.length
        || (+ta) - (+tb);
  });

  for (const key of keys) {
    const items = groups.get(key).sort((a, b) => a.init - b.init);
    const d0 = items[0];

    const block = document.createElement('div');
    block.className = 'task-block';
    block.innerHTML = `
      <div class="task-head">
        <span class="task-suite">${SUITE_LABEL[d0.suite]}</span>
        <span class="task-instr">&ldquo;${d0.instruction}&rdquo;</span>
        <span class="task-level">${LEVEL_LABEL[d0.level]}</span>
      </div>
      <div class="vgrid"></div>`;

    const grid = block.querySelector('.vgrid');
    for (const d of items) {
      grid.appendChild(makeCard(d));
    }
    listEl.appendChild(block);
  }

  appendMore();
}

/* ── filter chips ──────────────────────────────────────────────────────── */
for (const chip of document.querySelectorAll('.chip[data-filter]')) {
  chip.addEventListener('click', () => {
    const { filter, value } = chip.dataset;
    state[filter] = value;
    if (value !== 'all') state.expanded = true;   // a filter is a request for more, not less
    for (const sib of document.querySelectorAll(`.chip[data-filter="${filter}"]`)) {
      sib.classList.toggle('active', sib === chip);
    }
    render();
  });
}

const autoBtn = document.getElementById('autoplay-toggle');
autoBtn.addEventListener('click', () => {
  state.autoplay = !state.autoplay;
  autoBtn.classList.toggle('active', state.autoplay);
  autoBtn.textContent = state.autoplay ? 'Autoplay on scroll' : 'Click to play';
  for (const v of listEl.querySelectorAll('video')) {
    if (state.autoplay) { if (v.src) v.play().catch(() => {}); } else { v.pause(); }
  }
});

/* respect a reduced-motion preference by starting paused */
if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
  state.autoplay = false;
  autoBtn.classList.remove('active');
  autoBtn.textContent = 'Click to play';
}

/* ── bibtex copy ───────────────────────────────────────────────────────── */
const copyBtn = document.getElementById('copy-bib');
copyBtn.addEventListener('click', async () => {
  try {
    await navigator.clipboard.writeText(document.getElementById('bibtex').textContent);
    copyBtn.textContent = 'Copied';
    setTimeout(() => (copyBtn.textContent = 'Copy'), 1600);
  } catch {
    copyBtn.textContent = 'Select and copy';
    setTimeout(() => (copyBtn.textContent = 'Copy'), 1600);
  }
});

render();
