/* iSharded's shared drawing. page() inlines this at the isharded:ui.js marker, so it fetches
   nothing, holds no page state, and sets widths from script, which the assistant's policy allows. */
const UI = (() => {
  const esc = value => String(value ?? '').replace(/[&<>'"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
  const num = (value, digits = 1) => value == null || Number.isNaN(Number(value)) ? '—' : Number(value).toFixed(digits).replace(/\.0+$/, '');
  const bytes = value => { if (!value) return '0 B'; const units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']; const i = Math.min(Math.floor(Math.log(value) / Math.log(1024)), units.length - 1); return `${num(value / 1024 ** i, i > 2 ? 1 : 0)} ${units[i]}`; };
  const ms = value => value == null ? '—' : `${num(value)} ms`;
  const duration = seconds => { if (seconds == null) return '—'; const d = Math.floor(seconds / 86400), h = Math.floor(seconds % 86400 / 3600), m = Math.floor(seconds % 3600 / 60); return d ? `${d} d ${h} h` : h ? `${h} h ${m} min` : `${m} min`; };
  const relative = stamp => { const s = Math.max(0, Math.floor(Date.now() / 1000 - stamp)); return s < 5 ? 'just now' : s < 60 ? `${s} s ago` : s < 3600 ? `${Math.floor(s / 60)} min ago` : s < 86400 ? `${Math.floor(s / 3600)} h ago` : `${Math.floor(s / 86400)} d ago`; };
  const clock = stamp => new Date(stamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const count = (n, word) => `${n} ${word}${n === 1 ? '' : 's'}`;
  const whole = value => Number(value ?? 0).toLocaleString();
  const span = (start, end) => Number(end) - 1 > Number(start) ? `${Number(start)}–${Number(end) - 1}` : `${Number(start)}`;
  // GGUF ids carry a repository and a file, and a split file ends in its part number.
  const modelName = id => { const text = String(id || ''); if (/\.gguf$/i.test(text)) return text.split(/[:\\/]/).pop().replace(/\.gguf$/i, '').replace(/-\d{5}-of-\d{5}$/, ''); if (/^([A-Za-z]:[\\/]|[\\/.~])/.test(text)) return text.split(/[\\/]/).filter(Boolean).pop() || text; return text; };
  const jobRunning = job => Boolean(job) && !['ready', 'failed', 'cancelled'].includes(job.phase);
  const median = list => { const sorted = [...list].sort((a, b) => a - b), mid = sorted.length >> 1; return sorted.length ? sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2 : null; };
  const glyphHtml = kind => `<i class="glyph ${esc(kind)}" aria-hidden="true"></i>`;
  const TIER_ORDER = ['vram', 'unified', 'ram', 'disk'];
  const TIER_NAMES = { vram: 'VRAM', unified: 'Unified', ram: 'RAM', disk: 'Disk' };
  const SPILL_SLOW = 1.5, BAND_SLOW = 1.25, BAND_NOTE = 1.5;

  function tierOf(data, workerId, poolId) {
    const kind = (data.workers?.[workerId]?.hardware?.memory_pools || []).find(pool => pool.pool_id === poolId)?.kind;
    return TIER_NAMES[kind] ? kind : 'disk';
  }

  /* The route as the drawings read it. `tab` holds what this tab remembers between polls:
     the last five compute readings a stage gave, and its last direct link reading. */
  function routeModel(data, tab = {}) {
    const job = jobRunning(data.load_job) ? data.load_job : null;
    const planned = !(data.route || []).length, route = planned ? job?.route || [] : data.route;
    if (!route.length) return null;
    const fresh = tab.seen !== data, compute = tab.compute ||= {}, direct = tab.direct ||= {};
    tab.seen = data;
    const plan = data.active_plan || {}, spills = plan.memory?.stages || [];
    const stages = route.map((stage, index) => {
      const id = stage.worker_id, worker = data.workers?.[id], telemetry = worker?.runtime?.telemetry || {};
      const start = Number(stage.start_layer), end = Number(stage.end_layer), history = compute[id] ||= [];
      if (fresh && telemetry.compute_ms_ema != null) { history.push(telemetry.compute_ms_ema); if (history.length > 5) history.shift(); }
      const round = telemetry.ring?.link_rtt_us;
      if (fresh && round) direct[id] = round / 2000;
      const next = route[(index + 1) % route.length];
      const probe = route.length > 1 ? (data.links || []).find(link => link.source === id && link.target === next.worker_id)?.median_ms ?? null : null;
      const linkMs = route.length > 1 ? direct[id] ?? probe : null;
      const totals = {};
      for (const item of stage.placements || []) if (item.bytes) { const kind = tierOf(data, id, item.pool_id); totals[kind] = (totals[kind] || 0) + item.bytes; }
      const tiers = TIER_ORDER.filter(kind => totals[kind]).map(kind => ({ kind, bytes: totals[kind] }));
      const computeMs = telemetry.compute_ms_ema ?? history[history.length - 1] ?? null;
      return {
        n: index + 1, workerId: id, worker, start, end, layerCount: end - start, computeMs, steadyMs: history.length ? median(history) : computeMs, linkMs,
        linkSource: direct[id] != null ? 'direct' : probe != null ? 'probe' : null, tiers, tier: tiers[tiers.length - 1]?.kind ?? null,
        spilledBytes: spills.find(item => item.worker_id === id && Number(item.start_layer) === start)?.spilled_bytes || 0,
        trouble: !worker || !worker.live ? 'bad' : worker.quarantined ? 'warn' : null, failed: !worker || !worker.live || Boolean(worker.quarantined), moving: null, slowest: false, ratioToOthers: null,
      };
    });
    for (const stage of stages) stage.perLayerMs = stage.computeMs == null ? null : stage.computeMs / Math.max(stage.layerCount, 1);
    for (const stage of stages) {
      const others = stages.filter(other => other !== stage && other.perLayerMs != null).map(other => other.perLayerMs);
      stage.ratioToOthers = stage.perLayerMs != null && others.length && median(others) ? stage.perLayerMs / median(others) : null;
      if (!stage.trouble && stage.spilledBytes > 0 && stage.ratioToOthers >= SPILL_SLOW) { stage.trouble = 'warn'; stage.spillSlow = true; }
    }
    const timed = stages.filter(stage => stage.computeMs != null);
    if (timed.length) {
      const slowest = timed.reduce((a, b) => b.steadyMs > a.steadyMs ? b : a);
      const others = timed.filter(stage => stage !== slowest).map(stage => stage.steadyMs);
      slowest.slowest = true;
      slowest.standsOut = others.length > 0 && slowest.steadyMs >= BAND_SLOW * median(others);
    }
    if (job) {
      const progress = worker => { const item = job.workers?.[worker] || {}; return { to: worker, bytesDone: item.bytes_done || 0, bytesTotal: item.bytes_total || 0, rate: item.bytes_per_second || 0 }; };
      if (planned) for (const stage of stages) stage.moving = progress(stage.workerId);
      else if (job.model === data.model) for (const target of job.route || []) {
        const from = Number(target.start_layer), held = stages.find(stage => stage.start <= from && from < stage.end);
        if (held && held.workerId !== target.worker_id && !held.moving) held.moving = progress(target.worker_id);
      }
    }
    const sum = key => stages.reduce((n, stage) => n + (stage[key] ?? 0), 0), last = stages[stages.length - 1];
    return {
      name: modelName(data.model || job?.model), layers: Number(plan.layers) || last.end, stages, planned,
      passMs: timed.length === stages.length ? sum('computeMs') + sum('linkMs') : null, computeMs: sum('computeMs'), linksMs: sum('linkMs'), closingLinkMs: last.linkMs,
      ratePerSecond: data.metrics?.tokens_per_second_ema ?? null, tokensPerPass: stages[0].worker?.runtime?.telemetry?.draft_tokens_per_pass ?? null,
      draftModel: plan.draft_model ? modelName(plan.draft_model) : null, epoch: data.epoch, serving: data.status === 'ready',
    };
  }

  const stageRange = stage => `${stage.layerCount > 1 ? 'layers' : 'layer'} ${span(stage.start, stage.end)}`;
  // Entry, exit, flagged and the slowest three: the stages a shortened list keeps.
  function keyStages(model) {
    const keep = new Set([model.stages[0].n, model.stages[model.stages.length - 1].n]);
    for (const stage of model.stages) if (stage.trouble || stage.moving) keep.add(stage.n);
    for (const stage of model.stages.filter(item => item.computeMs != null).sort((a, b) => b.computeMs - a.computeMs).slice(0, 3)) keep.add(stage.n);
    return keep;
  }
  function stageWords(model, stage) {
    const last = stage.n === model.stages.length, parts = [`stage ${stage.n}`, stageRange(stage)];
    parts.push(stage.computeMs == null ? 'not measured yet' : `${num(stage.computeMs)} ms compute`);
    if (stage.linkMs != null) parts.push(`${num(stage.linkMs)} ms ${last ? 'link back to the entry' : 'link to the next stage'}`);
    if (stage.spilledBytes) parts.push(`${bytes(stage.spilledBytes)} below its best memory`);
    if (stage.worker?.quarantined) parts.push('quarantined');
    else if (stage.trouble === 'bad') parts.push('not answering');
    return parts.join(', ');
  }

  const make = (tag, className, text) => { const element = document.createElement(tag); if (className) element.className = className; if (text != null) element.textContent = text; return element; };
  const fillTiers = (element, stage) => {
    const total = stage.tiers.reduce((n, item) => n + item.bytes, 0);
    for (const item of stage.tiers) { const part = make('i', `tier-${item.kind}`); part.style.width = `${item.bytes / total * 100}%`; element.append(part); }
  };

  /* Time blocks and gaps across the width, at least 3 px and 2 px, so the slowest thing stays the widest. */
  function timeLayout(model, width) {
    const known = model.stages.filter(stage => stage.steadyMs != null);
    if (!known.length) return null;
    const perLayer = median(known.map(stage => stage.steadyMs / Math.max(stage.layerCount, 1)));
    const sizes = model.stages.flatMap(stage => [stage.steadyMs ?? perLayer * stage.layerCount, stage.linkMs ?? 0]);
    const floors = sizes.map((_, index) => index % 2 ? 2 : 3);
    const raw = sizes.reduce((a, b) => a + b, 0) || 1;
    let scale = width / raw;
    for (let round = 0; round < 3; round += 1) {
      const pinned = sizes.map((size, index) => size * scale < floors[index]);
      const room = width - floors.reduce((n, floor, index) => n + (pinned[index] ? floor : 0), 0);
      const free = sizes.reduce((n, size, index) => n + (pinned[index] ? 0 : size), 0);
      if (free > 0 && room > 0) scale = room / free;
    }
    const px = sizes.map((size, index) => Math.max(size * scale, floors[index]));
    const fit = width / px.reduce((a, b) => a + b, 0);
    let at = 0;
    return model.stages.map((_, index) => { const t0 = at; at += px[index * 2] * fit; const t1 = at; at += px[index * 2 + 1] * fit; return { t0: t0 / width * 100, t1: t1 / width * 100, h1: at / width * 100 }; });
  }
  // A figure moves every two seconds, and a fan that shivers looks broken.
  function steady(next, held, width) {
    if (!held || !next || held.length !== next.length) return next;
    const moved = next.some((item, index) => { const a = item.t1 - item.t0, b = held[index].t1 - held[index].t0, c = item.h1 - item.t1, d = held[index].h1 - held[index].t1; return Math.abs(a - b) > Math.max(b * .02, 100 / width) || Math.abs(c - d) > Math.max(d * .02, 100 / width); });
    return moved ? next : held;
  }

  /* The drawing: layers along the top, one pass in time along the bottom, a band between for each stage. */
  function drawBoard(element, model, options = {}) {
    const memo = element.board ||= { rove: 0, pinned: null, slid: false };
    memo.model = model; memo.options = options; element.routeHtml = null;
    if (!element.dataset.wired) wireBoard(element);
    const width = element.clientWidth - 96;
    if (width <= 0) return;
    const stages = model.stages, x = stages.map(stage => ({ x0: stage.start / model.layers * 100, x1: stage.end / model.layers * 100 }));
    const time = steady(timeLayout(model, width), memo.time, width);
    memo.time = time;
    const signature = JSON.stringify([width, model.layers, time, options.last, stages.map(stage => [stage.workerId, stage.start, stage.end, stage.trouble, stage.failed, stage.tiers.map(item => [item.kind, item.bytes]), stage.slowest, stage.standsOut, stage.moving, time ? num(stage.computeMs) : 0, num(stage.linkMs), num(stage.ratioToOthers), stage.computeMs == null])]);
    if (signature !== memo.signature) {
      memo.signature = signature;
      const had = element.contains(document.activeElement) ? document.activeElement.dataset.stage : null;
      paintBoard(element, model, options, x, time, width);
      if (had != null) element.querySelector(`button[data-stage="${had}"]`)?.focus();
    }
    showStage(element, memo.pinned);
  }

  function paintBoard(element, model, options, x, time, width) {
    const memo = element.board, stages = model.stages, moving = stages.filter(stage => stage.moving);
    element.textContent = '';
    const mimic = make('div', `mimic${moving.length ? ' moving' : ''}`);
    mimic.setAttribute('role', 'group');
    mimic.setAttribute('aria-label', `The route of ${model.name}, entry first. The stage table has the same facts.`);
    const row = (className, gutter) => { mimic.append(make('span', 'gut', gutter || '')); const lane = make('div', `m-lane ${className}`); mimic.append(lane); return lane; };

    const names = row('m-names');
    const rail = row('m-rail', `Layers 0–${model.layers - 1}`);
    const tick = width / model.layers;
    stages.forEach((stage, index) => {
      const block = make('button', `blk${stage.trouble ? ` ${stage.trouble}` : ''}${tick >= 3 ? ' tick' : ''}`);
      block.type = 'button';
      block.dataset.stage = index;
      block.tabIndex = index === memo.rove ? 0 : -1;
      block.setAttribute('aria-label', `${stage.workerId}, ${stageWords(model, stage)}`);
      block.style.left = `${x[index].x0}%`;
      block.style.width = `calc(${x[index].x1 - x[index].x0}% - 2px)`;
      if (tick >= 3) block.style.setProperty('--tick', `${tick.toFixed(3)}px`);
      fillTiers(block, stage);
      rail.append(block);
    });
    if (moving.length) {
      const load = row('m-load');
      for (const stage of moving) {
        const index = stage.n - 1, share = stage.moving.bytesTotal ? stage.moving.bytesDone / stage.moving.bytesTotal * 100 : 0;
        const bar = make('div', 'lay'), fill = make('i');
        bar.setAttribute('role', 'progressbar'); bar.setAttribute('aria-valuemin', '0'); bar.setAttribute('aria-valuemax', '100'); bar.setAttribute('aria-valuenow', share.toFixed(0));
        bar.setAttribute('aria-label', `${stage.moving.to} taking stage ${stage.n}`);
        bar.style.left = `${x[index].x0}%`; bar.style.width = `calc(${x[index].x1 - x[index].x0}% - 2px)`; fill.style.width = `${share}%`;
        bar.append(fill); load.append(bar);
      }
      if (moving.length === 1 && !model.planned) {
        const stage = moving[0], left = stage.moving.rate ? Math.ceil((stage.moving.bytesTotal - stage.moving.bytesDone) / stage.moving.rate) : null;
        load.append(make('span', 'lay-note', `${stage.moving.to} is taking these layers: ${bytes(stage.moving.bytesDone)} of ${bytes(stage.moving.bytesTotal)}${left != null ? `, about ${left} s left` : ''}`));
      }
    }
    let lanes = [];
    if (time) {
      const fan = row('m-fan');
      const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
      svg.setAttribute('viewBox', '0 0 1000 100'); svg.setAttribute('preserveAspectRatio', 'none'); svg.setAttribute('aria-hidden', 'true');
      const bands = stages.map((stage, index) => {
        const band = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        band.dataset.stage = index;
        if (stage.trouble) band.classList.add(stage.trouble); else if (stage.standsOut) band.classList.add('slow');
        svg.append(band);
        return band;
      });
      fan.append(svg);
      const noted = stages.filter(stage => stage.ratioToOthers >= BAND_NOTE).sort((a, b) => b.ratioToOthers - a.ratioToOthers).slice(0, 3);
      const notes = noted.map(stage => { const note = make('span', 'band-note', `${num(stage.ratioToOthers)} times the other stages, per layer`); fan.append(note); return { stage, note }; });
      const line = row('m-time', `${options.last ? 'Last pass' : 'One pass'}${model.passMs != null ? `, about ${num(model.passMs, 0)} ms` : ''}`);
      const blocks = stages.map((stage, index) => {
        const unmeasured = stage.computeMs == null, block = make('div', `blk${stage.failed || unmeasured ? ' hollow' : ''}${stage.trouble ? ` ${stage.trouble}` : ''}`);
        block.dataset.stage = index;
        const wire = make('span', `wire${index === stages.length - 1 ? ' back' : ''}`);
        line.append(block, wire);
        return { block, wire };
      });
      const labels = row('m-ms');
      memo.place = progress => {
        stages.forEach((stage, index) => {
          const from = x[index], to = time[index], mix = (a, b) => a + (b - a) * progress;
          const t0 = mix(from.x0, to.t0), t1 = mix(from.x1, to.t1), h1 = mix(from.x1, to.h1);
          blocks[index].block.style.left = `${t0}%`; blocks[index].block.style.width = `${t1 - t0}%`;
          blocks[index].wire.style.left = `${t1}%`; blocks[index].wire.style.width = `${h1 - t1}%`;
          bands[index].setAttribute('points', `${from.x0 * 10},0 ${from.x1 * 10},0 ${t1 * 10},100 ${t0 * 10},100`);
        });
      };
      memo.place(1);
      element.append(mimic);
      lanes = [names, labels];
      stages.forEach((stage, index) => { if ((time[index].t1 - time[index].t0) / 100 * width >= 15) blocks[index].block.textContent = stage.n; });
      const placed = [];
      for (const { stage, note } of notes) {
        // The note stays inside its band, and a band too narrow to hold it leaves the figure to the readout.
        const index = stage.n - 1, mid = ((x[index].x0 + x[index].x1) / 2 + (time[index].t0 + time[index].t1) / 2) / 2 / 100 * width;
        const room = ((x[index].x1 - x[index].x0) + (time[index].t1 - time[index].t0)) / 2 / 100 * width - 12;
        if (room < 84) { note.remove(); continue; }
        if (note.offsetWidth > room) note.style.width = `${room}px`;
        const half = note.offsetWidth / 2, at = Math.max(half, Math.min(mid, width - half));
        if (placed.some(other => Math.abs(other - at) < half * 2 + 12)) { note.remove(); continue; }
        placed.push(at);
        note.style.left = `${at}px`;
      }
      const taken = [];
      const label = (text, at, strong) => {
        const item = make('span', `t-label${strong ? ' strong' : ''}`, text);
        labels.append(item);
        const left = Math.max(0, Math.min(at, width - item.offsetWidth)), right = left + item.offsetWidth;
        if (taken.some(([a, b]) => left < b + 8 && a < right + 8)) { item.remove(); return; }
        taken.push([left, right]);
        item.style.left = `${left}px`;
      };
      const slowest = stages.find(stage => stage.slowest);
      if (slowest && stages.length > 1) label(`${num(slowest.computeMs)} ms, slowest stage`, time[slowest.n - 1].t0 / 100 * width, true);
      const links = stages.slice(0, -1).filter(stage => stage.linkMs != null);
      if (links.length) {
        const worst = links.reduce((a, b) => b.linkMs > a.linkMs ? b : a), at = time[worst.n - 1];
        label(`${num(worst.linkMs)} ms link`, (at.t1 + at.h1) / 2 / 100 * width - 28, false);
      }
    } else {
      element.append(mimic);
      memo.place = null;
      if (!model.planned) { element.append(make('p', 'm-empty', 'The time line fills in after the first reply.')); }
      lanes = [names];
    }
    placeNames(names, model, x, width);

    const foot = make('div', 'board-foot'), readout = make('p', 'readout'), key = make('div', 'key');
    readout.setAttribute('aria-live', 'polite');
    const entry = (name, build) => { const item = make('span'); const swatch = make('i'); build(swatch); item.append(swatch, document.createTextNode(name)); key.append(item); };
    for (const kind of TIER_ORDER) if (stages.some(stage => !stage.trouble && stage.tiers.some(item => item.kind === kind))) entry(TIER_NAMES[kind], swatch => { swatch.className = `tier-${kind}`; });
    if (time && stages.length > 1) {
      entry('Network link', swatch => { swatch.className = 'line'; });
      entry('Link back to the entry', swatch => { swatch.className = 'line back'; });
    }
    foot.append(readout, key);
    if (time) foot.append(make('p', 'key-note', 'The bottom line is one pass laid end to end, not the time per token, because pipelining and the draft overlap work.'));
    element.append(foot);
    memo.readout = readout;
    if (time && !memo.slid) slide(element);
  }

  // Trouble and the ends are named first, and the rest only while they fit.
  function placeNames(lane, model, x, width) {
    const stages = model.stages, last = stages[stages.length - 1];
    const first = stage => stage.trouble ? 0 : stage.moving ? 1 : stage.n === 1 || stage === last ? 2 : stage.slowest ? 3 : 4;
    const taken = [];
    for (const stage of [...stages].sort((a, b) => first(a) - first(b) || a.n - b.n)) {
      if (stages.length > 24 && first(stage) === 4) continue;
      const end = stage.n === 1 ? ', entry' : stage === last ? ', exit' : '';
      const item = make('span', `m-name${stage.trouble ? ` ${stage.trouble}` : ''}`);
      const title = make('b');
      if (stage.trouble) title.insertAdjacentHTML('beforeend', glyphHtml(stage.trouble));
      title.append(`${stage.n}  ${stage.workerId}`);
      item.append(title, `Layers ${span(stage.start, stage.end)}${stage.worker?.quarantined ? ', quarantined' : stage.trouble === 'bad' ? ', not answering' : end}`);
      lane.append(item);
      const wide = item.offsetWidth, at = x[stage.n - 1].x0 / 100 * width;
      const left = stage === last && stages.length > 1 ? width - wide : Math.max(0, Math.min(at, width - wide));
      if (taken.some(([a, b]) => left < b + 10 && a < left + wide + 10)) { item.remove(); continue; }
      taken.push([left, left + wide]);
      if (stage === last && stages.length > 1) item.classList.add('end');
      item.style.left = `${left}px`;
    }
  }

  function slide(element) {
    const memo = element.board, calm = matchMedia('(prefers-reduced-motion: reduce)').matches;
    memo.slid = true;
    if (calm || document.hidden || !memo.place) return;
    const began = performance.now(), place = memo.place;
    place(0);
    const step = now => {
      const k = Math.min((now - began) / 700, 1);
      place(1 - (1 - k) ** 3);
      if (k < 1 && memo.place === place) requestAnimationFrame(step);
    };
    requestAnimationFrame(step);
  }

  function restStage(model) {
    return model.stages.find(stage => stage.trouble === 'bad') || model.stages.find(stage => stage.trouble) || model.stages.find(stage => stage.moving) || model.stages.find(stage => stage.slowest) || model.stages[0];
  }
  function showStage(element, index) {
    const memo = element.board, model = memo.model;
    if (!model || !memo.readout) return;
    const stage = model.stages[index ?? -1] || restStage(model);
    memo.readout.textContent = '';
    memo.readout.append(make('b', '', stage.workerId), `, ${stageWords(model, stage)}.`);
    for (const item of element.querySelectorAll('.on')) item.classList.remove('on');
    if (index != null) for (const item of element.querySelectorAll(`[data-stage="${index}"]`)) item.classList.add('on');
  }
  function focusStage(element, index) {
    const memo = element.board;
    if (!memo) return;
    const buttons = [...element.querySelectorAll('button[data-stage]')];
    memo.rove = Math.max(0, Math.min(index, buttons.length - 1));
    buttons.forEach((button, at) => { button.tabIndex = at === memo.rove ? 0 : -1; });
    memo.pinned = memo.rove;
    buttons[memo.rove]?.focus();
    showStage(element, memo.rove);
  }
  function wireBoard(element) {
    element.dataset.wired = '1';
    const at = event => { const value = event.target.closest?.('[data-stage]')?.dataset.stage; return value == null ? null : Number(value); };
    element.addEventListener('pointerover', event => { const index = at(event); if (index != null) showStage(element, index); });
    element.addEventListener('pointerleave', () => showStage(element, element.board.pinned));
    element.addEventListener('click', event => { const index = at(event); if (index != null) { element.board.pinned = index; showStage(element, index); } });
    element.addEventListener('focusin', event => { const index = at(event); if (index != null) { element.board.pinned = index; showStage(element, index); } });
    element.addEventListener('focusout', event => { if (!element.contains(event.relatedTarget)) { element.board.pinned = null; showStage(element, null); } });
    element.addEventListener('keydown', event => {
      const memo = element.board, index = at(event), end = memo.model.stages.length - 1;
      if (index == null) return;
      const moves = { ArrowLeft: index - 1, ArrowRight: index + 1, Home: 0, End: end };
      if (event.key in moves) { event.preventDefault(); focusStage(element, moves[event.key]); }
      else if (event.key === 'Enter') { event.preventDefault(); memo.options.onOpen?.(memo.model.stages[index]); }
    });
  }

  /* The same facts standing up, for a phone and for the chat rail. */
  function drawRouteList(element, model, options = {}) {
    const stages = model.stages, slowest = Math.max(...stages.map(stage => stage.computeMs ?? 0)), keep = keyStages(model);
    const shorten = stages.length > 12 && !options.all;
    const worstLink = stages.slice(0, -1).reduce((a, b) => (b.linkMs ?? 0) > a ? b.linkMs : a, 0);
    const list = make('ol', 'route-list');
    list.setAttribute('aria-label', 'Route stages, entry first');
    let skipped = 0;
    const gap = () => { if (skipped) { list.append(make('li', 'route-gap', `${count(skipped, 'stage')} not shown`)); skipped = 0; } };
    for (const stage of stages) {
      if (shorten && !keep.has(stage.n)) { skipped += 1; continue; }
      gap();
      const stop = make('li', `route-stop${stage.trouble ? ` ${stage.trouble}` : ''}`);
      const who = make('span', 'route-who');
      if (stage.trouble) who.insertAdjacentHTML('beforeend', glyphHtml(stage.trouble));
      const name = make(options.plain ? 'span' : 'button', options.plain ? '' : 'name-link', stage.workerId);
      if (!options.plain) { name.type = 'button'; name.dataset.worker = stage.workerId; name.dataset.key = `r:${stage.workerId}`; }
      who.append(name);
      const note = stage.worker?.quarantined ? ', quarantined' : stage.trouble === 'bad' ? ', not answering' : stage.slowest && stages.length > 1 ? ', slowest stage' : stage.spilledBytes ? `, ${bytes(stage.spilledBytes)} below its best memory` : '';
      stop.append(make('span', 'route-no', stage.n), who, make('span', 'route-val', ms(stage.computeMs)), make('span', 'route-sub', `Layers ${span(stage.start, stage.end)}${note}${options.memory && stage.tiers.length ? `, ${options.memory(stage)}` : ''}`));
      if (stage.computeMs != null && slowest) { const bar = make('span', 'route-bar'), fill = make('i'); fill.style.width = `${stage.computeMs / slowest * 100}%`; bar.append(fill); stop.append(bar); }
      if (stage.moving) {
        const move = make('div', 'route-move', `${stage.moving.to}: ${bytes(stage.moving.bytesDone)} of ${bytes(stage.moving.bytesTotal)}`), bar = make('div', 'lay'), fill = make('i');
        fill.style.width = `${stage.moving.bytesTotal ? stage.moving.bytesDone / stage.moving.bytesTotal * 100 : 0}%`;
        bar.append(fill); move.append(bar); stop.append(move);
      }
      list.append(stop);
      const last = stage.n === stages.length;
      if (!last && stage.linkMs != null && (stage.linkMs > 2 || stage.linkMs === worstLink)) list.append(make('li', 'route-link', `${num(stage.linkMs)} ms link`));
    }
    gap();
    const sheet = make('div');
    sheet.append(list);
    if (stages.length > 12) {
      const toggle = make('button', 'link', options.all ? 'Show key stages' : 'Show every stage');
      toggle.type = 'button'; toggle.dataset.action = 'route-all'; toggle.dataset.key = 'route-all';
      sheet.append(toggle);
    }
    if (element.routeHtml === sheet.innerHTML) return;
    const focused = element.contains(document.activeElement) ? document.activeElement.dataset.key : null;
    element.routeHtml = sheet.innerHTML;
    element.replaceChildren(...sheet.childNodes);
    if (focused) element.querySelector(`[data-key="${CSS.escape(focused)}"]`)?.focus();
  }

  /* A layer rail alone, for lanes and for a plan's preview. */
  function drawRail(element, stages) {
    element.textContent = '';
    element.classList.add('rail-line');
    for (const stage of stages) { const part = make('span', 'seg-blk'); part.style.flex = `${stage.layerCount} 1 0`; fillTiers(part, stage); element.append(part); }
  }

  function sparkline(values) {
    if (values.length < 2) return '';
    const low = Math.min(...values), high = Math.max(...values), range = high - low || 1;
    const point = (value, index) => [(index / (values.length - 1) * 92 + 1).toFixed(1), (17 - (value - low) / range * 14).toFixed(1)];
    const end = point(values[values.length - 1], values.length - 1);
    return `<svg class="spark" viewBox="0 0 96 20" aria-hidden="true"><polyline points="${values.map((value, index) => point(value, index).join(',')).join(' ')}"/><circle cx="${end[0]}" cy="${end[1]}" r="2"/></svg>`;
  }

  return { routeModel, drawBoard, drawRouteList, drawRail, focusStage, keyStages, stageWords, sparkline, glyphHtml, tierOf, median, num, bytes, ms, duration, relative, clock, count, whole, span, modelName, jobRunning, TIER_ORDER, TIER_NAMES };
})();
