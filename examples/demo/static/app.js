(() => {
const $ = s => document.querySelector(s);
let docs = [], cur = null;
const fmt = x => (x * 100).toFixed(0) + '%';
async function loadDocs() {
  const d = await (await fetch('/api/docs')).json(); docs = d.docs;
  const sel = $('#doc'); sel.innerHTML = '';
  const all = document.createElement('option'); all.value = '*'; all.textContent = `★ 全文書から探す (${docs.length} 冊、データレイク)`; sel.appendChild(all);
  const groups = {}; docs.forEach(x => (groups[x.group || 'その他'] ||= []).push(x));
  Object.entries(groups).forEach(([g, arr]) => { const og = document.createElement('optgroup'); og.label = g; arr.forEach(x => { const o = document.createElement('option'); o.value = x.id; o.textContent = `${x.name} (${x.pages}p)`; og.appendChild(o); }); sel.appendChild(og); });
  pick(docs[0]?.id);
}
function pick(id) {
  if (id === '*') { cur = { id: '*', name: '全文書', pages: docs.reduce((a, x) => a + x.pages, 0), corpus: true, samples: [{ q: 'What was 3M\'s total net sales in 2022?', pages: [] }, { q: '任天堂の2026年3月期の売上高はいくらか', pages: [] }, { q: 'Which company discusses share repurchases in its 10-K?', pages: [] }] }; $('#doc').value = '*';
    $('#docinfo').textContent = `${docs.length} 冊 ${cur.pages} ページを横断。候補は 1 文書あたり最大 10 ページ`; const c = $('#samples'); c.innerHTML = ''; cur.samples.forEach(s => { const b = document.createElement('span'); b.className = 'chip'; b.textContent = s.q; b.onclick = () => { $('#q').value = s.q; cur._gold = []; }; c.appendChild(b); }); return; }
  cur = docs.find(x => x.id === id); if (!cur) return; $('#doc').value = id;
  $('#docinfo').textContent = `${cur.kind || ''} ${cur.tree ? '/ 木あり' : ''} ${cur.pdf ? '/ PDF プレビュー可' : ''}`;
  const c = $('#samples'); c.innerHTML = ''; (cur.samples || []).forEach(s => { const b = document.createElement('span'); b.className = 'chip'; b.textContent = s.q.length > 60 ? s.q.slice(0, 60) + '…' : s.q; b.title = s.q; b.onclick = () => { $('#q').value = s.q; cur._gold = s.pages || []; }; c.appendChild(b); });
  // 棄却デモ用: 同じ会社の別年度の質問も出す
  docs.filter(x => x.group === cur.group && x.id !== cur.id).forEach(x => (x.samples || []).slice(0, 1).forEach(s => { const b = document.createElement('span'); b.className = 'chip'; b.style.borderColor = '#f85149'; b.textContent = '別年度の質問: ' + s.q.slice(0, 50) + '…'; b.title = s.q + `  (元は ${x.name})`; b.onclick = () => { $('#q').value = s.q; cur._gold = []; }; c.appendChild(b); }));
}
$('#doc').onchange = e => pick(e.target.value);
$('#file').onchange = async e => { const f = e.target.files[0]; if (!f) return; $('#status').textContent = '索引を作成中 (ページ本文 → 埋め込み → 木)…'; const fd = new FormData(); fd.append('file', f);
  startTick('索引を作成中 (本文 → 埋め込み → 木)');
  const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json(); if (r.error) { stopTick(r.error); return; }
  await loadDocs(); pick(r.id); stopTick(`索引完了 ${r.pages} ${r.unit === 'slide' ? 'スライド' : r.unit === 'section' ? '節' : 'ページ'}、${r.elapsed_s.toFixed(1)} 秒`); };
let viewMode = 'image';
let shown = { doc: null, p: null };
async function showPage(p, docId) { if (!cur) return; const id = docId || (cur.corpus ? null : cur.id); if (!id) return; const meta = docs.find(x => x.id === id) || cur; shown = { doc: id, p };
  const d = await (await fetch(`/api/page?doc=${encodeURIComponent(id)}&p=${p}`)).json();
  $('#pno').textContent = (cur.corpus ? meta.name.slice(0, 24) + ' ' : '') + (meta.unit === 'slide' ? 'slide ' : 'p.') + (p + 1); $('#psec').textContent = d.section || ''; $('#ptext').textContent = d.text;
  const img = $('#pimg'); if (meta.pdf && viewMode === 'image') { img.hidden = false; $('#ptext').hidden = true; img.src = `/api/page_image?doc=${encodeURIComponent(id)}&p=${p}`; } else { img.hidden = true; $('#ptext').hidden = false; }
  $('#pview').hidden = !meta.pdf; }
document.querySelectorAll('#pview button').forEach(b => b.onclick = () => { viewMode = b.dataset.v; document.querySelectorAll('#pview button').forEach(x => x.classList.toggle('on', x === b)); if (shown.doc !== null) showPage(shown.p, shown.doc); });
$('#pimg').onclick = () => { if (shown.doc === null) return; $('#lbimg').src = `/api/page_image?doc=${encodeURIComponent(shown.doc)}&p=${shown.p}&zoom=3`; $('#lightbox').hidden = false; };
$('#lightbox').onclick = () => { $('#lightbox').hidden = true; };
let tick = null; function startTick(label) { const el = $('#status'); const t0 = performance.now(); clearInterval(tick); tick = setInterval(() => { el.innerHTML = `<span class="spin"></span>${label} ${((performance.now() - t0) / 1000).toFixed(0)} 秒`; }, 200); }
function stopTick(msg) { clearInterval(tick); $('#status').textContent = msg; }
function cells(cands, vals, tops, gold, labels, docsOf) { const w = document.createElement('div'); w.className = 'cells' + (labels ? ' wide' : '');
  cands.forEach((p, i) => { const c = document.createElement('div'); c.className = 'cell' + (tops && tops.includes(p) ? ' top' : '') + (gold && gold.includes(p) ? ' gold' : ''); const v = vals ? vals[i] : 0;
    c.innerHTML = `<i style="height:${Math.max(2, v * 100).toFixed(0)}%"></i><span>${labels ? labels[i] : 'p' + (p + 1)}</span>`; c.title = `${labels ? labels[i] : 'p.' + (p + 1)}: ${vals ? fmt(v) : ''}`; c.onclick = () => showPage(p, docsOf ? docsOf[i] : null); w.appendChild(c); }); return w; }
function stage(title, t) { const s = document.createElement('div'); s.className = 'stage'; s.innerHTML = `<h4>${title}<span class="t">${t !== undefined ? t.toFixed(1) + ' s' : ''}</span></h4>`; $('#pipeline').appendChild(s); return s; }
$('#go').onclick = () => { const mode = document.querySelector('input[name=mode]:checked').value; if (mode === 'ask') ask(); else sweep(); };
function ask() {
  const q = $('#q').value.trim(); if (!q || !cur) return; const gold = cur._gold || [];
  $('#pipeline').innerHTML = ''; $('#answer').hidden = true; $('#go').disabled = true; startTick('検索中');
  const url = cur.corpus ? `/api/ask_all?q=${encodeURIComponent(q)}&think=${$('#think').checked ? 1 : 0}` : `/api/ask?doc=${encodeURIComponent(cur.id)}&q=${encodeURIComponent(q)}&think=${$('#think').checked ? 1 : 0}&fast=${$('#fast').checked ? 1 : 0}`;
  const es = new EventSource(url);
  const lab = (c) => c.label || ('p' + (c.p + 1));
  es.onmessage = ev => { const d = JSON.parse(ev.data);
    if (d.stage === 'emb') { const s = stage(d.corpus ? `① ベクトル検索 top-50 — ${d.docs} 文書を横断 (1 文書あたり最大 10 ページ)` : `① ベクトル検索 top-50 <span class="muted">(緑枠 = 正解ページ)</span>`, d.t);
      s.appendChild(cells(d.candidates.map(c => c.p), d.candidates.map(c => Math.max(0, c.sim)), null, gold, d.corpus ? d.candidates.map(lab) : null, d.corpus ? d.candidates.map(c => c.doc) : null)); }
    else if (d.stage.startsWith('stage1')) { const s = stage(`② 確率判定 ${d.stage.replace('stage1-', '')}/2 — 25 ページを 1 token で採点、上位 5 (橙枠) を次へ。どれでもない ${fmt(d.none)}`, d.t);
      const ps = d.corpus ? d.candidates.map(c => c.p) : d.candidates; const idx = ps.map((p, i) => i).sort((a, b) => d.probs[b] - d.probs[a]).slice(0, 5); const tops = d.corpus ? [] : idx.map(i => ps[i]);
      const w = cells(ps, d.probs, tops, gold, d.corpus ? d.candidates.map(lab) : null, d.corpus ? d.candidates.map(c => c.doc) : null); if (d.corpus) idx.forEach(i => w.children[i].classList.add('top')); s.appendChild(w); }
    else if (d.stage === 'final') { const s = stage(`③ 最終判定 — 残り ${d.candidates.length} ページ + どれでもない` + (d.abstain ? ' <span class="badge abst">棄却: ' + (d.corpus ? 'どの文書にも無い' : 'この文書には無い') + '</span>' : ' <span class="badge ok">根拠あり</span>'), d.t);
      const b = document.createElement('div'); b.className = 'bars'; const arr = d.candidates.map((c, i) => [c, d.probs[i]]).sort((x, y) => y[1] - x[1]);
      arr.forEach(([c, v]) => { const p = d.corpus ? c.p : c; const r = document.createElement('div'); r.className = 'bar'; r.innerHTML = `<span>${d.corpus ? lab(c) : 'p.' + (p + 1)}${!d.corpus && gold.includes(p) ? ' ✔' : ''}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div><span class="v">${fmt(v)}</span>`; r.onclick = () => showPage(p, d.corpus ? c.doc : null); b.appendChild(r); });
      const r = document.createElement('div'); r.className = 'bar none'; r.innerHTML = `<span>どれでもない</span><div class="track"><div class="fill" style="width:${(d.none * 100).toFixed(1)}%"></div></div><span class="v">${fmt(d.none)}</span>`; b.appendChild(r); s.appendChild(b);
      if (d.corpus && d.hits?.length) showPage(d.hits[0].p, d.hits[0].doc); else if (d.pages?.length) showPage(d.pages[0]); }
    else if (d.stage === 'answer_start') { const a = $('#answer'); a.hidden = false; a.dataset.raw = ''; a.innerHTML = `<div class="gen"><span class="spin"></span><b>回答を生成中</b> — 根拠 ${(d.labels || d.pages.map(p => 'p.' + (p + 1))).join(', ')} を 27B に渡しています${d.think ? ' (まず考えてから書きます)' : ''}<span class="muted" id="genmeta"></span></div><div id="anstext"></div>`; startTick(d.think ? '回答を考え中' : '回答を生成中'); }
    else if (d.stage === 'answer_think') { const m = $('#genmeta'); if (m) m.textContent = ` — 思考 ${d.chars} 文字…`; }
    else if (d.stage === 'answer_delta') { const a = $('#answer'); a.dataset.raw += d.delta; const t = $('#anstext'); if (t) t.textContent = a.dataset.raw; const m = $('#genmeta'); if (m) m.textContent = ' — 書いています…'; }
    else if (d.stage === 'answer') { const a = $('#answer'); a.hidden = false; a.innerHTML = (d.abstained ? '<b style="color:#f85149">棄却</b> — ' : '<b>回答</b> — ') + d.text.replace(/page\s+(\d+)/gi, (m, n) => `<span class="cite" data-p="${n - 1}">page ${n}</span>`).replace(/(\d+)\s*ページ/g, (m, n) => `<span class="cite" data-p="${n - 1}">${n} ページ</span>`); a.querySelectorAll('.cite').forEach(c => c.onclick = () => showPage(+c.dataset.p)); }
    else if (d.stage === 'done') { es.close(); $('#go').disabled = false; stopTick(`完了 ${d.t.toFixed(1)} 秒`); } };
  es.onerror = () => { es.close(); $('#go').disabled = false; stopTick('エラー'); };
}
function sweep() {
  const q = $('#q').value.trim(); if (!q || !cur) return; $('#pipeline').innerHTML = ''; $('#answer').hidden = true; $('#go').disabled = true; startTick('全ページを判定中');
  const s = stage(`網羅: 全 ${cur.pages} ページに「${q}」に触れているか yes/no 判定 (4B)`); const prog = document.createElement('div'); prog.className = 'prog'; prog.innerHTML = '<i style="width:0"></i>'; s.appendChild(prog);
  const grid = document.createElement('div'); grid.className = 'grid'; const cellsArr = []; for (let p = 0; p < cur.pages; p++) { const b = document.createElement('b'); b.title = 'p.' + (p + 1); b.onclick = () => showPage(p); grid.appendChild(b); cellsArr.push(b); } s.appendChild(grid);
  const es = new EventSource(`/api/sweep?doc=${encodeURIComponent(cur.id)}&topic=${encodeURIComponent(q)}`);
  es.onmessage = ev => { const d = JSON.parse(ev.data);
    if (d.stage === 'sweep') { prog.firstChild.style.width = (d.progress / d.total * 100) + '%'; Object.entries(d.probs).forEach(([p, v]) => { cellsArr[+p].style.background = v >= 0.5 ? `rgba(63,185,80,${0.5 + v * 0.5})` : `rgb(${31 + Math.round(v * 60)},${41 + Math.round(v * 80)},${55 + Math.round(v * 120)})`; cellsArr[+p].title = `p.${+p + 1}: ${fmt(v)}`; }); }
    else if (d.stage === 'hits') { startTick('要約を生成中'); const h = stage(`該当ページ ${d.pages.length} 件`, d.t); const b = document.createElement('div'); b.className = 'bars'; d.pages.forEach(p => { const r = document.createElement('div'); r.className = 'hit'; r.innerHTML = `<b>p.${p + 1}</b><span class="muted">${(d.sections[p] || '').replace(/ > /g, ' › ')}</span>`; r.onclick = () => showPage(p); b.appendChild(r); }); h.appendChild(b); }
    else if (d.stage === 'summary') { const a = $('#answer'); a.hidden = false; a.innerHTML = '<b>要約</b> — ' + d.text.replace(/page\s+(\d+)/gi, (m, n) => `<span class="cite" data-p="${n - 1}">page ${n}</span>`); a.querySelectorAll('.cite').forEach(c => c.onclick = () => showPage(+c.dataset.p)); }
    else if (d.stage === 'done') { es.close(); $('#go').disabled = false; stopTick(`完了 ${d.t.toFixed(1)} 秒`); } };
  es.onerror = () => { es.close(); $('#go').disabled = false; stopTick('エラー'); };
}
loadDocs();
})();
