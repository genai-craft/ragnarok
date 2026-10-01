(() => {
const $ = s => document.querySelector(s);
let docs = [], cur = null;
const fmt = x => (x * 100).toFixed(0) + '%';
async function loadDocs() {
  const d = await (await fetch('/api/docs')).json(); docs = d.docs;
  const sel = $('#doc'); sel.innerHTML = '';
  const groups = {}; docs.forEach(x => (groups[x.group || 'その他'] ||= []).push(x));
  Object.entries(groups).forEach(([g, arr]) => { const og = document.createElement('optgroup'); og.label = g; arr.forEach(x => { const o = document.createElement('option'); o.value = x.id; o.textContent = `${x.name} (${x.pages}p)`; og.appendChild(o); }); sel.appendChild(og); });
  pick(docs[0]?.id);
}
function pick(id) { cur = docs.find(x => x.id === id); if (!cur) return; $('#doc').value = id;
  $('#docinfo').textContent = `${cur.kind || ''} ${cur.tree ? '/ 木あり' : ''}`;
  const c = $('#samples'); c.innerHTML = ''; (cur.samples || []).forEach(s => { const b = document.createElement('span'); b.className = 'chip'; b.textContent = s.q.length > 60 ? s.q.slice(0, 60) + '…' : s.q; b.title = s.q; b.onclick = () => { $('#q').value = s.q; cur._gold = s.pages || []; }; c.appendChild(b); });
  // 棄却デモ用: 同じ会社の別年度の質問も出す
  docs.filter(x => x.group === cur.group && x.id !== cur.id).forEach(x => (x.samples || []).slice(0, 1).forEach(s => { const b = document.createElement('span'); b.className = 'chip'; b.style.borderColor = '#f85149'; b.textContent = '別年度の質問: ' + s.q.slice(0, 50) + '…'; b.title = s.q + `  (元は ${x.name})`; b.onclick = () => { $('#q').value = s.q; cur._gold = []; }; c.appendChild(b); }));
}
$('#doc').onchange = e => pick(e.target.value);
$('#file').onchange = async e => { const f = e.target.files[0]; if (!f) return; $('#status').textContent = '索引を作成中 (ページ本文 → 埋め込み → 木)…'; const fd = new FormData(); fd.append('file', f);
  const r = await (await fetch('/api/upload', { method: 'POST', body: fd })).json(); if (r.error) { $('#status').textContent = r.error; return; }
  await loadDocs(); pick(r.id); $('#status').textContent = `索引完了 ${r.pages} ページ、${r.elapsed_s.toFixed(1)} 秒`; };
async function showPage(p) { if (!cur) return; const d = await (await fetch(`/api/page?doc=${encodeURIComponent(cur.id)}&p=${p}`)).json(); $('#pno').textContent = `p.${p + 1}`; $('#psec').textContent = d.section || ''; $('#ptext').textContent = d.text; }
function cells(cands, vals, tops, gold) { const w = document.createElement('div'); w.className = 'cells';
  cands.forEach((p, i) => { const c = document.createElement('div'); c.className = 'cell' + (tops && tops.includes(p) ? ' top' : '') + (gold && gold.includes(p) ? ' gold' : ''); const v = vals ? vals[i] : 0;
    c.innerHTML = `<i style="height:${Math.max(2, v * 100).toFixed(0)}%"></i><span>p${p + 1}</span>`; c.title = `p.${p + 1}: ${vals ? fmt(v) : ''}`; c.onclick = () => showPage(p); w.appendChild(c); }); return w; }
function stage(title, t) { const s = document.createElement('div'); s.className = 'stage'; s.innerHTML = `<h4>${title}<span class="t">${t !== undefined ? t.toFixed(1) + ' s' : ''}</span></h4>`; $('#pipeline').appendChild(s); return s; }
$('#go').onclick = () => { const mode = document.querySelector('input[name=mode]:checked').value; if (mode === 'ask') ask(); else sweep(); };
function ask() {
  const q = $('#q').value.trim(); if (!q || !cur) return; const gold = cur._gold || [];
  $('#pipeline').innerHTML = ''; $('#answer').hidden = true; $('#go').disabled = true; $('#status').textContent = '実行中…';
  const es = new EventSource(`/api/ask?doc=${encodeURIComponent(cur.id)}&q=${encodeURIComponent(q)}&think=${$('#think').checked ? 1 : 0}&fast=${$('#fast').checked ? 1 : 0}`);
  es.onmessage = ev => { const d = JSON.parse(ev.data);
    if (d.stage === 'emb') { const s = stage(`① ベクトル検索 top-50 <span class="muted">(緑枠 = 正解ページ)</span>`, d.t); s.appendChild(cells(d.candidates.map(c => c.p), d.candidates.map(c => Math.max(0, c.sim)), null, gold)); }
    else if (d.stage.startsWith('stage1')) { const s = stage(`② 確率判定 ${d.stage.replace('stage1-', '')}/2 — 25 ページを 1 token で採点、上位 5 (橙枠) を次へ。どれでもない ${fmt(d.none)}`, d.t);
      const tops = d.candidates.map((p, i) => [p, d.probs[i]]).sort((a, b) => b[1] - a[1]).slice(0, 5).map(x => x[0]); s.appendChild(cells(d.candidates, d.probs, tops, gold)); }
    else if (d.stage === 'final') { const s = stage(`③ 最終判定 — 残り ${d.candidates.length} ページ + どれでもない` + (d.abstain ? ' <span class="badge abst">棄却: この文書には無い</span>' : ' <span class="badge ok">根拠あり</span>'), d.t);
      const b = document.createElement('div'); b.className = 'bars'; const arr = d.candidates.map((p, i) => [p, d.probs[i]]).sort((x, y) => y[1] - x[1]);
      arr.forEach(([p, v]) => { const r = document.createElement('div'); r.className = 'bar'; r.innerHTML = `<span>p.${p + 1}${gold.includes(p) ? ' ✔' : ''}</span><div class="track"><div class="fill" style="width:${(v * 100).toFixed(1)}%"></div></div><span class="v">${fmt(v)}</span>`; r.onclick = () => showPage(p); b.appendChild(r); });
      const r = document.createElement('div'); r.className = 'bar none'; r.innerHTML = `<span>どれでもない</span><div class="track"><div class="fill" style="width:${(d.none * 100).toFixed(1)}%"></div></div><span class="v">${fmt(d.none)}</span>`; b.appendChild(r); s.appendChild(b);
      if (d.pages?.length) showPage(d.pages[0]); }
    else if (d.stage === 'answer') { const a = $('#answer'); a.hidden = false; a.innerHTML = (d.abstained ? '<b style="color:#f85149">棄却</b> — ' : '<b>回答</b> — ') + d.text.replace(/page\s+(\d+)/gi, (m, n) => `<span class="cite" data-p="${n - 1}">page ${n}</span>`).replace(/(\d+)\s*ページ/g, (m, n) => `<span class="cite" data-p="${n - 1}">${n} ページ</span>`); a.querySelectorAll('.cite').forEach(c => c.onclick = () => showPage(+c.dataset.p)); }
    else if (d.stage === 'done') { es.close(); $('#go').disabled = false; $('#status').textContent = `完了 ${d.t.toFixed(1)} 秒`; } };
  es.onerror = () => { es.close(); $('#go').disabled = false; $('#status').textContent = 'エラー'; };
}
function sweep() {
  const q = $('#q').value.trim(); if (!q || !cur) return; $('#pipeline').innerHTML = ''; $('#answer').hidden = true; $('#go').disabled = true; $('#status').textContent = '全ページを判定中…';
  const s = stage(`網羅: 全 ${cur.pages} ページに「${q}」に触れているか yes/no 判定 (4B)`); const prog = document.createElement('div'); prog.className = 'prog'; prog.innerHTML = '<i style="width:0"></i>'; s.appendChild(prog);
  const grid = document.createElement('div'); grid.className = 'grid'; const cellsArr = []; for (let p = 0; p < cur.pages; p++) { const b = document.createElement('b'); b.title = 'p.' + (p + 1); b.onclick = () => showPage(p); grid.appendChild(b); cellsArr.push(b); } s.appendChild(grid);
  const es = new EventSource(`/api/sweep?doc=${encodeURIComponent(cur.id)}&topic=${encodeURIComponent(q)}`);
  es.onmessage = ev => { const d = JSON.parse(ev.data);
    if (d.stage === 'sweep') { prog.firstChild.style.width = (d.progress / d.total * 100) + '%'; Object.entries(d.probs).forEach(([p, v]) => { cellsArr[+p].style.background = v >= 0.5 ? `rgba(63,185,80,${0.4 + v * 0.6})` : `rgba(88,166,255,${v * 0.5})`; cellsArr[+p].title = `p.${+p + 1}: ${fmt(v)}`; }); }
    else if (d.stage === 'hits') { const h = stage(`該当ページ ${d.pages.length} 件`, d.t); const b = document.createElement('div'); b.className = 'bars'; d.pages.forEach(p => { const r = document.createElement('div'); r.className = 'bar'; r.innerHTML = `<span>p.${p + 1}</span><div class="track"></div><span class="v muted small">${(d.sections[p] || '').slice(0, 40)}</span>`; r.onclick = () => showPage(p); b.appendChild(r); }); h.appendChild(b); }
    else if (d.stage === 'summary') { const a = $('#answer'); a.hidden = false; a.innerHTML = '<b>要約</b> — ' + d.text.replace(/page\s+(\d+)/gi, (m, n) => `<span class="cite" data-p="${n - 1}">page ${n}</span>`); a.querySelectorAll('.cite').forEach(c => c.onclick = () => showPage(+c.dataset.p)); }
    else if (d.stage === 'done') { es.close(); $('#go').disabled = false; $('#status').textContent = `完了 ${d.t.toFixed(1)} 秒`; } };
  es.onerror = () => { es.close(); $('#go').disabled = false; $('#status').textContent = 'エラー'; };
}
loadDocs();
})();
