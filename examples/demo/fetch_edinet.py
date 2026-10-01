"""EDINET API v2 から有価証券報告書 (docTypeCode 120) の PDF を取る。キーは ~/.config/ragnarok/edinet_key か EDINET_API_KEY (表示しない)。
対象会社名 (filerName の部分一致) と提出日の範囲を走査して、該当書類を /data/ragnarok/edinet/<会社>_<期末>.pdf に保存する。"""
import datetime as dt, json, os, re, sys, time
import httpx
KEY = os.environ.get("EDINET_API_KEY") or open(os.path.expanduser("~/.config/ragnarok/edinet_key")).read().strip()
OUT = "/data/ragnarok/edinet"; os.makedirs(OUT, exist_ok=True)
TARGETS = ["トヨタ自動車", "ソニーグループ", "任天堂", "キーエンス", "ファーストリテイリング", "ソフトバンクグループ", "日本電信電話", "信越化学工業", "リクルートホールディングス", "三菱商事"]
# 走査する提出日: 3 月決算は 6 月後半、8 月決算 (ファーストリテイリング) は 11 月後半
RANGES = [(y, 6, 16, 30) for y in (2024, 2025, 2026)] + [(y, 11, 18, 30) for y in (2023, 2024, 2025)]
C = httpx.Client(timeout=60, verify=False)
def get(url, **params):
    for k in range(4):
        try:
            r = C.get(url, params={**params, "Subscription-Key": KEY}); 
            if r.status_code == 200: return r
        except Exception:
            pass
        time.sleep(2 * (k + 1))
    raise RuntimeError(f"fail {url} {params}")
found = {}
for y, m, d0, d1 in RANGES:
    for d in range(d0, d1 + 1):
        day = dt.date(y, m, d)
        if day.weekday() >= 5 or day > dt.date.today(): continue
        res = get("https://api.edinet-fsa.go.jp/api/v2/documents.json", date=day.isoformat(), type=2).json().get("results") or []
        for x in res:
            if x.get("docTypeCode") != "120" or x.get("withdrawalStatus") not in (None, "0"): continue
            name = x.get("filerName") or ""
            for t in TARGETS:
                if t in name and x.get("pdfFlag") == "1":
                    key = (t, x.get("periodEnd"))
                    found.setdefault(key, {"docID": x["docID"], "filer": name, "periodEnd": x.get("periodEnd"), "submit": day.isoformat(), "docDescription": x.get("docDescription")})
        time.sleep(0.3)
print(len(found), "件の有報を発見", flush=True)
meta = []
for (t, pe), x in sorted(found.items()):
    fy = (pe or "")[:4]
    path = f"{OUT}/{t}_{pe}.pdf"
    if not os.path.exists(path):
        r = get(f"https://api.edinet-fsa.go.jp/api/v2/documents/{x['docID']}", type=2)
        if "pdf" not in r.headers.get("content-type", "") and not r.content[:4] == b"%PDF":
            print("  PDF なし", t, pe, r.headers.get("content-type")); continue
        open(path, "wb").write(r.content); time.sleep(0.5)
    meta.append({**x, "company": t, "file": path, "bytes": os.path.getsize(path)})
    print(f"  {t:14s} 期末 {pe} 提出 {x['submit']} {x['docID']} {os.path.getsize(path)/1e6:.1f}MB", flush=True)
json.dump(meta, open(f"{OUT}/meta.json", "w"), ensure_ascii=False, indent=1)
