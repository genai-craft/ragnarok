"""デモを Playwright で自動操作して録画し、各シーンの開始時刻を timeline.json に残す (ナレーションの尺に合わせて待つ)。
使い方: python examples/demo/record_video.py  (デモ :8608 が動いていること)。出力 /data/ragnarok/video/raw/*.webm と timeline.json"""
import asyncio, json, os, time
from playwright.async_api import async_playwright
V = "/data/ragnarok/video"; NAR = {n["scene"]: n for n in json.load(open(f"{V}/narration.json"))}
BASE = "http://127.0.0.1:8608"
tl = []; T0 = None
def mark(scene):
    tl.append({"scene": scene, "t": round(time.time() - T0, 2)}); print(f"{tl[-1]['t']:6.1f}s {scene}", flush=True)
async def hold(scene, extra=0.5):
    await asyncio.sleep(NAR[scene]["dur"] + extra)
async def main():
    global T0
    os.makedirs(f"{V}/raw", exist_ok=True)
    async with async_playwright() as p:
        b = await p.chromium.launch()
        ctx = await b.new_context(viewport={"width": 1440, "height": 900}, record_video_dir=f"{V}/raw", record_video_size={"width": 1440, "height": 900}, locale="ja-JP")
        pg = await ctx.new_page(); T0 = time.time()
        # title
        await pg.goto(f"{BASE}/static/title.html"); mark("title"); await hold("title")
        # pin
        await pg.goto(f"{BASE}/"); await pg.wait_for_selector("#doc option", state="attached"); await asyncio.sleep(0.8)
        await pg.select_option("#doc", "3M_2022_10K"); await pg.uncheck("#think"); mark("pin1")
        await pg.fill("#q", ""); await pg.type("#q", "What was 3M's total net sales in 2022, and how did it change from 2021?", delay=35); await hold("pin1", 0)
        await pg.click("#go"); await pg.wait_for_selector(".stage"); mark("pin2"); await hold("pin2", 0)
        await pg.wait_for_selector(".cell.top", timeout=120000); mark("pin3"); await hold("pin3", 0)
        await pg.wait_for_selector("#answer:not([hidden])", timeout=180000); mark("pin4"); await hold("pin4")
        # abstain
        await pg.fill("#q", ""); mark("abst1"); await pg.type("#q", "What is the FY2018 capital expenditure amount (in USD millions) for 3M?", delay=30); await pg.click("#go"); await hold("abst1", 0)
        await pg.wait_for_selector("#answer:not([hidden])", timeout=180000); await pg.wait_for_selector(".badge.abst", timeout=5000); mark("abst2"); await hold("abst2")
        # sweep
        await pg.check("input[name=mode][value=sweep]"); await pg.fill("#q", ""); mark("sweep1"); await pg.type("#q", "share repurchases or buybacks of the company's own stock", delay=30); await pg.click("#go"); await hold("sweep1", 0)
        await pg.wait_for_selector(".bars", timeout=300000); mark("sweep2"); await pg.wait_for_selector("#answer:not([hidden])", timeout=300000); await hold("sweep2")
        # japanese
        await pg.check("input[name=mode][value=ask]"); await pg.select_option("#doc", "soumu_r7"); await pg.fill("#q", ""); mark("ja1")
        await pg.type("#q", "生成AIの利用率は国別にどのくらいか", delay=60); await pg.click("#go"); await hold("ja1", 0)
        await pg.wait_for_selector("#answer:not([hidden])", timeout=180000); mark("ja2"); await hold("ja2", 2.5)
        # outro
        await pg.goto(f"{BASE}/static/outro.html"); mark("outro"); await hold("outro", 1.5)
        mark("end")
        await ctx.close(); await b.close()
    json.dump(tl, open(f"{V}/timeline.json", "w"), ensure_ascii=False, indent=1)
asyncio.run(main())
