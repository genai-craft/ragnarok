# ragnarok — the last RAG (最後の RAG)

[English](README.md) · [日本語]

**ベクトル × 目次の木 × 遅延グラフ × 校正された確率判定。GPU 1 枚、全部ローカル。**
[openvons](https://github.com/genai-craft/openvons) (有限候補に確率で答える判定エンジン、Jev 型の校正済み選択) の上に作っています。

**デモ: https://ragnarok.openvons.com** · 動画 (2 分 40 秒、日本語ナレーション・字幕): [デモサイトの mp4](https://ragnarok.openvons.com/static/ragnarok_demo_v2.mp4) · [リリース添付](https://github.com/genai-craft/ragnarok/releases/latest)

![ragnarok デモ: 確率判定 2 段、棄却、網羅](docs/img/demo.gif)

| 候補 50 → 25+25 → 10、各 1 token | 「どれでもない」≥ 50% で棄却 | 全ページを 25 ms ずつ判定 |
|---|---|---|
| ![stages](docs/img/stages.jpg) | ![abstain](docs/img/abstain.jpg) | ![sweep](docs/img/sweep.jpg) |

## 何が新しいか

RAG はふつう、ベクトル検索か、目次の木 (PageIndex)、か知識グラフ (GraphRAG) のどれか一つを選び、それぞれ得意・不得意があります。
ragnarok は **横断にはベクトル**、**構造には木**、**網羅には事前抽出ではなく質問時の一斉判定**を使い、そのすべての「選ぶ」段を
**生成ではなく 1 token の確率判定**で行います。これで既存手法に無い 3 つが手に入ります:

1. **同じコストで高い精度** — 候補 50 を 1 token の forward 3 回で並べ替える (生成型 reranker より速い) → FinanceBench の根拠ページ的中 hit@5 **0.67 → 0.89**、回答正解率 **0.47 → 0.77**
2. **本当に効く棄却** — 「どれでもない」が校正済みの選択肢として入っている。文書に答えが無ければ無いと言う。文書が似ていても (同じ会社の別年度) 見分ける: AUROC **0.96** (埋め込み類似度は 0.73)
3. **抽出インデックス無しの網羅** — 「この話題に触れている箇所を全部」は、ページごとの yes/no 判定 (4B で 1 ページ 25 ms) → 再現率 **0.93** (ベクトル top-10 は 0.57)

## 測った数字 (全部ローカル: Qwen3-Embedding-0.6B + Qwen3.8-27B-AWQ-INT4 / Qwen3-4B on vLLM)

FinanceBench 公開セット (150 問、10-K 84 冊 12,013 ページ)。回答器と採点器は全行とも同じローカル 27B なので、絶対値ではなく行の比較で見てください。

| 検索 | 根拠 hit@1 | 根拠 hit@5 | 回答正解率 |
|---|---|---|---|
| ベクトル検索 (ページ埋め込み) | 0.36 | 0.67 | 0.47 |
| + cross-encoder rerank (bge-reranker-v2-m3、top-50) | 0.28 | 0.61 | 0.46 |
| + 生成型 listwise rerank (RankGPT 方式、27B、top-25) | 0.57 | 0.81 | 0.59 |
| PageIndex 方式 (木を見せて節 id を生成) | 0.39 | 0.58 | 0.45 |
| **ragnarok: ベクトル top-50 → 1 token 判定 2 段 (25+25 → 10 → 5)** | 0.63 | **0.89** | 0.68 |
| 同上、回答器を思考あり | 0.63 | 0.89 | **0.77** |

棄却 — 質問を*別の*文書にぶつけ、「どれでもない」の確率を見る:

| 相手の文書 | 埋め込み類似度の閾値 AUROC | ragnarok「どれでもない」確率 AUROC | none>0.5 のとき: 誤棄却 / 棄却 |
|---|---|---|---|
| 別会社の 10-K | 0.993 | 0.985 | 3% / 90% |
| **同じ会社、3 年以上離れた 10-K** | **0.731** | **0.960** | **1% / 79%** |

網羅 — 「X に触れているページを全部」(6 冊 × 8 話題、正解 = 27B の全ページ判定):

| 手法 | 再現率 | 適合率 | 判定の呼び出し |
|---|---|---|---|
| ベクトル top-10 | 0.57 | 0.48 | 0 |
| キーワード一致 | 0.80 | 0.57 | 0 |
| ベクトル top-50 → 4B yes/no | 0.82 | 0.66 | 50 |
| **全ページ → 4B yes/no (網羅モード)** | **0.93** | 0.61 | ~155 (1 冊 約 4 秒) |

全部の表、うまくいかなかったこと (10-K での木の探索、4B への rerank 蒸留)、GraphRAG / PageIndex / ベクトル RAG との向き不向き: [docs/comparison.md](docs/comparison.md)、[bench/README.md](bench/README.md)。

日本語の有価証券報告書 (EDINET、10 社 26 期、130 問の合成 QA): ベクトル hit@5 0.78 → 確率判定 2 段 **0.91**、回答正解率 **0.92** (27B 思考あり)。詳細は [bench/README.md](bench/README.md)。

## しくみ

```
質問 ──► ページ埋め込み (top-50)
     ──► 判定 ×2: 「この 25 ページのどれに答えがあるか / どれでもない」 → 各 5 ページ
     ──► 判定: 「この 10 ページのどれか / どれでもない」 → ページ + P(どれでもない)
     ──► P(どれでもない) ≥ 0.5 ? 棄却 : そのページだけで回答 (根拠付き)
網羅:  全ページ ──► 「このページは X に触れているか yes/no」(4B、25 ms) ──► 該当箇所を全部 ──► 要約
```

判定は openvons の `LLMBackend` (OpenAI 互換サーバーの guided choice + 先頭 token の logprob) で、1 回の forward で選択肢全体の確率分布が返ります。

## 使い方

```bash
git clone https://github.com/genai-craft/ragnarok && cd ragnarok
uv venv && uv pip install -e ".[demo]"
# モデルサーバー (vLLM): 判定・回答の 27B を :8310、速い判定の 4B を :8311 — GPU とモデルはスクリプト内の環境変数で
JUDGE_GPU=0 FAST_GPU=0 scripts/serve_models.sh
python - <<'PY'
import asyncio
from ragnarok.engine import Engine
eng = Engine()                                   # Qwen3-Embedding-0.6B + 上の 2 サーバー
ix = eng.index_pdf("your.pdf")                   # ページ + 埋め込み + レイアウト木 (LLM なし)
async def main():
    r = await eng.retrieve(ix, "2022 年度の売上高は？")
    print(r.pages, r.none_final, r.abstain)      # ページ (0 始まり)、P(どれでもない)、棄却か
    print(await eng.answer(ix, "2022 年度の売上高は？", r.pages, think=True))
    hits = await eng.sweep(ix, "自社株買い")       # 全ページの {page: P(yes)}
asyncio.run(main())
PY
python -m examples.demo.server --port 8608     # Web デモ
```

デモは **PDF / DOCX / PPTX / HTML / TXT / Markdown** を受け付け (PDF 以外はページの代わりに節・スライドで区切る)、判定の各段階を確率付きで表示し、根拠の PDF ページをプレビューし、回答をストリーミングし、網羅モードを持ちます。
小さい構成なら 4B だけでも全部動きます (rerank の hit@5 は 0.88 → 0.77)。`logprobs` と guided choice を返す OpenAI 互換サーバーなら何でも判定役にできます。

## 埋め込みモデル

既定は [google/embeddinggemma-2](https://huggingface.co/google/embeddinggemma-2) (Apache-2.0、100 言語以上、768d で Matryoshka 切り詰め可、**ページ画像を同じ空間に埋め込める** — 本文の無いページは画像で埋め込むので、スキャン PDF や図表が OCR なしで検索できる)。Qwen3-Embedding-0.6B との比較では英語は誤差の範囲、日本語は少し良く、候補の天井 recall@50 が 0.98 (FinanceBench) / 1.00 (有報) に上がる。`RAGNAROK_EMBED=Qwen/Qwen3-Embedding-0.6B` で戻せる。`Engine(embed_dim=256)` で記憶域を 1/4 に (精度ほぼ同じ)。 `Engine(image_pages=True)` で全ページを画像でも埋め込むと、英語の 10-K では融合スコア (text + 0.3·image) で候補 recall@5 +3pt。日本語文書は画像が足を引くので自動で 0 にする。 詳細: [bench/README.md](bench/README.md)。

**スキャン PDF**: 文字層の無いページは画像で埋め込み (英語のスキャンは OCR なしで recall@5 0.79)、`Engine(vlm_url=...)` に OpenAI 互換の VLM (vLLM の Qwen3-VL-8B、バッチで 1 頁 1 秒) を与えると索引時に OCR する。1960〜80 年代の公文書で回答正解率 0.75 (英) / 0.79 (日)。詳細: [bench/README.md](bench/README.md)。

## 複数文書 (フォルダ・NAS・データレイク)

```bash
ragnarok index /mnt/nas/share --store /data/ragnarok/store      # 再帰・差分 (更新時刻とサイズ)。PDF/DOCX/PPTX/HTML/TXT/MD
ragnarok ask   --store /data/ragnarok/store "2024年度の設備投資額は"     # 索引した全文書を横断して探す
ragnarok sweep --store /data/ragnarok/store "自社株買い"                 # 全文書で、その話題に触れているページを全部
```

全文書のページ埋め込みを 1 つの行列で持ち (20 万ページ超は faiss を自動で使う)、候補は 1 文書あたりの上限付き、判定の選択肢に文書名が入るので 1 回の判定で文書をまたいで順位付けし、コーパス全体に対して棄却できます。
110 冊 17,418 ページから文書名を教えずに探す測定: FinanceBench 文書 hit@1 0.75 → **0.87**、根拠ページ hit@5 0.53 → **0.75**; 有報 QA 0.32 → **0.64** / 0.39 → **0.65** (横断ベクトル top-5 → ragnarok)。デモにも「全文書から探す」があります。

## まだできていないこと

- 木は補助であって中核ではない: FinanceBench 型 (表の中の数値) では、全ノードに LLM 要約を付けても木の探索は埋め込みに負けた。構造が重い文書 (法令・契約) 向けに残し、別に測る
- 関係の質問 (「X と繋がる人物は」) は未測定。網羅・集約は測った。質問時に判定でエッジを確かめる遅延グラフが次の計画
- 日本語は端から端まで動く (デモに EDINET の有報 26 期と情報通信白書を同梱)。日本語 QA は出典ページから生成した合成問題で、FinanceBench より易しい

## ライセンス

Apache-2.0 — 商用利用・改変・再配布は自由。ただし**出典の記名**を: [LICENSE](LICENSE) と [NOTICE](NOTICE) を残し、「ragnarok — https://github.com/genai-craft/ragnarok」と記載してください ([CITATION.cff](CITATION.cff))。
ragnarok は TypeSafe AI 社および同社製品 Jev とは無関係です。Jev 型の校正済み判定の独立実装である openvons の上に作られています。
