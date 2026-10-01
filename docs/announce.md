# 告知文 (2026-10-02)

## X (日本語)

ragnarok — "the last RAG" を公開しました。
ベクトル × 目次の木 × 遅延グラフ × 確率判定を 1 つのエンジンに。GPU 1 枚、全部ローカル。

FinanceBench 150 問 (同じローカル 27B で比較):
・根拠ページの的中 hit@5 0.67 → 0.88
・回答正解率 0.47 → 0.73
・「文書に無い」の棄却 AUROC 0.96 (埋め込みは 0.73)
・網羅 (触れている箇所を全部) 再現率 0.93 (ベクトル top-10 は 0.57)

肝は、候補から「選ぶ」段を生成でなく 1 token の確率判定にしたこと。速くて、確率と「どれでもない」が出る。
日本語の有価証券報告書 (EDINET、10 社 26 期) もそのまま動きます。

デモ https://ragnarok.openvons.com
GitHub https://github.com/genai-craft/ragnarok
(動画 2 分 40 秒を添付)

## X (English)

ragnarok — "the last RAG" is out. Vector × tree × graph-lite × calibrated 1-token decisions, fully local on one GPU.

FinanceBench (150 q, same local 27B answerer): evidence hit@5 0.67 → 0.88, answer accuracy 0.47 → 0.73, abstention AUROC 0.96 (vs 0.73 for embedding similarity), coverage recall 0.93 (vs 0.57 for vector top-10).

The trick: every "choose among candidates" step is a 1-token probability decision instead of generation — faster than a generative re-ranker, and you get calibrated probabilities plus a real "none of these".

Demo https://ragnarok.openvons.com · Code https://github.com/genai-craft/ragnarok

## X スレッド用の補足 (2〜4 本目)

2/ 正直に書くと、PageIndex 型の「ベクトル不要の木」は FinanceBench 型 (表の中の数値) では埋め込みに負けました (要約を付けても 0.31 vs 0.67)。木は構造の重い文書向けの補助に降格。負けた数字も README と docs/comparison.md に全部載せています。

3/ 一番効いたのはシンプルで、埋め込み top-50 → 25 ずつ 1 forward で採点 → 10 に絞って再採点、の 3 回。cross-encoder (bge-reranker-v2-m3) は 10-K の表ページに弱く 0.61、生成型 listwise rerank (RankGPT 方式) は 0.81。確率判定 2 段が 0.88。

4/ 棄却の見せ場: 同じ会社の 3 年離れた 10-K に質問をぶつけると、埋め込み類似度では見分けられない (AUROC 0.73) が、「どれでもない」の確率は 0.96。誤棄却 1% で 79% を弾きます。有報 (5 期分の表がある) では別の質問設計が要る、という注意も README に。

5/ 4B だけでも動きます (hit@5 0.77)。凍結 4B + 学習 head で 27B 並みにする蒸留は失敗 (valid 0.10)。長い選択肢には openvons の head 方式が向かない。これも書いてあります。
