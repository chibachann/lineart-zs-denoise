# 線画のゼロショットノイズ除去 — 先行研究サーベイ

調査日 2026-11-15。対象は単一画像デノイズ、漫画・イラスト・スキャン線画の復元、および関連する自己教師あり手法。研究計画の参考文献を起点に、2026 年までの関連研究を会議録・著者論文・公式実装で確認した。以下の「本研究との違い」と Research Gaps は、各文献の手法と評価条件を [研究計画](plan.html) に照らした整理である。

## 本研究の設定

テスト時に得られるのはノイズ付きの線画 1 枚。外部の学習画像や事前学習済みの重みを使わず、元の解像度で細い筆線、その濃さとつながり、平らな塗りを保存する。合成データの正解は評価と原因分析にのみ使う。計画上の問いは、Zero-Shot Noise2Noise（ZS-N2N）の固定 2×2 間引きが線幅・向きに依存する劣化を生むか（G1）、その劣化を軽量な改良で抑えられるか（G2）、紙目やスキャンむら、JPEG 劣化にどこまで拡張できるか（G3）である。

## 1. ZS-N2N と単一画像デノイズの後続・関連手法

| 手法・出典 | 要点 | 本研究との違い |
| --- | --- | --- |
| **ZS-N2N**（Mansour & Heckel、CVPR 2023） | 画像 1 枚を非重複 2×2 ブロックに分け、互いに異なる対角上の画素平均から擬似ペアを作る。約 2.2 万パラメータの残差 CNN を、双方向のペア予測とフル解像度出力の間引き整合性で学習する。画素間で独立なゼロ平均ノイズと、局所的に近い信号を前提とする。[原著 §3–4・付録](https://arxiv.org/html/2303.11253) | 基準手法。原著の主な評価は自然写真・カメラ画像・顕微鏡画像。本研究では、ノイズが独立でも細線の幾何だけでペアの信号差が生じる点を検証する。 |
| **ZS-N2M / Zero-Shot Noise2Mean**（Liu ほか、AAAI 2025） | 固定対角平均で作った 2 枚の値を位置ごとのランダムマスクで入れ替え、各入力からペア平均を予測する。第 2 段では、元画像のブロック平均から、第 1 段の復元出力のブロック平均を予測する。[原著 §2–3](https://ojs.aaai.org/index.php/AAAI/article/download/32575/34730)、[公式コード](https://github.com/Doyle59217/ZS-N2M) | 擬似ペア間の信号差を直接扱う最も近い比較対象。平均を使えば同一ペアのクリーン成分間の二乗差は 1/4 になるが、位置ごとの入れ替えは元の対角分割を保つ。方向ごとに画素の組み方を変える M1・M2 は、学習前の幾何のずれそのものを検証できる。 |
| **Zero-Shot CFC**（Jiang ほか、BMVC 2025） | 単一画像をフーリエ領域で複数帯域に分け、帯域をまたいで位置や内容が揃うテクスチャを手掛かりに、軽量ネットを学習する。主な評価対象は実写のノイズ。参照した会議録・著者版からは公式実装を確認できなかった。[会議録](https://bmvc2025.bmva.org/proceedings/577/)、[著者版 §3–4](https://arxiv.org/html/2510.12646v1) | 空間間引きによる信号差を回避する比較軸。微細な筆線が帯域間でどれだけ一貫して見えるか、紙目や JPEG の周期成分がその手掛かりと紛れるかは、線画で別途測る必要がある。 |
| **LoTA-N2N**（Hu ほか、arXiv 2026、プレプリント） | 相補的な部分画像で初期学習し、固定した第 1 段の出力からクリーン部分画像の近似を作る。第 2 段では、擬似ターゲット残差と予測誤差の交差項をパッチ単位で推定して抑え、符号の異なる領域が全体平均で相殺される問題を扱う。[原著 §3–5](https://arxiv.org/html/2607.24135v1) | M1・M2 がペアの作り方を変えるのに対し、こちらはペアから得る学習目的を補正する。線の周りで誤差を調べる E2・M3 に近い診断軸。自然画像や合成劣化を加えた顕微鏡・X 線画像の結果から、スキャン線画での線保存は判断できない。 |
| **Median2Median**（J. Wang & G. Wang、arXiv 2025、プレプリント） | 局所パッチから複数方向の補間候補を作り、中央値付近の値を選別して 2 つの擬似観測へランダムに割り当てる。主に方向性のある構造化ノイズをシミュレーションで検証した。[原著 §3–4](https://arxiv.org/html/2510.01666) | M6・E5 の候補。方向を使う発想は M2 と重なる。真の 1 px 線が少数の補間方向にしか現れないと、線自体が中央値から遠い値として捨てられる可能性がある。 |
| **ZS-HPD**（Zhao ほか、CVPR 2026） | 局所勾配の強度で画素候補を順位付け・グループ化する間引き器と、画像全体から類似画素を探すガウス分布に基づくランダムサンプラーを併用する。低周波帯と高周波帯に異なる重みを掛けるスペクトル損失で、小型ネットを画像ごとに学習する。[原著 §3–4](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_Zero-Shot_Image_Denoising_via_Hybrid_Prior-Guided_Pseudo_Sample_Generation_CVPR_2026_paper.pdf) | 画像内容に応じた間引きと細部保護を既に扱うため M2・M3 の重要な比較対象。本研究の検証軸は筆線の接線方向、幅、格子位相、および骨格の保存であり、原著の一般画像における細部・テクスチャの評価を補う。 |
| **Positive2Negative**（Li ほか、CVPR 2025） | 推定残差からゼロ平均かつほぼ対称な正負のノイズを作り、再汚染画像の復元出力に整合性を課す。元の解像度のまま学習し、間引きやマスクによる情報欠落を避ける設計。[原著 §3](https://openaccess.thecvf.com/content/CVPR2025/papers/Li_Positive2Negative_Breaking_the_Information-Lossy_Barrier_in_Self-Supervised_Single_Image_Denoising_CVPR_2025_paper.pdf) | 格子に依存する間引きを使わない対照。推定ノイズに細線が混入した場合や、紙面の白と墨の黒でクリッピングにより残差が非対称になる場合の線保存は、別に検証が要る。 |
| **Pixel2Pixel**（Ma ほか、IEEE TPAMI 2025） | 離れた位置の類似画素を集めた画素バンクを用い、単一画像の自己相似性から多数の擬似学習例を生成する。[原著要旨](https://pubmed.ncbi.nlm.nih.gov/40031719/) | 局所 2×2 に閉じない対照。平坦な塗りや反復模様では類似候補を得やすい一方、孤立した短線や交点の候補の質は写真ベンチマークの平均値から判断できない。 |
| **ZS-NCD**（Zafari ほか、NeurIPS 2025） | 事前学習していないニューラル圧縮モデルを単一のノイズ画像のパッチに適合させ、重複パッチを集約する。圧縮のエントロピー制約を過適合の抑制に使う。[会議論文](https://proceedings.neurips.cc/paper_files/paper/2025/file/df30b1caf39e0e1f5353011398335af4-Paper-Conference.pdf)、[公式コード](https://github.com/Computational-Imaging-RU/ZS-NCDenoiser) | 擬似ペアを用いない別系統のゼロショット法。対角分割の影響を受けず、圧縮が面積の小さい線を残せるかと、パッチ集約を含む CPU 費用が比較点になる。 |

ZS-N2M の平均ターゲットは、入力に使った観測値を含む。クリーン信号間の差の縮小と、入力・ターゲットのノイズの独立性は別の性質である。同様に Median2Median のランダム割り当ては 2 枚の期待輝度を揃えるが、近傍の共有や選別を経た擬似ペアの独立性は実測すべき量として扱う。[ZS-N2M §2–3](https://ojs.aaai.org/index.php/AAAI/article/download/32575/34730)、[Median2Median §3.1](https://arxiv.org/html/2510.01666)

## 2. 線画・イラストに特化した復元

| 手法・出典 | 要点 | 本研究との違い |
| --- | --- | --- |
| **waifu2x / nunif**（nagadomi、公開実装） | アニメ・イラスト用の学習済みモデル。等倍のノイズ除去ができ、公式の説明ではイラスト向けモデルは JPEG ノイズを想定し、Art/Scan 向けのモデルも提供する。[公式実装](https://github.com/nagadomi/waifu2x)、[Art/Scan の公式説明](https://github.com/nagadomi/nunif/blob/master/waifu2x/docs/gui_ja.md) | E5 の JPEG・スキャンで、等倍の実用比較に使える。学習済みのイラスト事前分布を利用する条件で、独立ノイズに対する単一画像内学習とは利用できる情報が異なる。 |
| **漫画の描線とスクリーントーンの分離**（Ito ほか、Eurographics 2015 Short Papers） | LoG からトーン除去用マスク、フローに沿った DoG から描線保存用マスクを作り、連結成分を使って統合する。学習データを要しない。[原著 §2](https://diglib.eg.org/items/79cb54a0-109d-492e-ae83-0827a4eaccae) | 方向と連結性を利用する先例で M2・M3 に近い。目的は線画層とトーン層の分離であり、原画像に意図されたトーンと平坦な塗りを保持したまま劣化を除去する本研究とは出力が異なる。 |
| **Exploiting Aliasing for Manga Restoration / MR-Net**（Xie ほか、CVPR 2021） | 劣化した漫画をグレースケールのトーン表現に変換・整形し、トーンのエイリアシングにもとづいて領域ごとの二値トーンを目的解像度で再合成する。[原著](https://openaccess.thecvf.com/content/CVPR2021/papers/Xie_Exploiting_Aliasing_for_Manga_Restoration_CVPR_2021_paper.pdf)、[著者のプロジェクトページ](https://ttwong12.github.io/papers/mangarestore/mangarestore.html) | 漫画のスキャン・縮小劣化に近く、意図された周期模様を保存対象とする先例。学習済みネットでのトーン復元が中心で、1 px 線に対する方向別のデノイズ診断とは目的が異なる。 |
| **Learning to Simplify**（Simo-Serra ほか、SIGGRAPH 2016） | ラフスケッチと清書済み線画のペアで全畳み込みネットを教師あり学習し、重ね描きなどを統合して線を整理する。[原著](https://esslab.jp/publications/SimoSerraSIGGRAPH2016.pdf) | 紙から取り込んだラフ線画にも関係するが、線の取捨選択や形の変更を許す清書タスク。本研究は元から存在する細線・濃度・接続を評価する。 |
| **壁画線画抽出の CycleGAN デノイズ**（Feng ほか、npj Heritage Science 2025） | 壁画から抽出したノイズ付き線画と参照線画の 2 ドメインを用いて、CycleGAN で線画を復元する。著者は SSIM に加えて線の接続性を評価する。[原著の Denoising of line drawings 節](https://www.nature.com/articles/s40494-025-01908-3) | 線画のノイズ除去と接続性を直接扱う近接研究。画像同士の厳密なペアを要しないが、ドメインの学習画像と参照線画を使う。壁画からの線抽出に伴う欠落補完と、本研究の同解像度での既存線保存とでは許容される編集が異なる。 |

## 3. 自己教師ありデノイズの基礎と比較手法

| 手法・出典 | 要点 | 本研究との違い |
| --- | --- | --- |
| **Noise2Noise**（Lehtinen ほか、ICML 2018） | 同一信号を観測したノイズ付きペアから学習する。二乗損失では、潜在信号と入力を条件としたターゲットノイズの平均がゼロなら、クリーンターゲットによる学習と同じ最適解を得る。[原著 §2](https://proceedings.mlr.press/v80/lehtinen18a.html) | 同一信号の別観測がある設定。ZS-N2N の 2 枚は観測画像内の異なる画素から作るため、潜在信号も一致するかが本研究の問いになる。 |
| **Neighbor2Neighbor**（Huang ほか、CVPR 2021） | 非重複の小ブロックごとに隣接画素をランダムに一組選び、擬似ペアを生成する。入力・教師間の信号差を考慮する正則化を加える。原著の標準評価ではノイズ画像集合で学習する。[原著 §3–4](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf) | M1 の直接の先例。近傍のランダム化自体は既出であり、補完的な 2 画素平均の分割と線方向・格子位相別の評価が本研究で調べる部分。 |
| **Noise2Fast**（Lequyer ほか、Nature Machine Intelligence 2022） | チェッカーボード状に画素を抜き出し、1 枚から 4 枚の縮小画像を作る。小型 CNN をテスト時に適合し、元画像への誤差の推移を用いて停止を判断する。[著者プレプリント](https://arxiv.org/abs/2108.10209) | 平均を伴わない格子型の単一画像法。格子位相による細線の欠損が ZS-N2N に固有かを比較できる。 |
| **Noise2Void**（Krull ほか、CVPR 2019） | 中心画素を入力から隠し、その周囲から中心の観測値を予測する。単独のノイズ画像から学習例を作れる。信号の局所依存とノイズの条件付き独立性を利用する。[原著 §2–3](https://openaccess.thecvf.com/content_CVPR_2019/papers/Krull_Noise2Void_-_Learning_Denoising_From_Single_Noisy_Images_CVPR_2019_paper.pdf) | 2×2 の対角平均を使わない。短線の端点や交点のように、中心画素を隠すと周囲から予測しづらい箇所が線画での検証点。 |
| **Noise2Self**（Batson & Royer、ICML 2019） | 予測する画素集合を入力から除外する J-invariance を定式化。適切なノイズ独立性と不偏性の下、ノイズ画像だけで自己教師ありリスクを測ってモデルを訓練・調整できる。[原著](https://proceedings.mlr.press/v97/batson19a.html) | ブラインドスポットの理論的な比較軸。自己教師あり損失の低さと細線の再現率の関係には、スパースな線を評価する別の指標が必要。 |
| **Self2Self**（Quan ほか、CVPR 2020） | 1 枚をベルヌーイマスクで分割し、ドロップアウトを用いた自己予測で学習する。推論時に複数の出力を平均する。[原著 §3](https://openaccess.thecvf.com/content_CVPR_2020/papers/Quan_Self2Self_With_Dropout_Learning_Self-Supervised_Denoising_From_Single_Image_CVPR_2020_paper.pdf) | ZS-N2N 原著で高ノイズ時に強い比較対象。線保存と計算費用を比較する際は、推論平均の回数と学習の反復数が重要。 |
| **AP-BSN**（Lee ほか、CVPR 2022） | Pixel-shuffle downsampling のストライドを学習時と推論時で変え、実写ノイズの画素間相関の緩和と細部保存の両立を図る。[原著 §3](https://openaccess.thecvf.com/content/CVPR2022/papers/Lee_AP-BSN_Self-Supervised_Denoising_for_Real-World_Images_via_Asymmetric_PD_and_CVPR_2022_paper.pdf) | 原著の主設定はノイズ画像集合を使う。画素間隔を広げるとノイズ相関が減る一方、線を予測する近傍情報も弱くなるという M6 の設計上の比較軸。 |
| **MASH**（Chihaoui & Favaro、CVPR 2024） | マスク量を適応させ、局所画素のシャッフルで空間相関を弱める単一画像のブラインドスポット法。カメラと顕微鏡の実ノイズで評価する。[原著](https://openaccess.thecvf.com/content/CVPR2024/html/Chihaoui_Masked_and_Shuffled_Blind_Spot_Denoising_for_Real-World_Images_CVPR_2024_paper.html) | M6・E5 の比較対象。シャッフルでノイズ相関とともに短い筆線の配置も崩れうるため、線の接続性まで測る意味がある。 |
| **Recorrupted-to-Recorrupted / R2R**（Pang ほか、CVPR 2021） | 既知または推定したノイズ統計を使い、観測に人工ノイズを正負で加えたペアから、クリーン教師付き損失に対応する目的を作る。単一画像の実験も報告する。[原著 §3–4](https://openaccess.thecvf.com/content/CVPR2021/papers/Pang_Recorrupted-to-Recorrupted_Unsupervised_Deep_Learning_for_Image_Denoising_CVPR_2021_paper.pdf) | 空間間引きがないため、既知の合成ノイズでサンプラー起因の劣化を調べる対照。紙目や JPEG を含む未知の複合劣化では、再汚染モデルの妥当性が追加条件になる。 |
| **Deep Image Prior / DIP**（Ulyanov ほか、CVPR 2018） | 未学習の生成 CNN を 1 枚の観測に適合させ、ネットの構造と早期停止を画像の事前分布として用いる。[原著](https://openaccess.thecvf.com/content_cvpr_2018/papers/Ulyanov_Deep_Image_Prior_CVPR_2018_paper.pdf) | 擬似ペアを用いない E1 の対照。ZS-N2N 原著の DIP は正解を見て停止点を選ぶので、正解なしの停止規則を用いた結果とは条件を区別する。[ZS-N2N §4.1](https://arxiv.org/html/2303.11253) |
| **BM3D**（Dabov ほか、IEEE TIP 2007） | 類似パッチを 3 次元に積み、変換領域で協調フィルタリングする学習不要の古典法。[原著](https://doi.org/10.1109/TIP.2007.901238) | E1 の非学習対照。合成実験で真のノイズ標準偏差を与える条件と実画像から推定する条件では利用できる情報が異なる。 |

## Research Gaps

### 1. 間引き時の信号差と、学習後の細線の消失を結び付ける（G1、E1–E2）

2×2 ブロックのクリーンな暗さを行順に `a,b,c,d` とすると、ZS-N2N の対角平均の差は `(b+c−a−d)/2`。理想的な水平・垂直の 1 px 線がブロックを横切れば両側に 1 画素ずつ入り、差は 0。2 画素が対角に入る 45° 線では片側に集中する。ブロックの原点を 1 px ずらすと、同じ 45° の線でも各ブロックに入る線画素数や差の大きさが変わる。端点・曲線・アンチエイリアスにも同じ式を適用できるが、学習後の出力を式だけから決めることはできない。[ZS-N2N §3.2](https://arxiv.org/html/2303.11253)、[研究計画 H1–H2](plan.html#hyp)

- クリーン画像で測ったペア間の絶対差または二乗差と、復元後の骨格再現率・線濃度を、角度・線幅・ブロック原点の 4 位相ごとに対応づける。同じ描画の位相違いは同一画像の繰り返しと扱う。
- ペア予測と間引き整合性の各項の除去を、モデル規模・学習回数を揃えて比較する。原著のアブレーションではペア予測項を外すと恒等写像を学習するため、線が残る場合も背景ノイズの残留と併せて解釈する。[ZS-N2N 付録 A](https://arxiv.org/html/2303.11253)弱ノイズまたは無劣化入力に加え、黒・白付近で平均ゼロが崩れうるクリップ付きガウスノイズと、切り詰め前の条件を区別する。入力のクリップによる効果とサンプラーの効果をそれぞれ調べられる。[ZS-N2N 付録 A](https://arxiv.org/html/2303.11253)

### 2. 幾何を使う効果を、ランダム化・モデル規模・計算量の効果から切り分ける（G2、E3–E4・E6）

近傍のランダム化は Neighbor2Neighbor、擬似ターゲットの差の縮小は ZS-N2M、勾配に応じたサンプリングは ZS-HPD に先例がある。差分として検証できるのは、**細線の接線と格子位相を考慮し、線の信号が両方に写るペアを軽量に選ぶことで、線保存を改善できるか**である。[Neighbor2Neighbor §3](https://openaccess.thecvf.com/content/CVPR2021/papers/Huang_Neighbor2Neighbor_Self-Supervised_Denoising_From_Single_Noisy_Images_CVPR_2021_paper.pdf)、[ZS-N2M §3](https://ojs.aaai.org/index.php/AAAI/article/download/32575/34730)、[ZS-HPD §3](https://openaccess.thecvf.com/content/CVPR2026/papers/Zhao_Zero-Shot_Image_Denoising_via_Hybrid_Prior-Guided_Pseudo_Sample_Generation_CVPR_2026_paper.pdf)

- M1 は対角・左右分割・上下分割を同じ CNN で比較する。M2 では接線上にある画素をどちらの部分画像へ割り当てるかを明記し、両方に線の信号が入ることをクリーンなテストパターンで確かめる。ノイズ画像から選択を決める場合、元のノイズが独立でも選択後の条件付き統計は変わりうる。
- M3 は重みマップの有無で比較し、ノイズのエッジの強調を適合率で確認する。偶数寸法で格子が揃う M4 の 90° 回転・反転は 2 種の対角カーネルを交換または保存する。単純な変換平均で各ブロックの信号欠損が解消する保証はなく、モデルの向き依存を測る対照になる。
- ZS-N2M を擬似ペア差を扱う外部比較に置き、ZS-HPD を勾配に応じた生成手法として参照する。計画の改善幅（線画 PSNR +0.5 dB、線再現率 +5 ポイント、写真での低下 0.1 dB 以内）を同一入力で判定する。計算時間は前処理・学習・推論を含めた M4 CPU の実測で、ZS-N2N の 2 倍以内という基準に照らす。[研究計画 G2・評価基準](plan.html#criteria)

### 3. 欠損・過剰な太り・誤接続を同時に捉える（E1–E5）

今回確認した単一画像法の評価には、写真や顕微鏡画像の PSNR・SSIM や定性的な細部比較が多い。線画に対して、幅・角度・格子位相を制御し、細線の保存と背景のノイズ残留を同時に測るプロトコルが本研究の比較上の空白となる。文書の二値化には文字の骨格と線幅を考慮する評価があり、clDice は予測骨格と正解マスク、および正解骨格と予測マスクの相互被覆を調和平均する。[文書二値化の評価原著](https://doi.org/10.1109/TIP.2012.2219550)、[clDice 原著](https://openaccess.thecvf.com/content/CVPR2021/papers/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_Segmentation_CVPR_2021_paper.pdf)

- 骨格再現率は予測線の膨張でも上がりうるため、画素適合率、clDice、線濃度、線の近傍の PSNR を併記する。clDice は骨格の被覆を捉えるため、線幅や濃度には別の指標を使う。必要に応じて、代表例で切断と誤接続を拡大表示する。
- 二値化の規則と位置ずれの許容幅を先に固定し、アンチエイリアスや薄い筆線では閾値への感度も確認する。同一ノイズ・同一画像で手法を対応づけ、シードを平均した画像単位の差に対して計画の検定を適用する。[研究計画の評価基準](plan.html#criteria)

### 4. 紙・スキャナ・圧縮の劣化と、意図された線やトーンを識別する（G3、E5）

実写ノイズ向けの CFC・MASH と、方向性のある構造ノイズを対象にした Median2Median は有力な比較候補。漫画復元はスクリーントーンを意図された内容として扱う。紙目、走査方向の筋、JPEG のブロックやリンギングでは、相関の尺度や向きが異なり、筆線やハッチングと重なる条件もある。[Zero-Shot CFC](https://bmvc2025.bmva.org/proceedings/577/)、[MASH](https://openaccess.thecvf.com/content/CVPR2024/html/Chihaoui_Masked_and_Shuffled_Blind_Spot_Denoising_for_Real-World_Images_CVPR_2024_paper.html)、[Median2Median](https://arxiv.org/html/2510.01666)、[MR-Net](https://ttwong12.github.io/papers/mangarestore/mangarestore.html)

- 参照付き合成データでは、紙目・走査むら・JPEG を個別に加え、クリーン信号のペア間差と、劣化を通したペア間の残差相関を別々に測る。間引き間隔を広げるときは両者の変化を比較する。意図したトーンやハッチングを残す条件も定義する。
- 自作の実スキャンでは、位置合わせ可能な原画がある場合に線指標を計算し、参照が得られない場合は背景ノイズ、線の消失、偽線やハローを同じ拡大範囲で比較する。等倍の waifu2x Art/Scan を実用比較に置ける。

この調査で確認した範囲では、単一画像だけを使う軽量な復元について、**線幅・角度・格子位相ごとの擬似ペアの信号差と、学習後の線の保存を対応づけ、写真の画質と CPU 計算時間まで同じ比較の中で測る**余地がある。新規性の判断には、特に ZS-N2M と ZS-HPD に対してこの比較で何を追加できたかが重要になる。
