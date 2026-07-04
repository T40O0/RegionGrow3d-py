# RegionGrow3D Python 版 — ユーザマニュアル

> 🌐 **English version**: [MANUAL.md](MANUAL.md)

最終更新: 2026-05-12

---

## 目次
1. [基本概念](#1-基本概念)
2. [インストール](#2-インストール)
3. [Web UI の使い方](#3-web-ui-の使い方)
4. [CLI の使い方](#4-cli-の使い方)
5. [モード詳細](#5-モード詳細)
6. [入力ファイルの種類と配置](#6-入力ファイルの種類と配置)
7. [出力の見方](#7-出力の見方)
8. [MATLAB との比較・検証](#8-matlab-との比較検証)
9. [トラブルシューティング](#9-トラブルシューティング)
10. [パフォーマンスのヒント](#10-パフォーマンスのヒント)
11. [API リファレンス](#11-api-リファレンス)

---

## 1. 基本概念

RegionGrow3D は **連続する不安定セルから landslide cluster を成長させ、力のつり合い (3D Janbu 法) が
取れた時点で停止** する決定論的解析手法です。確率分布に従ってせん断抵抗角 φ' と粘着力 c' を変えな
がら 10 回程度走らせ、加重平均で **地すべり危険度 (susceptibility) 0–100 %** を出します。

主な処理段階:
```
DEM (.tif)
  ↓ pad + 勾配 (gradient_prince)
  ↓ 土層厚 (soil_depth: Roering 5000 yr or 一様)
  ↓ 静水圧 / 乾燥 / (水文力学) で W, σ_s 算出
  ↓ 地震 PGA (off / uniform / raster) を加算
  ↓ 尾根+谷無成長帯マスク (acc 閾値 / slope units)
  ↓ せん断強度パラメータペア (φ', c') ごとに RegionGrow:
        Q = N·sin(α) - (c·A + (N-U)·tan(φ))·cos(α) + PGA·W
        unstable = (Q > 0) → クラスタ化 → erosion → 力が釣り合うまで growth
  ↓ Σ slides_final * prob → susceptibility map
  ↓ sus_*.tif 出力
```

### 1.1 詳細フロー — RegionGrow とは何をしているのか

#### (a) 「不安定セル」の定義
各セル単体に対し、無限長斜面の極限平衡式 (1D Janbu) で

```
Q_i = N_i·sin(α_i)              ← 重量と地震力による下方分力
       − (c·A_i + (N_i − U_i)·tan(φ)) · cos(α_i)  ← せん断抵抗
       + PGA·W_i                ← 擬似静的地震荷重
```

を計算し、**`Q_i > 0` となるセルを "セル単独で不安定" とマークする**。ここで:
- `N_i = W_i · cos(α_i)` (法線力)、 `A_i = cellsize² / cos(α_i)` (斜面面積)
- `U_i = γ_w · hw_i · A_i` (間隙水圧、mw に応じた hw = mw·depth)
- `W_i = γ·depth·cellsize²` (セル重量)

これが **`slides_initial_io` 配列** で、RegionGrow の **成長起点** になります。

#### (b) クラスタ化と erosion — 詳細手順

`Q > 0` のセル群 (= `slides_initial_io` のバイナリマスク) を **個別の候補すべり面** に
切り分ける前処理。次の 4 段階を順に適用します。

**① no-grow セルのマスクアウト** ([region_grow.py:80-85](python/region3d/region_grow.py:80))

```python
FB_assign[ngi, ngj] = False     # no_grow セルを seed から除外
```

クラスタが尾根・谷を跨いで連結しないようにする。実質的にここで地形的な分割が決まる。

**② 微小クラスタの一括除去 (`bwareaopen`)** ([region_grow.py:88](python/region3d/region_grow.py:88))

```python
FB_assign = bwareaopen(FB_assign, cluster_size_thresh)   # default 7
```

8 連結成分が `cluster_size_thresh` (デフォルト 7) セル未満のものを **マスク全体から削除**。
10 m DEM なら 7 セル = 700 m² 程度で、それ未満のクラスタは
- 数値誤差や境界の凹凸に由来するノイズ
- 物理的に landslide とみなせない微小な不安定セル

として除外。

**③ 8 連結ラベリング (`bwconncomp`)** ([region_grow.py:91](python/region3d/region_grow.py:91))

```python
pixel_idx_list, num_objects, _ = bwconncomp_F(FB_assign)
```

残ったマスクを 8 連結で連結成分にラベリングし、各成分の **ピクセルインデックスリスト** を
返す。これが「初期クラスタ群」= `slides_initial_io` の中身。各クラスタは以降独立に処理される。

**④ クラスタごとの形状クリーンアップ (erosion)** ([region_grow.py:184-201](python/region3d/region_grow.py:184))

各クラスタを **「磨いてから渡す」** 工程。流れは下のフロー 1 本だけ:

```
クラスタ C (= ③ から渡されたピクセル集合)
   │
   ├─ 健全？  (size ≥ 7  &  spur なし  &  連続)
   │     │ NO  → このクラスタを破棄して次へ           ┐
   │     │ YES → 次へ                                 │ ※ ここだけが
   │                                                  │   "クラスタ消滅"
   ├─ C を保存しておく (C_save = C)                    │   イベント
   │
   ├─ erosion を 2 回適用 (境界を 1 セルずつ計 2 セル内側に削る)
   │
   ├─ 削った後も依然として健全？  (size ≥ 7 & spur なし & 連続)
   │     │ YES → C := 削った形           (磨き成功、採用)
   │     │ NO  → C := C_save              (磨き失敗、元に戻す)
   │
   └─→ この C を slides_eroded_io に登録し、Janbu/grow loop へ
```

ポイントは **増減も分離も「試行と取り消し」だけ** という点:

| 段階 | 操作 | 「クラスタが減る/分かれる」と聞こえる挙動 | 実際 |
|---|---|---|---|
| 健全性チェック (前) | spur / 連続性が NG なら破棄 | クラスタが消える | **唯一の消滅イベント** |
| erosion 適用 | 境界を 2 セル削る | 痩せる、場合により分割 | **試行のみ**。次の連続性チェックで NG なら元に戻すため、分割は確定しない |
| 健全性チェック (後) | 削った結果が NG なら **C_save に戻す** | "分割されたら戻す" | **常に磨く前か磨いた後どちらか一方が採用される** |

つまり:
- **erosion で実際にクラスタが 2 つに分かれることはない** — 分かれた場合は失敗とみなして元の 1 個に戻す
- erosion で**クラスタが完全に消えることもない** — 消えるほど痩せた場合も元に戻す
- 各クラスタは必ず「磨いた形 or 元の形」のどちらか 1 通りで次段へ

erosion → 妥当性チェック → revert という設計の意図は？
**「磨ける時だけ磨く」**ベストエフォート方針です。erosion の主目的は
**粗い境界の平滑化** で、事前にはどのクラスタで効くか判別できないので
一度試して、結果が劣化していたら採用しないという仕組みです。

> ⚠️ **erosion はクラスタを分割する手段ではありません。**
> 1 セル幅の細い首で繋がっただけの 2 塊が erosion 後に分かれた場合、
> `continuity_check` で連続性 NG と判定されて **元の 1 個に戻されます**
> ([growth.py:169](../python/region3d/growth.py)、[region_grow.py:193-201](../python/region3d/region_grow.py))。
> 分割したい場合は erosion ではなく nogrow マスクや spur テストで処理します。

ここまでで得られた形状群が **`slides_eroded_io`** = 「形が clean で Janbu 計算を回せる
候補すべり面のセット」となり、次段 (c) の Cluster grow loop で 1 クラスタずつ力が
釣り合うまで反復します。

#### (c) Cluster grow loop — 1 クラスタずつ独立に
各クラスタ `C` について以下を **力のつり合いが取れるまで** 反復:

1. **3D Janbu 計算**: クラスタ全体を 1 つの剛体として 3D Janbu 法 (Hungr 1989) で
   滑動方向 (α 最大の主方向) を決定し、

   ```
   F = Σ_C (c·A_i + (N_i − U_i)·tan(φ)) · cos(α_i) / cos(α_C)
   D = Σ_C N_i · sin(α_C)  + PGA · Σ_C W_i
   err = D − F
   ```

   `err <= 0` なら釣り合っているのでクラスタ確定 (`slides_final_io` に追加)。

2. **隣接セルを wedge として追加**: クラスタ周囲を回転角 ±20° (`rot_range`) で
   8 方向 (`rot_num`) スキャンし、alpha-shape で「次に取り込むべきセル群 (wedge)」を
   決定。複数候補から `err` を最小化するセル集合を選ぶ。

3. **境界制約のチェック**: wedge が `no_grow` セルに当たった場合、その方向には伸ばさない。

4. クラスタが上記反復で `max_growth_cycles` (デフォルト 120) を超える、err が再び増え始めた、
   などの停止条件で打ち切り。

#### (d) no-grow マスクの役割
`no_grow_io` は **クラスタ成長が物理的・地形的に超えてはならない線** を指定する
バイナリマスク。生成方法 3 通り (詳細 §5.4):
- **acc-threshold**: 流向→流量→「acc > X (谷)」「inverted acc > Y (尾根)」を細線化
- **slope units** (Alvioli 2016/2025): ハーフベイスン + アスペクト円周分散で分割
- **既存 .mat 読込**: MATLAB 由来または前回の Python 計算結果を再利用

**no-grow を有効化 (mode=1) する効果**:
- クラスタが谷を横切って伸びるのを防ぐ → 物理的に正しい単独すべり面が得られる
- 尾根を越えた成長を防ぐ → 1 クラスタが複数斜面を跨ぐ過剰結合を抑制

**no-grow を無効化 (mode=0) すると**: クラスタが地形を無視して任意の方向に伸び、
時に 1 つの巨大クラスタが大流域全体を覆ってしまうことがある (= over-merging)。
高密度の DEM ほど影響が大きい。

#### (e) 確率分布での重ね合わせ
せん断強度パラメータ (φ, c) ペアを `shear_strength.mat` の `prob_phi`, `prob_coh`, `prob` から N 組 (典型 10)
取り、それぞれで上記 RegionGrow を走らせる。最終 susceptibility は

```
sus_map[i,j] = Σ_k  prob[k] · slides_final_io[k][i,j]
```

で **0 ≤ sus ≤ Σprob ≈ 1.0 (=100%)**。すなわち「弱い土ほど起きやすい / 強い土でも起きる」
という強度感度を 1 枚のマップに集約したもの。

---

## 2. インストール

### 2.1 Docker (推奨)

Python 環境を作る必要がなく、ネイティブ依存も全てコンテナ内で解決されるので
最も再現性の高い方法です。

```bash
# build (一度だけ、約 5 分)
docker build -t region3d:latest .

# Web UI 起動 (lib と output をホストにマウント)
docker run --rm -p 8501:8501 \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  -v "$(pwd)/python/output_webui:/app/python/output_webui" \
  region3d:latest

# CLI 直接実行
docker run --rm \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  region3d:latest \
  python python/driver.py --soil_strength_mode 2 --phi_uniform 30 --coh_uniform 5
```

`docker-compose.yml` も同梱:
```bash
docker compose up           # Web UI
docker compose run --rm region3d python python/driver.py   # CLI
```

ベースイメージ: `python:3.13-slim` (約 200 MB)。完成イメージは依存込みで約 1 GB。
ボリュームマウントしないと DEM がコンテナ内になく、結果も永続化されません。

### 2.2 conda (ローカルインストール)
```bash
conda env create -f environment.yml
conda activate region3d
```
`environment.yml` で全パッケージを `conda-forge` に固定しています。
ネイティブ依存 (`rasterio` の GDAL、`numba` の LLVM、`scipy/numpy` の BLAS)
を互換バージョンで揃えます。

#### pip のみで構築する場合 (フォールバック)
conda が使えない環境では:
```bash
python -m venv .venv
.venv/Scripts/activate     # Windows  (POSIX 系: source .venv/bin/activate)
pip install -r requirements.txt
```

#### 必須ライブラリ
| パッケージ | 用途 |
|---|---|
| `numpy` (>=2.0) | 数値計算 |
| `scipy` | morphology, distance transform, .mat I/O |
| `rasterio` | GeoTIFF 読み書き |
| `scikit-image` | imreconstruct (fillsinks)、skeletonize 検証 |
| `numba` | JIT 高速化 (soil_depth, flow_accumulation, alpha-shape) |
| `matplotlib` | 結果プロット |
| `streamlit` | Web UI |
| `pandas` | UI 統計テーブル |

### 2.3 動作確認
```bash
python python/tests/smoke_test_modules.py
python python/tests/test_alpha_shape.py
python python/tests/test_bwboundaries.py
python python/tests/test_skel.py
```

---

## 3. Web UI の使い方

```bash
streamlit run python/gui.py
```

### 3.1 サイドバーの構成

```
📁 入力ファイル配置ガイド (折りたたみ)
📍 DEM       既存 lib/DEM/*.tif から選択 + アップロード
⚙ モード設定
  💧 土壌水分    [0=乾燥 | 1=静水圧]
                 - mw スライダー (mode=1)
  🟫 土層厚      [1=Roering | 2=一様]
                 - 計算ソース (.mat読込 / Python計算)
                 - .mat ドロップダウン or Roering 期間 / 一様値
  🪨 せん断強度パラメータ  [1=分布 | 2=一様]
                 - 単一 run-index 選択 (mode=1)
                 - ⚡ 全ラン並列実行 (2本同時) チェック (mode=1 & All runs)
                 - φ, c (mode=2)
  🚧 成長境界    [0=なし | 1=尾根+谷]
                 - 計算ソース (.mat読込 / Python計算)
                 - .mat ドロップダウン or ridge/valley 閾値
  🌐 地震        [off | uniform | raster]
                 - PGA / scaling / TIFF パス
  🌿 根系強度    S_roots (kPa)
📐 基本物性 (折りたたみ): Gs, γ_w, γ_dry, γ_sat
📤 出力先     親ディレクトリ + 実行ID + 既存上書き確認
▶ 解析開始
```

### 3.2 実行フロー

1. パラメータ設定 → [解析開始] クリック
2. **計算中**:
   - サイドバー全ロック (誤操作防止)
   - メイン画面に警告バナー + 進捗バー + ETA + ログ
   - [⏹ 停止] ボタンで中断可能
3. **完了後**:
   - 完了メッセージ
   - 結果タブ自動表示
   - **並列モード時**は aggregate 後に **引張/圧縮マップ `net_force_prob_<susname>.tif`**
     も自動生成される (§3.6)

### 3.3 結果タブ

| タブ | 内容 |
|---|---|
| 🗺 マップ | レイヤー切替 (susceptibility / 無成長帯 / 土層厚 / PGA)、陰影図オーバーレイ、TIFF ダウンロード |
| 📊 統計 | 有効セル数、>0% / >50% / >90% の比率、想定面積、ラン別クラスタ数 ((φ, c, prob) 表 + 棒グラフ) |
| 📈 ヒストグラム | 正のセル (>0%) の度数分布 (log scale) + CDF + 中央値 / 90 パーセンタイル |

### 3.4 ブラウザを閉じた場合
Streamlit セッションは切れるが **subprocess は継続**し、`out_dir/<susname>/` に成果物が
保存される。再接続するとセッションが新規になり、ログ表示は消えるが結果タブで読み出し可能。

### 3.5 デタッチ起動 (Windows、夜通し運用)

ターミナルから普通に起動するとそのシェルにバインドされ、シェル終了とともに
Streamlit プロセスも消えます。長時間稼働させたい場合は `Start-Process` で
**親シェルから切り離して** 起動してください (子プロセスは explorer.exe / system に
re-parent され、シェル終了の影響を受けなくなります):

```powershell
$py     = "$env:LOCALAPPDATA\miniconda3\envs\gis_conda\python.exe"   # 使用する env の python
$repo   = "C:\Users\040869\Documents\GitHub\RegionGrow3d-py"          # リポジトリのパス
$logDir = Join-Path $repo "python\output_webui"
$logFile = Join-Path $logDir ".streamlit_server.log"
$errFile = Join-Path $logDir ".streamlit_server.err.log"
$pidFile = Join-Path $logDir ".streamlit_server.pid"

if (-not (Test-Path $logDir)) { New-Item -ItemType Directory -Path $logDir -Force | Out-Null }
"" | Set-Content -Path $logFile -Encoding utf8
"" | Set-Content -Path $errFile -Encoding utf8

$proc = Start-Process -FilePath $py `
    -ArgumentList "-m","streamlit","run",(Join-Path $repo "python\gui.py"),`
                  "--server.headless=true","--server.address=0.0.0.0" `
    -WindowStyle Hidden `
    -RedirectStandardOutput $logFile `
    -RedirectStandardError $errFile `
    -PassThru
$proc.Id | Set-Content -Path $pidFile -Encoding ASCII
"detached PID=$($proc.Id); log=$logFile"
```

**生死確認 / 停止:**

```powershell
# 8501 にリスナーがいるか
Get-NetTCPConnection -LocalPort 8501 -State Listen -ErrorAction SilentlyContinue

# ログを tail
Get-Content "$repo\python\output_webui\.streamlit_server.log" -Wait -Tail 20

# 停止
$pidVal = (Get-Content "$repo\python\output_webui\.streamlit_server.pid").Trim()
Stop-Process -Id $pidVal -Force -ErrorAction SilentlyContinue
Remove-Item "$repo\python\output_webui\.streamlit_server.pid" -ErrorAction SilentlyContinue
```

> ⚠️ **スリープ**: デタッチ起動でも Windows がスリープに入るとプロセスは
> 停止します。本格的に夜通し動かしたい場合は、Windows の電源設定 → スリープを
> 「なし」にするか、`gui.py` 起動時に `SetThreadExecutionState` の keep-awake
> 呼び出しを追加するパッチが必要です (既存の `runner._set_keep_awake` フックは
> 解析実行中のみ有効で、アイドル時には呼ばれません)。

### 3.6 並列実行 (⚡ 全ラン並列) と引張/圧縮マップ

せん断強度=分布(mode 1) で **All runs** のとき、サイドバーの
**「⚡ 全ラン並列実行 (2本同時)」** にチェックすると、10 個の φ ランを直列でなく
**2プロセス同時**に走らせ (`python/_sus_parallel.py` オーケストレータ経由)、最後に
`--aggregate` で合成します。各 φ の寄与は可換な確率加重和なので、**結果は直列と
ビット一致**します。中断しても各 φ の contrib は保存され、再実行で残りだけ再開します。

- **メモリ**: 1ランあたり約 20 GB。2本並列には合計約 40 GB 必要です。**Docker/WSL2 で
  実行する場合は VM メモリを増やしてください** — `%USERPROFILE%\.wslconfig` に:
  ```ini
  [wsl2]
  memory=54GB
  ```
  を設定し `wsl --shutdown` で反映 (既定は host の約 50%)。ネイティブ実行時も
  空き RAM を確認のうえで。
- **precompute 中は進捗バーが 0% のまま**数分続きます (DEM・導関数・力の事前計算)。
  ログにフェーズが出ていれば正常。**Start を連打しない**でください (二重起動ガードが
  弾きますが、待つのが正解)。

**引張/圧縮マップ (自動生成)**: 並列モードの aggregate 完了後、
`net_force_prob_<susname>.tif` が自動生成されます。各すべりセルの
**正味の力 q = 駆動 − 抵抗** (`Interslice_Force`) は、物理的には
**q > 0 = 引張**（駆動優勢＝active、頭部側）、**q < 0 = 圧縮**（抵抗優勢＝passive、
末端側）です。ただし可視化の都合で **q の符号を反転した `−q` を格納**しています
（`Σ prob·(−q)`）。この反転により **出力 TIF の値は 圧縮が正 (赤系統) / 引張が負
(青系統)** になります。0 を中心とした発散カラーマップで表示してください。CLI からは
`--tension_compression 1` で単独生成可 (contrib が必要)。

---

## 4. CLI の使い方

driver.py は subprocess としても直接 CLI としても使えます。

### 4.1 最小実行
```bash
python python/driver.py --DEM_path lib/DEM/<your_DEM>.tif --test_no 1
```
`--DEM_path` と `--test_no` は常に必須です。選択したモードに応じて
追加の `--*_mat` パスも条件付きで必須になります (4.2 参照、実行時に
エラーメッセージで案内)。

### 4.2 全 CLI オプション
```bash
python python/driver.py --help
```

主要オプション:
| オプション | 例 | 説明 |
|---|---|---|
| `--DEM_path PATH` | `lib/DEM/<your_DEM>.tif` | **必須**。入力 DEM (絶対 or repo 相対) |
| `--test_no INT` | `1` | **必須**。出力サブフォルダ命名用の数値 ID |
| `--susname_override STR` | `my_scenario` | 出力サブフォルダ名 (空なら test_no 5 桁) |
| `--out_dir PATH` | `python/output` | 出力親ディレクトリ |
| `--soil_moisture_mode 0\|1` | `1` | 0=乾燥 / 1=静水圧 |
| `--mw FLOAT` | `0.5` | 飽和率 (mode=1) |
| `--soil_depth_mode 1\|2` | `1` | 1=Roering / 2=一様 |
| `--soil_depth_uniform FLOAT` | `2.0` | 一様時の土層厚 [m] |
| `--soil_depth_endtime FLOAT` | `5000.0` | Roering シミュ期間 [yr] |
| `--soil_depth_source mat\|compute` | `mat` | .mat 読込 / Python 計算 |
| `--soil_depth_mat PATH` | `lib/soil_depth/<DEM_stem>_soil_depth.mat` | mat ソース時の読込先 |
| `--nogrow_mode 0\|1` | `1` | 0=なし / 1=尾根+谷 |
| `--nogrow_source mat\|compute\|grass` | `mat` | 同上 (`grass`=本家 r.slopeunits、Docker) |
| `--no_grow_mat PATH` | `lib/no_grow/<DEM_stem>_no_grow.mat` | 同上 |
| `--ridge_acc_thresh FLOAT` | `5` | 尾根の流量蓄積閾値 (`compute`) |
| `--valley_acc_thresh FLOAT` | `100` | 谷の流量蓄積閾値 (`compute`) |
| `--slope_unit_thresh/areamin/cvmin/rf/maxiter` | 500000/100000/0.3/2/50 | `grass` の r.slopeunits パラメータ — §5.4 参照 |
| `--soil_strength_mode 1\|2` | `1` | 1=分布 / 2=一様 |
| `--phi_uniform FLOAT` | `25` | 一様 φ' [deg] |
| `--coh_uniform FLOAT` | `2` | 一様 c' [kPa] |
| `--seismic_mode off\|uniform\|raster` | `off` | 地震入力 |
| `--uniform_PGA FLOAT` | `0.3` | uniform 時の PGA [g] |
| `--PGA_path PATH` | `lib/seismic/PGA.tif` | raster 時の TIFF パス |
| `--pseudo_scaling FLOAT` | `1.0` | PGA スケーリング |
| `--S_roots FLOAT` | `10` | 根系強度 [kPa] |
| `--save_intermediates 0\|1` | `1` | 1 で depth/nogrow/PGA/hillshade を TIFF 保存 |
| `--run-index INT` | `None` | 0始まりの単一ランのみ計算し `<out>/<susname>/contribs/` に保存（低メモリ・レジューム、§4.4）。その run の部分 `sus_*.tif` も出力。`0 ≤ index < ラン数` 必須、`--aggregate` と排他 |
| `--aggregate 0\|1` | `0` | `contribs/contrib_run*.npz` を合算して最終マップを書き終了。`--DEM_path` と `--test_no`/`--susname_override` のみで可。寄与が不完全/分布不一致ならエラー（§4.4） |
| `--tension_compression 0\|1` | `0` | `contribs/` から **引張/圧縮マップ `net_force_prob_*.tif`** を生成し終了。per-cell 正味力 q（q>0=引張/q<0=圧縮）の**符号を反転(`−q`)**して φ 確率加重するので**出力は圧縮=正/引張=負**（§3.6）。通常ランと同じ入力が必要、region-grow は走らせない。並列モードでは aggregate 後に自動実行 |
| `--max_cell_offset INT` | `400` | 境界拡張時の局所窓の上限 [セル]。到達したクラスタは `terminate_reason=7` となり **MATLAB と乖離**（MATLABは無制限に再試行）、警告を出力。巨大クラスタを完全成長させたい場合は増やす |

### 4.3 実用例

**地震 + 乾燥 + 一様土層厚**
```bash
python python/driver.py \
  --soil_moisture_mode 0 \
  --soil_depth_mode 2 --soil_depth_uniform 1.5 \
  --soil_strength_mode 2 --phi_uniform 30 --coh_uniform 5 \
  --nogrow_mode 0 \
  --seismic_mode uniform --uniform_PGA 0.3 \
  --susname_override eq_dry_test
```

**Python 単独 (MATLAB .mat なし)**
```bash
python python/driver.py \
  --soil_depth_source compute \
  --nogrow_source compute \
  --soil_strength_mode 2 --phi_uniform 25 --coh_uniform 2 \
  --susname_override python_only
```
→ Python 計算後、`lib/soil_depth/<DEM>_soil_depth_python.mat` と
`lib/no_grow/<DEM>_no_grow_python.mat` が自動保存され、次回以降は `mat` ソースで秒オーダー読込可。

⚠️ **`shear_strength.mat` は対象地盤ごとに利用者が自作する必要があります。**
強度試験データから `(prob, prob_phi, prob_coh)` を構築して
`lib/soil_strength/shear_strength.mat` に保存。データを持たない場合は上記のように
`soil_strength_mode 2` (一様 φ, c) で 1 ラン実行してください。

### 4.4 低メモリ・レジューム (`--run-index` + `--aggregate`)

確率分布ラン (`soil_strength_mode=1`) を大きな DEM で回すと、全ランを 1 プロセスで
計算すると成長状態を同時にメモリ保持します。`--run-index`/`--aggregate` は各インデックス
を別プロセスで計算し（1プロセス1成長＝低ピークメモリ）、後で合算します:

```bash
# 1. 各ランを個別プロセスで計算。各々が
#    <out>/<susname>/contribs/contrib_run<NN>.npz を（全成果物の後に）保存するため、
#    ファイルの存在＝そのインデックス完了。クラッシュ後の再実行で自動スキップされ安全。
for N in 0 1 2 3 4 5 6 7 8 9; do
  python python/driver.py $COMMON --run-index $N
done

# 2. 保存済み寄与を合算して最終 sus_<susname>_python.tif を書き出す。
python python/driver.py --DEM_path <dem> --susname_override <susname> \
  --out_dir <out> --aggregate 1
```

`--aggregate` は誤ったマップを書かないよう、インデックス欠落・重複、寄与間の分布サイズ
不一致（別 `shear_strength.mat` の残骸）でエラーにし、確率合計が 1 でなければ警告します。
各寄与はラン数と「スライドなし」フラグを記録し、一括ループの早期停止挙動を再現します。
共有中間ラスタ（depth/nogrow/PGA/hillshade）は `--run-index 0` が一度だけ書き出します。

> 複数 φ ランのレジューム対応オーケストレータは `python/_sus_parallel.py` にあります
> （2本並列、Windows/Docker 共通。GUI の「⚡ 全ラン並列実行」もこれを使用。§3.6）。

---

## 5. モード詳細

### 5.1 `soil_moisture_mode` — 土壌水分

| 値 | 動作 | 必要パラメータ |
|---|---|---|
| **0 = 乾燥** | hw=0, σ_s=0, W = γ_dry × depth × cellsize² | (なし) |
| **1 = 静水圧** | hw = mw × depth, σ_s = γ_w × hw | `mw` |
| ~~2 = 水文力学~~ | van Genuchten + 浸透式 (未実装) | SMAP, 砂粘土%, 降雨, Rosetta.csv |

mode=1 の `mw=0.5` は「土層厚の半分まで水で飽和」を意味します (詳細は要点ノート参照)。

### 5.2 `soil_depth_mode`

| 値 | 動作 |
|---|---|
| **1 = Roering** | Roering (2008) 非線形地形進化を `soil_depth_endtime` 年シミュレート (Numba JIT) |
| **2 = 一様** | depth = `soil_depth_uniform` を全域に適用 |

### 5.3 `soil_strength_mode`

| 値 | 動作 |
|---|---|
| **1 = 分布** | `shear_strength.mat` から (prob, φ, c) ペア×10 を読込、確率重み付けで susceptibility |
| **2 = 一様** | 単一 (φ_uniform, coh_uniform) で 1 ランのみ |

### 5.4 `nogrow_mode`

`nogrow_mode` は 2 値 (0=なし / 1=境界あり) で、有効時のソースを `nogrow_source` で
3 つから選びます。

| `nogrow_mode` | 動作 |
|---|---|
| **0 = なし** | 成長境界制約なし (どこでも成長可) |
| **1 = 境界あり** | 下記 3 ソースのいずれかから no-grow マスクを取得 |

`nogrow_source` (mode=1 時の生成方式):

| 値 | 名称 | 内容 |
|---|---|---|
| **`mat`** | 既存 .mat 読込 | MATLAB 由来または前回計算結果を再利用 (秒) |
| **`compute`** | acc-threshold ridges + valleys (TopoToolbox-style) | D8 流向→流量、`acc > valley_acc_thresh` (谷)、反転 DEM で `acc > ridge_acc_thresh` (尾根) を細線化 |
| **`grass`** | Slope units (本家 GRASS r.slopeunits, Alvioli 2016/2020) | 本家 `r.slopeunits.create`(MFD 既定)→ `r.slopeunits.clean`(最小面積で小ユニット除去)を実行 → 完全区分のスロープユニット → 境界を no-grow に変換。**GRASS 入り Docker イメージが必要** |

> 旧 `slopeunits`(本家アルゴリズムの純 Python 近似)は廃止しました。本家との一致度が低く
> (実 DEM で ARI≈0.35)、過分割・穴・飛び地などの問題があったためです。忠実な結果は
> 本家を直接動かす `grass` を使ってください。

`grass` モードの主なパラメータ (UI/CLI):

| パラメータ | 意味 (r.slopeunits) | 典型値 |
|---|---|---|
| `slope_unit_thresh` | `thresh`: 初期チャンネル閾値 [m²]。大きいほど流路網が疎、ユニットが大 | 100,000–1,000,000 (デフォルト **500,000**) |
| `slope_unit_areamin` | `areamin`: 最小ユニット面積 [m²]。細分化停止＋`clean` の `cleansize` に使用 | 50,000–200,000 (デフォルト **100,000**) |
| `slope_unit_cvmin` | `cvmin`: アスペクト円周分散の上限 (0–1)。これ以下なら細分化停止 | 0.25–0.5 (デフォルト **0.3**) |
| `slope_unit_rf` | `rf`: 各反復の閾値縮小係数 (整数に丸めて渡す) | 2–3 (デフォルト **2**) |
| `slope_unit_maxiter` | `maxiteration`: 反復細分化の上限 (早期収束あり) | 10–50 (デフォルト **50**) |

#### r.slopeunits ツールセットと文献の対応

本家 r.slopeunits は 4 モジュール構成:

| モジュール | 役割 | 使用 | 文献 |
|---|---|---|---|
| **r.slopeunits.create** | delineation 本体(ハーフベイスン＋アスペクト円周分散で細分化、MFD 既定) | ✅ | Alvioli 2016 |
| **r.slopeunits.clean** | `cleansize` 未満のユニットを併合/除去 | ✅ | (後処理) |
| **r.slopeunits.metrics** | 品質指標 (V·I) を計算 | (optimize 経由) | Alvioli 2016 |
| **r.slopeunits.optimize** | `cvmin`/`areamin` を範囲探索して自動最適化 | △ オプション | Alvioli 2016/2020 |

- create は**単一レベル(非 nested)**のスロープユニットを生成します(create に nested
  パラメータは存在しません)。
- **重要**: `create` 単体は最小面積を強制しません(areamin は細分化停止の閾値のみ)。
  数セルの極小ユニットを消すには **`clean`(cleansize=areamin)** が必須で、本パイプラインは
  create→clean を自動で連結します。

##### `--slope_unit_optimize 1`(GUI「🎯 Optimize」)

`r.slopeunits.optimize` による **cvmin/areamin の自動最適化**(地形ベースの目的関数
**F = V·I**, Alvioli 2016)。**地すべりインベントリは不要**(`basin` は DEM 外形を自動生成)。
`cvmin`/`areamin` の探索範囲(GUIで指定、文献既定 cvmin∈[0.05,0.25]・areamin∈[50000,200000])
内で F を最大化し、得た最適値で create+clean を実行して最終マップを作ります。
`thresh`/`rf`/`maxiter` は固定。

注意:
- **非常に遅い**(create+clean+metrics を多数回反復。小DEMで数分、実DEMで数時間規模)。
  実用は**小さな代表領域で最適値を求め、本番に流用**するのが現実的。
- **DEM は 1 m 以上必須**。metrics が解像度を整数に丸めるため、0.5 m DEM だと
  `resolution=0` でエラーになります(0.5 m は optimize OFF の create+clean なら可)。
- **地すべりポリゴンは r.slopeunits では使いません**(delineation は地形のみ)。インベントリは
  下流の susceptibility 検証で使うものです。

#### スロープユニットのベクタ出力 (SHP / GeoJSON / GPKG)

`grass` 計算と同時に、斜面ユニットを**ポリゴン**で書き出せます(海岸線クリップ対応)。
本家出力は完全区分(穴/飛び地なし)なので追加の穴埋めは不要です。詳細・GUI/CLI の
使い方は **[スロープユニットのベクタ出力](slope_units_vector_ja.md)** を参照。

選択指針:
- **MATLAB 結果を再現したい** → `mat` (既存ファイル) または `compute`
- **DEM だけで簡易に尾根/谷を出したい(GRASS 不要)** → `compute`
- **本家 r.slopeunits の忠実なスロープユニットが欲しい** → `grass`(Docker)

### 5.5 `seismic_mode`

擬似静的解析として PGA × W を Q に加算。

| 値 | 動作 |
|---|---|
| **off** | PGA = 0 |
| **uniform** | PGA = uniform_PGA を全域均一 |
| **raster** | PGA を TIFF から読込 (NaN セルは 0) |

最終 PGA は `pseudo_scaling` 倍されます。

---

## 6. 入力ファイルの種類と配置

| ファイル | 配置先 | 必須? | Python で再生可? |
|---|---|---|---|
| **DEM (.tif)** | `lib/DEM/<name>.tif` | ✅ 必須 | — (入力データ) |
| **土層厚 .mat** | `lib/soil_depth/<DEM_stem>_soil_depth.mat` | mode=1 + source=mat 時 | ✅ `--soil_depth_source compute` で生成 |
| **無成長帯 .mat** | `lib/no_grow/<DEM_stem>_no_grow.mat` | mode=1 + source=mat 時 | ✅ `--nogrow_source compute` または `grass` で生成 |
| **せん断強度パラメータ分布 .mat** | `lib/soil_strength/shear_strength.mat` | strength_mode=1 時 | ⚠ **対象地盤の強度試験データから利用者が自作**。スキーマ: `prob[N], prob_phi[N], prob_coh[N]`、Σ prob = 1 |
| **PGA TIFF** | 任意 (`lib/seismic/` 推奨) | seismic=raster 時 | — (外部入力) |

⚠️ `lib/` 配下のファイルは `.gitignore` で除外されています。サンプル DEM や .mat は
USGS 公式リポジトリ <https://code.usgs.gov/ghsc/lhp/regiongrow3d> から入手するか、
自前 DEM をドロップして Python 計算してください。

### .mat ファイルのスキーマ
**soil_depth**: キー `depth` (2D array)

**no_grow**: キー `nogrow_io`, `nogrow_idx`, `nogrow_i`, `nogrow_j`, `ridge_io`, `valley_io`

**shear_strength**: キー `prob[N]`, `prob_phi[N]`, `prob_coh[N]`

---

## 7. 出力の見方

### 7.1 ファイル
出力ディレクトリ `<out_dir>/<susname>/`:

| ファイル | 内容 |
|---|---|
| `sus_<susname>_python.tif` | susceptibility 0-100% (主出力) |
| `net_force_prob_<susname>.tif` | 引張/圧縮マップ。per-cell 正味力 q（q>0=引張, q<0=圧縮）の**符号を反転(`−q`)**して φ 確率加重した連続場 → **出力は 圧縮=正 / 引張=負** (並列モード or `--tension_compression 1`、§3.6) |
| `depth.tif` | 土層厚 [m] (`save_intermediates=1`) |
| `nogrow_io.tif` | 無成長帯マスク 0/1 |
| `PGA.tif` | PGA [g] |
| `hillshade.tif` | 陰影図 (描画用) |
| `run_summary.json` | ラン別クラスタ数 + φ/c/prob |

### 7.2 susceptibility 値の解釈

**一言で**: そのセルが landslide になる確率を 0–100 % で表したもの。
強度分布の中で「滑らせた run の事前確率を全部足した値」。

#### 数式
```
sus[i,j] = Σ_k  prob[k] · slides_final[k][i,j]
```
- `k` = 0, 1, …, 9 が 10 通りの強度ラン
- 各 run は違う `(φ_k, c_k)` を使う（弱い土〜強い土の代表値）
- **`prob[k]`** = 強度ペア `(φ_k, c_k)` が真の土の強度である事前確率（離散化された強度分布
  の bin 質量）。`shear_strength.mat` で外部入力として与え、Σ prob[k] = 1
- `slides_final[k][i,j]` = 1 なら「k 番目の run でこのセルが landslide cluster に含まれた」、0 なら含まれなかった
- `prob[k]` で重み付けして加算 → そのセルの susceptibility (= 「滑る確率」)

#### 物理的な性質：単調性

Janbu の極限平衡式は φ, c に対して単調なので、**弱い土で滑ったセルは、より強い土でも滑ら
ない方向には絶対に動きません**。つまり各セルには「ここまで強い土なら持ちこたえる」閾値
`k*` があり、

```
slides_final[k][i,j] = 1  if k ≤ k*  (弱い土)
                     = 0  if k >  k*  (k* より強い土)
```

となり、susceptibility は **弱い側からの累積確率**:

```
sus[i,j] = prob[0] + prob[1] + … + prob[k*]
```

#### 例（具体的な数値）

確率分布 `prob = [0.01, 0.04, 0.13, 0.29, 0.29, 0.13, 0.07, 0.03, 0.02, 0.005]`
（index 0 = 最弱、index 9 = 最強、中央値は index 4–5 付近）の場合、
セルが滑る最大 index `k*` ごとに sus はおおむね以下の **離散値** を取ります:

| k* (滑る最強の run) | sus | 解釈 |
|---|---|---|
| **どの run でも滑らない** | **0 %** | 最弱の土 (index 0) でも安定 → 完全に安全 |
| 0 (最弱のみ) | **1 %** | 極端に弱い土でのみ滑る → ほぼ安全 |
| 0–1 | **5 %** | 非常に弱い土で滑る → 低リスク |
| 0–2 | **18 %** | やや弱い土で滑る → 弱風化土注意 |
| 0–3 | **47 %** | 中央値より弱い土で滑る → 中程度リスク |
| 0–4 | **76 %** | 中央値の土で既に滑る → **高リスク（実用上の危険ゾーン）** |
| 0–5 | **89 %** | 中央値より強い土でも滑る → 高リスク |
| 0–6 | **96 %** | 大半の土で滑る |
| 0–7 | **99 %** | 強い土でも滑る → 非常に高リスク |
| 0–8 | **99.5 %** | 最強以外すべてで滑る |
| 0–9 (全 run) | **100 %** | 最強の土でも滑る → 確実に landslide |

10 run の場合、susceptibility は上の **11 段階の離散値しか取りません**
（30 % や 50 % のような中間値は出ない）。

#### 実用判定の目安

| sus | 判定 |
|---|---|
| `0 %` | 強度分布の範囲ではどう転んでも安定 → 対策不要 |
| `< 50 %` | 弱い側の土でのみ滑る → 監視対象 |
| `≥ 50 %` | 中央値前後の典型的な土で滑り判定 → **危険ゾーン、対策検討** |
| `100 %` | 強度に関わらず滑る → 必ず対策 |

### 7.3 統計の見方
- **>0% セル**: 1 つでも不安定とされた領域 (overestimation 寄り)
- **>50% セル**: 過半確率で不安定 (信頼度高)
- **>90% セル**: 殆ど確実に landslide を起こす領域

---

## 8. MATLAB との比較・検証

### 8.1 比較スクリプト (開発者向け)

MATLAB 参照出力 (例: `post_processing/tests/<test_id>/sus_*.tif`) と Python 出力を pixel 比較
する `compare_with_matlab.py` / `analyze_diff.py` / `verify_against_upstream.py` が
開発時には存在します。これらは **MATLAB 由来の私的出力データに依存** するため公開リポジトリには
含まれていません (`.gitignore`)。Python 移植版の数値検証を再現したい場合は、本家 USGS の
MATLAB 環境で参照 .tif を生成して同名パスに配置してください。

### 8.2 upstream MATLAB との差分
`verify_against_upstream.py` (開発時のみ) で USGS 公式リポジトリと比較した結果、
ローカル `lib/driver.m` のみ 63 行差分 (パラメータ + `sigma_s_wedge` バグ修正)。
関数群 33 ファイルは全て upstream と同一。

### 8.3 既知の Python↔MATLAB 乖離

本移植はピクセル完全一致を目標とします。以下は **意図的または追跡中**の差分です。
変更した場合は MATLAB 参照を再生成して §8.1 を再実行してください:

- **境界拡張の上限** (`--max_cell_offset`, 既定 400): MATLAB は窓拡張を無制限に
  再試行。Python はメモリ/ハング対策で上限を設け、到達クラスタは
  `terminate_reason=7` で成長打ち切り（クラスタ毎に警告）。厳密比較では上限を
  大きく設定。
- **alpha-shape 境界** (`alpha_shape_boundary`): 収縮係数→alpha の対応が MATLAB
  `boundary()` の離散ランク選択ではなく外接円半径の線形補間。MATLAB fixture 未検証、
  未解決のパリティ項目。
- **根系補強** (`--S_roots > 0`): skip-slide 判定が現クラスタの `Q_mag` のみと比較
  （MATLAB は事前確保行列全体と比較）。既定の production 設定 `S_roots=0`
  (`F_roots=0`) では無効。

`terminate_reason` コード: 0=収束, 2=成長候補なし, 3=誤差増加, 4=最大成長サイクル,
5=幾何計算失敗, 6=退化(重み), 7=境界拡張の上限 (Python 固有)。

---

## 9. トラブルシューティング

### 9.1 GDAL / rasterio 警告
```
Warning 3: Cannot find gdalvrt.xsd (GDAL_DATA is not defined)
```
動作には影響しません。気になる場合は `conda install -c conda-forge gdal` で解消。

### 9.2 Numba コンパイルエラー
古い numpy との非互換が発生することがあります:
```bash
pip install --upgrade numba numpy
```

### 9.3 Streamlit ポート競合
他のサービスが 8501 を占有している場合:
```bash
streamlit run python/gui.py --server.port 8502
```

### 9.4 解析が遅い (フルランで 60 分超)
- `--soil_strength_mode 2` で 1 ラン化 (10 分以内)
- `--run-index 9` で単一インデックスのみ計算（部分マップ出力＋寄与保存、`--aggregate` で合算 — §4.4）
- `--soil_depth_mode 2` で土層厚計算を省略

### 9.5 メモリ不足
DEM が 5000×5000 を超える場合は要 16+ GB。`cluster_io_full` バッファが m × n bool で確保されます。
巨大 DEM の確率分布ランは §4.4 の `--run-index`/`--aggregate` 分割でピークメモリを下げられます。

### 9.6 [解析中] のままハング
- ログを確認: `python/output/<susname>/` 配下に出力されます
- subprocess は Web UI 終了後も継続するため、タスクマネージャから kill 可能

### 9.7 .mat 保存ファイルが大きい
Python の savemat はデフォルト非圧縮。大容量 (~100MB) になります。気になる場合:
```python
from scipy.io import savemat
savemat(path, data, do_compression=True)
```

---

## 10. パフォーマンスのヒント

### 10.1 計算時間の目安

絶対値は DEM のサイズ・地形複雑度・CPU で大きく変動します。下表は **相対コスト感** を
把握するための目安で、概算で捉えてください。実時間は実機で計測してください。

| ステップ | 相対コスト |
|---|---|
| DEM 読込 + 勾配 | 1× (= 軽い) |
| 土層厚 (mat) | 1× |
| 土層厚 (compute, Roering 5000 yr) | ~100× (重い、JIT で短縮) |
| 無成長帯 (mat) | 1× |
| 無成長帯 (compute, acc-threshold) | ~30× |
| 無成長帯 (compute, slope units) | ~50× |
| 静水圧 + 地震 + PGA | 1× |
| 力場計算 (1+8 rotation) | ~3× |
| RegionGrow (1 ラン) | ~100–300× (クラスタ数次第) |
| 全分布 (10 ラン) | ~10× の RegionGrow |

### 10.2 高速化テクニック
1. **Numba 必須**: 入っていないと soil_depth が 50 倍以上遅くなる
2. **mat ソース**: 同じ DEM で再計算しないなら mat にするだけで前処理が秒オーダー
3. **CPU 並列**: 現状は単スレッド。`prange` 化済の Numba 関数は自動で並列化

### 10.3 アルゴリズムオプション
- `cluster_size_thresh`: 7 (デフォルト)。小さいクラスタを除外する閾値。下げると計算量大
- `max_growth_cycles`: 120 (デフォルト)。クラスタ成長の最大反復
- `rot_num`: 8 (デフォルト)。力閉合チェックの回転数。下げると高速だが精度低下

---

## 11. API リファレンス

### 11.1 driver.py
```python
from driver import run, DEFAULTS
from types import SimpleNamespace

args = SimpleNamespace(**DEFAULTS)
args.DEM_path = 'lib/DEM/<your_DEM>.tif'  # 必須
args.test_no = 1                           # 必須 (出力サブフォルダ名に使われる)
args.susname_override = 'my_run'           # 任意 (自由な名前で上書き)
run(args)
```

### 11.2 region3d.region_grow
```python
from region3d.region_grow import region_grow_fxn

result = region_grow_fxn(Z, coh, phi, gam_w, gam_dry, gam_sat, Gs,
                         W, sigma_s, sigma_s_wedge, PGA, ...)
# returns RegionGrowResult with:
#   slides_initial_io, slides_eroded_io, slides_final_io  (bool 2D)
#   cluster_idx_initial / eroded / final  (list of 1D arrays)
#   diagnostics (dict: terminate_reason, growth_cycles)
```

### 11.3 region3d.preprocessing
```python
from region3d.preprocessing import (soil_depth, ridges_valleys,
                                    fillsinks, identify_flats,
                                    flow_direction, flow_accumulation)

depth = soil_depth(Z, cellsize, endtime=5000.0, use_numba=True)
rv = ridges_valleys(Z, cellsize, ridge_acc_thresh=5, valley_acc_thresh=100)
# rv.nogrow_io, rv.ridge_io, rv.valley_io
```

### 11.4 region3d.io
```python
from region3d.io import read_dem, write_raster, load_soil_depth, load_no_grow

Z, georef = read_dem('DEM.tif')
write_raster('out.tif', sus_map, georef)
depth = load_soil_depth('soil_depth.mat')  # returns ndarray
nogrow = load_no_grow('no_grow.mat')  # returns dict
```

---

## 参考文献

- Mathews, N. W., Leshchinsky, B. A., Olsen, M. J., & Booth, A. M. (2024).
  RegionGrow3D: A Deterministic Analysis for Characterizing Discrete Three-Dimensional
  Landslide Source Areas on a Regional Scale. *JGR Earth Surface*, 129.
- Roering, J. J. (2008). How well can hillslope evolution models "explain" topography?
  *GSA Bulletin*, 120(9-10), 1248-1262.
- Hungr, O., Salgado, F. M., & Byrne, P. M. (1989). Evaluation of a three-dimensional
  method of slope stability analysis. *Canadian Geotechnical Journal*, 26(4), 679-686.
- Schwanghart, W., & Scherler, D. (2014). TopoToolbox 2 - MATLAB-based software for
  topographic analysis and modeling. *Earth Surface Dynamics*, 2, 1-7.
