# RegionGrow3D — Python 移植版 + Web UI

> 🌐 **English version**: [README.md](README.md)

USGS の MATLAB 製 [RegionGrow3D](https://code.usgs.gov/ghsc/lhp/regiongrow3d) (Mathews et al., 2024,
*JGR Earth Surface*) を Python に移植したもの。地震・水位・降雨条件を変えた
**shallow-landslide susceptibility maps** を計算します。MATLAB ライセンス不要で動く Streamlit
ベースの Web UI 付き。

オリジナル論文: Mathews et al. (2024). *RegionGrow3D: A Deterministic Analysis for
Characterizing Discrete Three‑Dimensional Landslide Source Areas on a Regional Scale*.
JGR Earth Surface.

> ⚠️ **研究コード — コミュニティによる再実装です。** 本リポジトリは USGS と
> **公認・協力関係はありません**。ソフトウェアは「現状のまま (AS IS)」提供され、
> **本家 MATLAB 版と完全に一致する保証はありません**。業務利用の前に
> [`DISCLAIMER-jp.md`](DISCLAIMER-jp.md) を必ずご確認ください。

---

## 1. クイックスタート

### 必要環境
- Windows / macOS / Linux
- Docker (推奨) または Python 3.11+ (動作確認: 3.13)

### セットアップ (Docker — 推奨)
```bash
docker build -t region3d:latest .
docker run --rm -p 8501:8501 \
  -v "$(pwd)/lib:/app/lib" \
  -v "$(pwd)/python/output:/app/python/output" \
  region3d:latest
```
または `docker compose up`。Python 環境構築不要、依存ライブラリも全てコンテナ内で
解決されるので最も再現性が高い方法です。詳細: [docs/MANUAL.md](docs/MANUAL.md#22-docker)

### セットアップ (conda)
```bash
conda env create -f environment.yml
conda activate region3d
```
`environment.yml` で全パッケージを `conda-forge` から取得します。
`rasterio` (GDAL)、`numba` (llvmlite)、`scipy`/`numpy` (BLAS) などの
ネイティブ依存ライブラリを互換性のあるバージョンで揃えます。

### セットアップ (pip のみ — フォールバック)
conda が使えない場合は純粋な pip でも動作します:
```bash
python -m venv .venv
.venv/Scripts/activate     # Windows  (POSIX 系: source .venv/bin/activate)
pip install -r requirements.txt
```
Windows では `rasterio` / `numba` は wheel で入りますが、Linux / macOS で
対応 wheel が無い場合はシステムレベルで GDAL / LLVM を別途用意する
必要があります。

### Web UI 起動 (ローカル Python 環境)
```bash
streamlit run python/gui.py
```
ブラウザで http://localhost:8501 を開きパラメータを設定して [解析開始]。

Windows で長時間動かす場合はシェル終了の影響を受けないよう **デタッチ起動**
してください — [docs/MANUAL-jp.md §3.5](docs/MANUAL-jp.md#35-デタッチ起動-windows夜通し運用) 参照。

### コマンドライン実行
```bash
python python/driver.py --DEM_path lib/DEM/<your_DEM>.tif --test_no 1
```
選択したモードに応じて追加の `--*_mat` パスを指定してください
(全引数は [`docs/MANUAL-jp.md`](docs/MANUAL-jp.md) 参照)。
出力は `python/output/<susname>/sus_<susname>_python.tif` に生成されます
(`<susname>` はゼロ埋め `--test_no` または `--susname_override` の値)。

### サンプルデータの入手
> ⚠️ **重要**: DEM (.tif) と前処理済 .mat はサイズが大きいので **このリポジトリには含まれていません** (`.gitignore`)。
> 以下のいずれかで入手してください:
>
> - **USGS 公式リポジトリ**: <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
>   - `lib/DEM/<DEM>.tif`
>   - `lib/soil_depth/<DEM>_soil_depth.mat` (任意 — Python で計算可)
>   - `lib/no_grow/<DEM>_no_grow.mat` (任意 — Python で計算可)
>   - `lib/soil_strength/shear_strength.mat` (確率分布モードに必須 — **対象地盤ごとに利用者が自作**。スキーマ: `prob[N], prob_phi[N], prob_coh[N]`、Σ prob = 1)
>   - `lib/hydro_interp/*.mat` (mode 2 用、本 Python 版では未使用)
> - 任意の GeoTIFF を `lib/DEM/` にドロップ → Web UI から選択
>
> 土層厚と無成長帯の .mat は Python で計算可
> (`--soil_depth_source compute --nogrow_source compute`、生成ファイル名 `<DEM 名>_..._python.mat`)。
> **`shear_strength.mat` は対象地盤の強度試験データから利用者が自作するファイルです**
> (`prob[N], prob_phi[N], prob_coh[N]` の 3 つの 1D 配列、Σ prob = 1)。
> 強度分布データが無い場合は `--soil_strength_mode 2 --phi_uniform 25 --coh_uniform 2`
> のように一様強度モードで実行してください。別リポジトリ
> [`Simplified_Janbu_Method_3D_2D`](https://github.com/T40O0/Simplified_Janbu_Method_3D_2D)
> でも `shear_strength.mat` を作成できます。

---

## 2. ディレクトリ構成

```
RegionGrow3d-py/
├ lib/                       # 入力データの配置先 (内容は .gitignore で非追跡)
│  ├ DEM/                    # サンプル DEM (.tif) を置く
│  ├ soil_depth/             # 土層厚 .mat (任意 — Python 再生可)
│  ├ no_grow/                # 無成長帯 .mat (任意 — Python 再生可)
│  ├ soil_strength/          # せん断強度パラメータ分布 .mat (対象地盤ごとに利用者が自作)
│  └ seismic/                # PGA ラスタ置き場
├ python/
│  ├ gui.py                  # Streamlit Web UI
│  ├ driver.py               # CLI ドライバ (CLI / UI 共通)
│  └ region3d/               # コアアルゴリズム (下の「ポート対応表」参照)
│     ├ region_grow.py
│     ├ forces.py
│     ├ boundary.py
│     ├ growth.py
│     ├ localize.py
│     ├ derivatives.py
│     ├ bwmorph.py
│     ├ matlab_compat.py
│     ├ preprocessing.py
│     ├ slopeunits.py
│     ├ runner.py
│     └ io.py
├ docs/MANUAL-jp.md          # 詳細マニュアル (日本語)
├ docs/MANUAL.md             # 詳細マニュアル (英語)
├ README-jp.md / README.md   # 本ファイル
└ .gitignore
```

> **USGS オリジナルの MATLAB ソース** (`lib/driver.m`, `lib/functions/*.m`,
> `lib/hydro_interp/*.m`, `post_processing/susceptibility_map.m`) は
> `.gitignore` で除外されており、本リポジトリには **含まれません**。
> 必要な場合は upstream <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
> から取得してください。Python 移植版は単独で完結します。

### ポート対応表 (MATLAB → Python)

| MATLAB ソース (USGS, Mathews et al. 2024) | Python 移植版 (本リポジトリ) |
|---|---|
| `driver.m` | `python/driver.py` |
| `RegionGrowFxn.m` | `python/region3d/region_grow.py` |
| `Interslice_Force.m`, `Interslice_Force_Prism.m`, `force_closure_interslice.m` | `python/region3d/forces.py` |
| `boundary_geometry_interslice.m`, `polygeom.m`, `root_force_boundary.m`, `nogrow_not_eligible.m` | `python/region3d/boundary.py` |
| `downhill_dilate.m`, `spur_test.m`, `continuity_check.m`, `update_cluster_interslice.m` | `python/region3d/growth.py` |
| `create_localized_rasters_interslice.m` | `python/region3d/localize.py` |
| `pad_DEM.m`, `gradient_prince.m`, `hillshade.m` | `python/region3d/derivatives.py` |
| `soil_depth.m`, `fillsinks.m`, `identifyflats.m`, `flowacc.m`, `ridgelines.m`, `valleys.m`, `FLOWobj.m`, `FLOWobjInv.m`, `GRIDobj.m`, `copy2GRIDobj.m` | `python/region3d/preprocessing.py` *(一部 TopoToolbox 由来 — `LICENSE` 参照)* |
| `saveraster.m` + MATLAB `geotiffread` | `python/region3d/io.py` |
| MATLAB `bwmorph`, `bwconncomp`, `bwboundaries` (Image Processing Toolbox) | `python/region3d/bwmorph.py`, `python/region3d/matlab_compat.py` |
| *(新規 — MATLAB に対応元なし)* Alvioli 2016/2025 slope unit 分割 | `python/region3d/slopeunits.py` |
| *(新規 — MATLAB に対応元なし)* Streamlit UI 用永続化 manifest | `python/region3d/runner.py` |

Python 側の各モジュールには docstring 冒頭に対応する MATLAB 関数名が
記載されているため、コードからも対応関係が辿れます。

---

## 3. 機能概要

### 対応モード (driver.m と同じ枠組み)
| モード | 値 | Python 対応 |
|---|---|---|
| `soil_moisture_mode` | 0=乾燥 / 1=静水圧 (mw) / 2=水文力学 | ✅ / ✅ / ❌ |
| `soil_depth_mode` | 1=Roering / 2=一様 | ✅ / ✅ |
| `soil_strength_mode` | 1=分布 / 2=一様 | ✅ / ✅ |
| `nogrow_mode` | 0=なし / 1=境界あり | ✅ / ✅ |
| `nogrow_source` (mode=1 時) | mat / compute (acc-threshold) / slopeunits (Alvioli 2016/2025) | ✅ / ✅ / ✅ |
| `seismic_mode` | off / uniform / raster | ✅ / ✅ / ✅ |
| `root_mode` | uniform | ✅ |

### 前処理パイプライン
土層厚と無成長帯マスクは:
- **`mat`**: 既存の `.mat` (MATLAB or Python が事前生成) を読み込み (秒オーダー)
- **`compute`**: Python が DEM から計算 (Numba JIT 化、土層厚 5000 yr ≈ 3 分、尾根/谷 ≈ 1 分)。結果は `<DEM 名>_..._python.mat` に保存され、次回ドロップダウンに自動追加。

### Web UI 主要機能
- 6 モードの radio + 直下に該当パラメータ
- 入力ファイル配置ガイド
- DEM アップロード → `lib/DEM/` 自動保存
- 出力先: 親ディレクトリ + サブフォルダ名 (任意命名) + 既存フォルダ警告
- 進捗バー + ETA + ログストリーム
- 計算中は サイドバー全ロック + 警告バナー + 停止ボタン
- 結果タブ:
  - 🗺 マップ (susceptibility / 無成長帯 / 土層厚 / PGA、陰影図オーバーレイ)
  - 📊 統計 (有効セル数、>50% 比率、ラン別クラスタ数)
  - 📈 ヒストグラム (正のセルの分布 + CDF)
  - GeoTIFF ダウンロード

---

## 4. ライセンス

本リポジトリは **デュアルライセンス**:

| 範囲 | ライセンス | 詳細 |
|---|---|---|
| 既定 (大多数のファイル) | **CC0 1.0 Universal** (パブリックドメイン) | [`LICENSE-CC0`](LICENSE-CC0) |
| `python/region3d/preprocessing.py` の TopoToolbox 由来関数 (`fillsinks` / `identify_flats` / `flow_direction` / `flow_accumulation` ほか) | **GPL-3.0-or-later** | [`LICENSE-GPL`](LICENSE-GPL) |

各ファイル冒頭の SPDX 識別子で個別ライセンスを明示しています。

詳細・利用条件・GPL 対象関数の一覧は [`LICENSE`](LICENSE) を参照してください。

- オリジナル MATLAB (USGS RegionGrow3D): CC0 1.0 — <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
- TopoToolbox (Schwanghart & Scherler 2014): GPL-3.0 — <https://github.com/wschwanghart/topotoolbox>

> ⚠️ GPL-3.0 対象関数を含む結合作品 (例: コンテナイメージ、PyPI パッケージ等) を再配布する場合は GPL のソース提供義務が発生します。CC0 部分のみを切り出して別プロジェクトで使う場合は CC0 のままで自由です。

---

## 5. クレジット
- オリジナルアルゴリズム: Nicolas W. Mathews 他 (USGS)
- TopoToolbox: Wolfgang Schwanghart
- Python 移植 / Web UI: 本リポジトリ

---

## 6. References

- Mathews, N. W., Leshchinsky, B. A., Olsen, M. J., & Booth, A. M. (2024).
  *RegionGrow3D: A Deterministic Analysis for Characterizing Discrete
  Three-Dimensional Landslide Source Areas on a Regional Scale.*
  Journal of Geophysical Research: Earth Surface, 129.
  USGS 上流コード: <https://code.usgs.gov/ghsc/lhp/regiongrow3d>
- Schwanghart, W., & Scherler, D. (2014). *TopoToolbox 2 — MATLAB-based
  software for topographic analysis and modeling in Earth surface sciences.*
  Earth Surface Dynamics, 2, 1–7. <https://github.com/wschwanghart/topotoolbox>
- Alvioli, M., Marchesini, I., Reichenbach, P., Rossi, M., Ardizzone, F.,
  Fiorucci, F., & Guzzetti, F. (2016). *Automatic delineation of
  geomorphological slope units with `r.slopeunits` v1.0 and their
  optimization for landslide susceptibility modeling.* Geoscientific
  Model Development, 9, 3975–3991.
  (および `python/region3d/slopeunits.py` が参照している 2025 年版 r.slopeunits)
- Hungr, O. (1989). *An extension of Bishop's simplified method of slope
  stability analysis to three dimensions.* Géotechnique, 39(4), 559–562.
- `shear_strength.mat` 作成用の姉妹リポジトリ:
  <https://github.com/T40O0/Simplified_Janbu_Method_3D_2D>
