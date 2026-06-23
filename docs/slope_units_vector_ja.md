# スロープユニット (GRASS r.slopeunits) と ベクタ出力

no-grow ソース `grass`(`--nogrow_source grass`)は、本家 **GRASS GIS r.slopeunits**
を Docker コンテナ内で実行してスロープユニットを delineation し、その境界を no-grow
マスクに変換します。同時に斜面ユニットを**ポリゴン(SHP/GeoJSON/GPKG)**で書き出し、
任意で**海岸線クリップ**できます。

> 旧 Python ポート(`nogrow_source='slopeunits'`)と穴埋め/完全区分の後処理は廃止
> しました。本家との一致度が低かったため、忠実さを要する用途は本家 `grass` を使います。

## 前提: GRASS 入り Docker

`grass` は GRASS + r.slopeunits アドオンを必要とします。これらは**プロジェクトの
Docker イメージに同梱**されています([Dockerfile](../Dockerfile) が `grass-core`/
`grass-dev`/`git` を入れ、ビルド時に `g.extension r.slopeunits` でアドオンを焼き込み)。
よって `grass` は **Docker 上で実行**してください(ローカル Windows GRASS は addon の
配布 404・非 ASCII パス・大規模 DEM で不安定)。

```bash
docker compose up -d --build     # GUI → http://localhost:8501
# CLI 例:
docker compose run --rm region3d python python/driver.py \
  --DEM_path lib/DEM/dem_afterEQ_5m.tif --test_no 1 \
  --nogrow_mode 1 --nogrow_source grass \
  --slope_unit_thresh 500000 --slope_unit_areamin 100000 \
  --slope_unit_cvmin 0.3 --slope_unit_rf 2 --slope_unit_maxiter 50 \
  --soil_strength_mode 2 --phi_uniform 30 --coh_uniform 5
```

## 処理の流れ

1. `r.slopeunits.create`(MFD 既定)で delineation。
2. **`r.slopeunits.clean`(`cleansize=areamin`)** で最小面積未満を併合/除去
   — create だけでは数セルの極小ユニットが残るため必須([grass_runner_exec.py](../python/grass_runner_exec.py))。
3. 出力ラスタを取得し、境界(ユニット ID が隣接で異なる線)を no-grow マスク化
   ([grass_slopeunits.py](../python/region3d/grass_slopeunits.py) `nogrow_from_units`)。
4. ベクタ出力 ON 時、ユニットを連結成分ごとにポリゴン化(本家出力は完全区分なので
   穴/飛び地なし)→ 任意で海岸線クリップ → 書き出し。

出力:
- `lib/no_grow/<DEM>_no_grow_slopeunits_grass.mat`(no-grow マスク)
- `lib/no_grow/<DEM>_no_grow_slopeunits_grass_units.<fmt>`(ポリゴン)
- 属性: `unit_id` / `part`(連結成分番号)/ `n_cells` / `area_m2`、CRS は DEM と同じ

## GUI(no-grow = 🟢 GRASS r.slopeunits 選択時)

- thresh / areamin / cvmin / rf / maxiter スライダ
- 🧩 ベクタ出力 ON/OFF と形式(.shp/.geojson/.gpkg)
- ✂ **Clip to coastline**: `lib/coastline/` に置いた**陸(または海)ポリゴン**を選択。
  斜面ポリゴンを海岸線で止める。「the clip vector is SEA」で海ポリゴンとして減算。

## 海岸線クリップ(なぜ必要か)

post-EQ DEM は海面付近(標高≈0)のセルが有効値として残り、湾を覆う巨大ユニットが
できることがある。陸ポリゴン(例: 国土数値情報 行政区域 N03)で `intersection`
すると、DEM の標高に依存せず**実海岸線で**海域を除去できる
([clip_gdf](../python/region3d/vectorize.py))。クリップは「海岸を跨ぐポリゴンだけ」を
交差するため高速。

## パラメータと文献

`create`/`clean` のパラメータ意味と r.slopeunits ツールセット(create/clean/metrics/
optimize)と文献(Alvioli 2016/2020)の対応は、**[MANUAL-jp.md §5.4](MANUAL-jp.md)** を参照。
`optimize`(地形ベースの cvmin/areamin 自動最適化、F=V·I)は `--slope_unit_optimize 1`
/ GUI「🎯 Optimize」で利用可(遅い・DEM は ≥1m 必須・地すべりインベントリは不要)。
