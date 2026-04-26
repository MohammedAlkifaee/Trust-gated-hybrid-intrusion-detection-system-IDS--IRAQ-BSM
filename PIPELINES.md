# VANET IDS Pipelines

This codebase now exposes three explicit paths.

## 1. Offline Dataset Pipeline
Use this when you already have extracted CSV data.

1. Extract from F2MD:
```bash
python vanet_ids_v2.py extract --extract_root /home/instantf2md/F2MD/f2md-results/LuSTNanoScenario-ITSG5 --extract_out /tmp/features.csv
```
2. Add `attack_id` labels:
```bash
python vanet_ids_v2.py label --label_input /tmp/features.csv --attack_id 13 --label_output /tmp/features_labeled.csv
```
3. Train thesis-aligned RSU multi-head models:
```bash
python vanet_ids_v2.py train-rsu --input /tmp/features_labeled.csv --models_dir ./release_v3
```
4. Detect offline with the trained release:
```bash
python vanet_ids_v2.py detect-offline --models_dir ./release_v3 --input /tmp/features_labeled.csv --output /tmp/detect_offline.csv
```
5. Verify:
```bash
python vanet_ids_v2.py verify --labels /tmp/features_labeled.csv --detect_csv /tmp/detect_offline.csv --outdir ./verify_out
```

## 2. Realtime Pipeline With Live F2MD Scenario
Use this when F2MD is producing live `.bsm` files.

1. Start daemon:
```bash
python vanet_ids_v2.py start-daemon
```
2. Run scenario:
```bash
python vanet_ids_v2.py run-scenario
```
3. Start live detection against the raw BSM directory:
```bash
python vanet_ids_v2.py detect-live --models_dir ./release_v3 --input /home/instantf2md/F2MD/f2md-results/LuSTNanoScenario-ITSG5 --source_kind raw-dir --output ./detect_live.csv
```

The live runtime now uses the trained RSU multi-head release bundle:
- OBU plausibility and consistency flags
- engineered RSU features
- general, pos/speed, replay-stale, DoS, DoS IF, sybil, and integrity heads
- stacking meta-classifier for `p_final`
- sender trust updates during runtime
- adaptive sender threshold for the final decision

## 3. RSU Multi-Head Trainer Pipeline
Two entrypoints are kept:

CLI:
```bash
python rsu_trainer_all_in_one_v7.py --csv /tmp/features_labeled.csv --out-dir ./release_v3 --train-family all
```

GUI:
```bash
python rsu_trainer_all_in_one_v7.py
```

## Artifact Layout

The thesis-aligned release bundle is saved like this:

```text
release_v3/
  manifest.json
  training_report.json
  features.json
  scaler.joblib
  general_head.joblib
  general_calibrator.joblib
  pos_speed_head.joblib
  pos_speed_calibrator.joblib
  dos_head.joblib
  dos_calibrator.joblib
  dos_iforest.joblib
  sybil_head.joblib
  sybil_calibrator.joblib
  integrity_head.joblib
  integrity_calibrator.joblib
  replay_lstm.keras
  replay_config.json
  meta_classifier.joblib
  trust_config.json
  obu_thresholds.json
```

## Migration Note

- Previously, `vanet_ids_v2.py` contained a simplified RandomForest + trust fusion runtime that was mostly offline-oriented.
- Now, the runtime path uses the thesis-aligned RSU multi-head release bundle whenever the model directory contains `manifest.json` with `artifact_family = rsu_multi_head_v3`.
- Older `release_v2` directories are still detectable offline through an explicit fallback named `legacy_simple_runtime`.
