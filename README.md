# Backdoor as Probe

> [Zhongqi Wang](https://scholar.google.com.hk/citations?hl=zh-CN&user=Gi1brbgAAAAJ), [Jie Zhang*](https://scholar.google.com.hk/citations?user=hJAhF0sAAAAJ&hl=zh-CN), [Nie Sen](https://scholar.google.com/citations?user=fKQnNncAAAAJ&hl=zh-CN), Zhiyu Chen, [Shiguang Shan](https://scholar.google.com.hk/citations?hl=zh-CN&user=Vkzd7MIAAAAJ), [Xilin Chen](https://scholar.google.com.hk/citations?hl=zh-CN&user=vVx2v20AAAAJ)
>
> *Corresponding Author


<div align=center>
<img src='https://github.com/Robin-WZQ/Backdoor-as-Probe/blob/main/images/Intro.png' width=800>
</div>

Backdoors and adversarial perturbations are usually studied as separate security failures. We instead ask whether a model owner can reserve a private steering channel and use it to repair adversarial inputs at test time.

## 🧭 Requirements

- Linux
- Python 3.10 or newer
- PyTorch 2.1 or newer
- torchvision 0.16 or newer
- transformers 4.37 or newer

Create an isolated environment and install the dependencies from the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For a CUDA installation, install the PyTorch build matching the local CUDA runtime first, then install the remaining dependencies.

The default base model identifier is `openai/clip-vit-base-patch16`. A local checkpoint can be selected globally:

```bash
export BAP_BASE_MODEL_ID=/path/to/clip-vit-base-patch16
```

Pass `--local_files_only` to model-loading commands when network downloads must be disabled.

## External Inputs

A labeled dataset uses the following layout:

```text
/path/to/data/tables/clean/General/ImageNet/
  images/
  labels.csv
  classes.json
```

After probe implantation, the generated artifact has the following layout:

```text
runtime/artifacts/output_low_energy/
  edited_model/
  trigger.pt
  delta_y.pt
  delta_weight.pt
  clean_features_layers_6_token_0.pt
  output_low_energy_basis.pt
  implant_meta.json
  target_response_q95/
    detector.json
    threshold.pt
    calibration_scores.csv
```


## Usage


```bash
python -m BaP \
  --stages implant,attack,detect,directions,correct,evaluate,gate \
  --model_id /path/to/clip-vit-base-patch16 \
  --calibration_dir /path/to/data/train_data \
  --dataset_root /path/to/data/tables/clean/General/ImageNet \
  --artifact_dir runtime/artifacts/output_low_energy \
  --run_dir runtime/runs/imagenet \
  --trigger_path /path/to/trigger.pt \
  --device cpu \
  --limit 1 \
  --dry_run
```

### Implant the Default Probe

```bash
CUDA_VISIBLE_DEVICES=0 python -m BaP.probe.implant \
  --model_id /path/to/clip-vit-base-patch16 \
  --calibration_dir /path/to/data/train_data \
  --output_dir runtime/artifacts/output_low_energy \
  --trigger_path /path/to/trigger.pt \
  --probe_prompt "a white teapot" \
  --mlp_layer 6 \
  --token_index 0 \
  --input_low_energy_rank 256 \
  --output_low_energy_rank 32 \
  --trigger_norm 5 \
  --target_scale 40 \
  --quantile 0.95 \
  --device cuda:0 \
  --local_files_only
```

### Build the Benign Correction Direction

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/build_global_correction_directions.py \
  --edited_model_dir runtime/artifacts/output_low_energy/edited_model \
  --benign_dir /path/to/data/train_data \
  --clean_features_path runtime/artifacts/output_low_energy/clean_features_layers_6_token_0.pt \
  --mlp_layers 6 \
  --output runtime/artifacts/output_low_energy/global_correction_directions.pt \
  --device cuda:0 \
  --local_files_only
```

### Generate an Attack

The attack is generated against the current implanted Edited CLIP:

```bash
CUDA_VISIBLE_DEVICES=0 python -m BaP.attacks.generate \
  --attack pgd \
  --model_id runtime/artifacts/output_low_energy/edited_model \
  --image_dir /path/to/data/tables/clean/General/ImageNet/images \
  --labels_csv /path/to/data/tables/clean/General/ImageNet/labels.csv \
  --classes_json /path/to/data/tables/clean/General/ImageNet/classes.json \
  --dataset_name ImageNet \
  --output_dir runtime/runs/imagenet/attack \
  --epsilon 0.00392156862745 \
  --steps 10 \
  --alpha_scale 2.5 \
  --device cuda:0 \
  --local_files_only
```

### Evaluate the Target-Response Detector

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/benchmark_target_response_auc.py \
  --edited_model_dir runtime/artifacts/output_low_energy/edited_model \
  --delta_y_path runtime/artifacts/output_low_energy/delta_y.pt \
  --detector_path runtime/artifacts/output_low_energy/target_response_q95/detector.json \
  --clean_dir /path/to/data/tables/clean/General/ImageNet/images \
  --attack_dir runtime/runs/imagenet/attack \
  --attack_summary_json runtime/runs/imagenet/attack/pgd_generation_summary.json \
  --require_attack_provenance \
  --output_dir runtime/runs/imagenet/target_response_auc \
  --device cuda:0 \
  --local_files_only
```

### Run E2R1 Correction

```bash
CUDA_VISIBLE_DEVICES=0 python -m BaP.correction.current \
  --edited_model_dir runtime/artifacts/output_low_energy/edited_model \
  --image_dir runtime/runs/imagenet/attack \
  --clean_features_path runtime/artifacts/output_low_energy/clean_features_layers_6_token_0.pt \
  --global_direction_artifact runtime/artifacts/output_low_energy/global_correction_directions.pt \
  --output_dir runtime/runs/imagenet/correction \
  --device cuda:0 \
  --local_files_only
```

### Run the Complete Pipeline

```bash
CUDA_VISIBLE_DEVICES=0 python -m BaP \
  --stages implant,attack,detect,directions,correct,evaluate,gate \
  --model_id /path/to/clip-vit-base-patch16 \
  --calibration_dir /path/to/data/train_data \
  --dataset_root /path/to/data/tables/clean/General/ImageNet \
  --dataset_name ImageNet \
  --artifact_dir runtime/artifacts/output_low_energy \
  --run_dir runtime/runs/imagenet \
  --trigger_path /path/to/trigger.pt \
  --gpu_id 6 \
  --device cuda:0 \
  --local_files_only
```

### Implanted Model

We upload the implanted model at [here], you can directly download and detect adversarial samples.


🤝 Feel free to discuss with us privately!
