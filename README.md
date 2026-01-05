### Install Instructions

Docker is preferred:
```
    docker compose up -d
```

### Test all the models:

```
conda activate dguafas
python test_model.py --model-name=DGUA_FAS --path="./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar"
conda deactivate
conda activate gacdfas
python test_model.py --model-name=GACD_FAS --path="./pretrained_models/GACD_FAS/resnet18_pICM2O_best.pth"
conda deactivate
conda activate jpdfas
python test_model.py --model-name=JPD_FAS --path="./pretrained_models/JPD_FAS/full_resnet50.pth"
conda deactivate
conda activate lmfdfas
python test_model.py --model-name=LMFD_FAS --path="./pretrained_models/LMFD_FAS/icm_o.pth"
conda deactivate
conda activate flipfas
python test_model.py --model-name=FLIP_FAS --path="./pretrained_models/FLIP_FAS/msu_flip_mcl.pth.tar"
conda deactivate
```

### Get real and attack scores:

General format:
```
python -m DGUA_FAS --rdir="/path/to/dataset/" -ckpt "/path/to/pretrained/models" -edir "/path/to/director/to/store/scores/"
```

Example:
```
python evaluation.py -m DGUA_FAS --rdir="/cluster/nbl-users/Shreyas-Sushrut-Raghu/PAD_Survillance_DB/J7_NG/" \
-ckpt "./pretrained_models/DGUA_FAS/I&C&MtoO/best_model.pth.tar" -edir "./tmp/DGUA_FAS/pad_surveillance/j7ng"
```

### Adding a New SOTA Model

This section describes how to integrate a new state-of-the-art (SOTA) face anti-spoofing model into the benchmark.

#### 1. Directory Structure

Create a new directory under `sotas/` for your model:

```
sotas/NEW_MODEL_FAS/
├── __init__.py              # Standard interface exports
├── inference.py             # Docker-compatible standalone script
├── environment.yml          # Conda environment specification
├── Dockerfile               # Container setup
├── model.py                 # Model architecture (or nets/, etc.)
├── README.md                # Original paper documentation
└── [other model files]      # Additional implementation files
```

#### 2. Required Interface Implementation

Your `sotas/NEW_MODEL_FAS/__init__.py` must implement three mandatory functions:

```python
def get_model(config: Dict[str, Any], log: Logger, **kwargs) -> torch.nn.Module:
    """
    Initialize and return the model.
    Args:
        config: Configuration dictionary
        log: Logger instance
        **kwargs: Additional arguments (typically includes 'path' for checkpoint)
    Returns:
        Initialized model (optionally loaded with pretrained weights)
    """
    pass

def get_scores(data_loader: DataLoader, model: torch.nn.Module, log: Logger,
               position: Optional[int] = 0) -> Dict[str, List[float]]:
    """
    Run inference and return scores.
    Args:
        data_loader: DataLoader containing images
        model: The initialized model
        log: Logger instance
        position: Progress bar position (for parallel execution)
    Returns:
        Dictionary with keys "attack" and "real", each containing list of scores
    """
    pass

def transform_image(fname: str) -> torch.Tensor:
    """
    Preprocess an image for inference.
    Args:
        fname: Path to image file
    Returns:
        Preprocessed image tensor
    """
    pass
```

Your `sotas/NEW_MODEL_FAS/inference.py` must:
- Accept CLI arguments: `-ckpt` (checkpoint path), `-i` (input JSON file), `-o` (output JSON file), `--paths` (image paths)
- Read image paths from input JSON, run inference, and write scores to output JSON
- This is the script Docker containers execute during evaluation

#### 3. Code Integration Points

You must modify the following files to register your model:

**a) `util/__init__.py`** - Add to SOTA enum:
```python
class SOTA(Enum):
    # ... existing models ...
    NEW_MODEL_FAS = "NEW_MODEL_FAS"
```

**b) `sotas/__init__.py`** - Add to all getter functions:
```python
def get_model(model: SOTA, config: Dict, log: Logger, **kwargs) -> Module:
    if model == SOTA.NEW_MODEL_FAS:
        from sotas.NEW_MODEL_FAS import get_model
        return get_model(config, log, **kwargs)
    # ... existing models ...

def get_score_function(model: SOTA):
    if model == SOTA.NEW_MODEL_FAS:
        from sotas.NEW_MODEL_FAS import get_scores
        return get_scores
    # ... existing models ...

def get_transform_function(model: SOTA):
    if model == SOTA.NEW_MODEL_FAS:
        from sotas.NEW_MODEL_FAS import transform_image
        return transform_image
    # ... existing models ...
```

**c) `eval_loop.py`** - Add checkpoint paths to MODELS_CHECKPOINTS:
```python
MODELS_CHECKPOINTS = {
    # ... existing models ...
    SOTA.NEW_MODEL_FAS: {
        "protocol1": "/root/pretrained_models/NEW_MODEL_FAS/checkpoint1.pth",
        "protocol2": "/root/pretrained_models/NEW_MODEL_FAS/checkpoint2.pth",
    },
}
```

**d) `docker-compose.yaml`** - Add Docker service:
```yaml
new_model_fas:
  container_name: new_model_fas
  build:
    context: .
    dockerfile: sotas/NEW_MODEL_FAS/Dockerfile
  volumes:
    - ./sotas/NEW_MODEL_FAS:/root/code
    - ./pretrained_models:/root/pretrained_models
    - ./results:/root/results
  deploy:
    resources:
      reservations:
        devices:
          - driver: nvidia
            count: 1
            capabilities: [gpu]
  stdin_open: true
  tty: true
```

#### 4. Pretrained Models

Place your pretrained model checkpoints in:
```
pretrained_models/NEW_MODEL_FAS/
├── checkpoint1.pth
├── checkpoint2.pth
└── ...
```

These paths should match what you specified in `MODELS_CHECKPOINTS` in `eval_loop.py`.

#### 5. Testing Your Integration

**Local testing with conda:**
```bash
conda activate your_env
python test_model.py --model-name=NEW_MODEL_FAS --path="./pretrained_models/NEW_MODEL_FAS/checkpoint.pth"
conda deactivate
```

**Docker testing:**
```bash
docker compose up -d
docker exec new_model_fas python /root/code/inference.py -ckpt /root/pretrained_models/NEW_MODEL_FAS/checkpoint.pth -i input.json -o output.json
```

**Evaluation pipeline:**
```bash
python evaluation.py -m NEW_MODEL_FAS --rdir="/path/to/dataset/" \
-ckpt "./pretrained_models/NEW_MODEL_FAS/checkpoint.pth" -edir "./results/NEW_MODEL_FAS/"
```

#### 6. Quick Reference Checklist

- [ ] 1. Create `sotas/NEW_MODEL_FAS/` directory with all required files (see §1)
- [ ] 2. Implement `get_model()`, `get_scores()`, `transform_image()` in `__init__.py` (see §2)
- [ ] 3. Implement `inference.py` with required CLI interface (see §2)
- [ ] 4. Update `util/__init__.py` - add to SOTA enum (see §3a)
- [ ] 5. Update `sotas/__init__.py` - add to all getter functions (see §3b)
- [ ] 6. Update `eval_loop.py` - add checkpoint paths to MODELS_CHECKPOINTS (see §3c)
- [ ] 7. Update `docker-compose.yaml` - add Docker service (see §3d)
- [ ] 8. Create `environment.yml` with conda dependencies
- [ ] 9. Create `Dockerfile` for containerization
- [ ] 10. Place pretrained models in `pretrained_models/NEW_MODEL_FAS/` (see §4)
- [ ] 11. Test with `test_model.py` (see §5)
- [ ] 12. Test Docker container execution (see §5)
- [ ] 13. Verify evaluation pipeline runs successfully (see §5)

#### 7. Reference Examples

For implementation guidance, refer to existing models:
- **Simple structure:** `sotas/JPD_FAS/` - straightforward implementation
- **Complex structure:** `sotas/DGUA_FAS/` - includes fine-tuning support and multiple protocols
