# 06 Config System

## Configuration Directories
- Search space: `configs/search_space/text.yaml`, `configs/search_space/multimodal.yaml`
- Algorithm config: `configs/algorithms/default.yaml`
- Experiment config: `configs/experiments/*.yaml`

## Resolution Priority
1. Explicitly provided path
2. Environment variables (`RAGSEARCH_CONFIG` / `RAGSEARCH_CONFIG_MULTIMODAL`)
3. Default path (`configs/search_space/*`)
