# DFLASH Bench Report

## Settings
- dataset: `longbook_sum_eng`
- target_model: `/data/wanghanzhen/models/Qwen/Qwen3-8B`
- draft_model: `/data/wanghanzhen/models/z-lab/Qwen3-8B-DFlash-b16`
- max_new_tokens: `2048`
- attention_backends: `flashinfer`
- tp_size: `2`
- concurrencies: `1`
- questions_per_concurrency: `base=10`
- device_sm: `90`
- is_blackwell: `False`
- skip_baseline: `True`
- drop_first_batch: `true`

## Backend: `flashinfer`

### Baseline output tok/s
| conc | 1 |
| --- | --- |
| value | N/A |

### DFLASH output tok/s
| conc | 1 |
| --- | --- |
| value | 54.65 |

### Speedup (DFLASH / baseline)
| conc | 1 |
| --- | --- |
| value | N/A |

### DFLASH acceptance length
| conc | 1 |
| --- | --- |
| value | 1.251 |

