# DFLASH Bench Report

## Settings
- dataset: `longbook_qa_eng`
- target_model: `/data/wanghanzhen/models/Qwen/Qwen3-8B`
- draft_model: `/data/wanghanzhen/Projects/MTP/NIPS26/FlashMTP/cache/models/dflash_sample_40000_think_off_qwen3_8b_maxlen4096/epoch_6_step_29844`
- max_new_tokens: `512`
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
| value | 7.32 |

### Speedup (DFLASH / baseline)
| conc | 1 |
| --- | --- |
| value | N/A |

### DFLASH acceptance length
| conc | 1 |
| --- | --- |
| value | 1.505 |

