# RIFE

Third-party RIFE v4.x frame interpolation code + weights, used by the player's adaptive interpolation
(`streamer/rife_interp.py`, setting `interpolation` = auto; model v4.6) and by `streamer/bench_rife.py`.

## Sources

| What | From | Version |
|---|---|---|
| `vsrife_models/IFNet_HDv3_v4_6.py`, `IFNet_HDv3_v4_22_lite.py`, `IFNet_HDv3_v4_25_lite.py`, `IFNet_HDv3_v4_26.py`, `warplayer.py`, `LICENSE` | https://github.com/HolyWu/vs-rife (`vsrife/`) | tag `v5.7.0`, commit `784a6d0b87bcd1d2e20586f41d7c652c6632b438` |
| `vsrife_models/vsrife_init_reference.py.txt` | same repo, `vsrife/__init__.py` (renamed so it is never imported; it needs VapourSynth) | same commit |
| `weights/flownet_v*.pkl` | GitHub release `model` of HolyWu/vs-rife: `https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v<ver>.pkl` | release `model` |

The model files are the IFNet_HDv3 architectures of hzwer's Practical-RIFE (https://github.com/hzwer/Practical-RIFE),
lightly adapted by HolyWu (grid / flow divisor passed in, encoder `Head` split out). The `.pkl` weights are Practical-RIFE's
`flownet.pkl` files, re-hosted on vs-rife's GitHub release (Practical-RIFE itself distributes them via Google Drive /
Baidu only).

Downloaded files were not executed: no setup script was run. `rife_interp.py` imports the model module and loads the
weights with `torch.load(..., weights_only=True)` (they are plain state dicts).

## Licenses

- vs-rife: MIT (Copyright (c) 2021 HolyWu), see `vsrife_models/LICENSE`.
- Practical-RIFE (architecture + weights): MIT, https://github.com/hzwer/Practical-RIFE/blob/main/LICENSE

## Files

| File | Bytes | SHA-256 |
|---|---:|---|
| weights/flownet_v4.6.pkl | 21273159 | 008646e761f0e67cb77f0c6c44cfe3c3e5a05d9d9465311b9681ca650ce030db |
| weights/flownet_v4.22.lite.pkl | 21429303 | 8f6fb9105ba9e946762ee7190acbca3ca1cf14193eb81ca0955d492fb8558692 |
| weights/flownet_v4.25.lite.pkl | 24467405 | 81cdba223fe72a120130cc8552e5d2ecac824259d406f0c15323b3decf96b8b1 |
| weights/flownet_v4.26.pkl | 24636301 | 45c7f74156704769dc9f85cfcaf8552e1e926f9399dcfa3a553dee88fac6f53f |
| vsrife_models/IFNet_HDv3_v4_6.py | 4177 | c45aed7bf7e2d2e8b8ba733fb4ad3ef4333df301ef0dba8300da34f861679092 |
| vsrife_models/IFNet_HDv3_v4_22_lite.py | 4528 | 64067a86ef357e635e840534318afdae7ada32a9994b5ec541d3992f6c89bdba |
| vsrife_models/IFNet_HDv3_v4_25_lite.py | 4605 | ed017cc4dca920050f1a24a7efbf42eaf3c9647cc0094aacdf52e64c7b1be77e |
| vsrife_models/IFNet_HDv3_v4_26.py | 4599 | 822e04b96fcb121a742eeedc848488279eb93ae7658d958c8275293bf6358b60 |
| vsrife_models/warplayer.py | 486 | f17b5aea73676c059b4c155852217c954d5b4eb19cc08eff40b43edd6055ab73 |
| vsrife_models/LICENSE | 1063 | aea942caf766b16e6ef0b26e033fa92c0dea3124f6f7ad4a18ca1fb43e010d40 |
| vsrife_models/vsrife_init_reference.py.txt | 29898 | b0fbb78c86f50a2b0e321c973296e72f53e46c6c2a1eb0611fab696ca61b0828 |

Byte sizes of the weights match the asset sizes listed by the GitHub releases API.

## Model notes (from vs-rife's table)

| Model | Encoder (`Head`) | Pad to multiple of |
|---|---|---|
| 4.6 | no | 32 |
| 4.22.lite | yes (4 ch) | 32 |
| 4.25.lite | yes (4 ch) | 128 |
| 4.26 | yes (4 ch) | 64 |

With `scale < 1` the multiple becomes `max(m, m / scale)`.
