# 模型隔离环境登记

状态：G3 模型适配进行中；四个冻结基础模型的环境及首轮能力证据均已固定。

基础模型不得直接加入核心 `pyproject.toml`。每个模型在 `environments/<model_id>/` 维护独立 `pyproject.toml` 与 `uv.lock`，通过日值、窗口和预测账本契约与核心环境通信。

| 环境 | 预定模型 | 当前状态 | 能力门 |
|---|---|---|---|
| `timesfm_3` | 本地 TimesFM 3.0 权重 | 环境已锁定；合成与真实验证集U/M/X能力门通过 | U/M/X、9分位数、16GB 显存 |
| `chronos_2` | amazon/chronos-2 120M | 环境已锁定；合成与真实验证集U/M/X能力门通过 | U/M/X、9分位数、离线缓存 |
| `moirai_2_small` | Moirai 2.0 R small | 环境已锁定；U全时距及M/X短时距通过，长时距按证据N/A | U、M/X≤30天、9分位数 |
| `toto_2_313m` | Datadog Toto 2.0 313M | 环境已锁定；合成与两站真实验证U/M通过 | U/M、9分位数、16GB 显存 |
| `target_deep` | DLinear、PatchTST目标域模型 | 环境已锁定；合成训练/推理能力门通过 | U/M/X、三种子、训练边界 |

版本号必须依据安装当日的官方仓库和模型卡冻结，不能猜测包版本。能力探测不得读取测试真值，也不得依据测试误差决定保留或替换。

TimesFM 3.0权重采用`timesfm-non-commercial-license-v1.0`，仅用于本研究的非商业、非生产评测；代码运行时固定到Google Research官方仓库提交`8cb0628371af142e16b8c232cc9fbf667ffb12f9`。合成能力门命令：

```bash
./scripts/wsl/model-env.sh timesfm_3 python environments/timesfm_3/probe.py \
  --checkpoint weights/timesfm-3.0-pytorch \
  --output artifacts/model_probes/timesfm_3.json
```

Chronos-2采用Apache-2.0许可，运行时固定到Amazon官方仓库提交`4dbf163c2734c089cdf7da2b86fde48862ff9c6f`，权重固定到Hugging Face revision `29ec3766d36d6f73f0696f85560a422f50e8498c`。本地权重树哈希为`fe597872d0ac5dc01a8a7e4c64a5a86b3f99e4358737df8640f2088a94fcada4`。合成能力门命令：

```bash
./scripts/wsl/model-env.sh chronos_2 python environments/chronos_2/probe.py \
  --checkpoint weights/chronos-2 \
  --output artifacts/model_probes/chronos_2.json
```

Moirai-2代码采用Apache-2.0许可，权重采用CC-BY-NC-4.0许可。运行时固定到Salesforce官方仓库提交`cfd46d4510ed8896f263116f32928eede05b0a75`，权重revision为`30f43ff08c8494f4943ae1521e9d4e94a0fbb389`，本地树哈希为`15e21402ab051a61877d4c6ba4a0e1abd0d1ede33206797c9b77f758086b2089`。官方当前实现的多变量递归分支在超过64天时发生张量重排错误，因此M/X的90/180天固定记为N/A，不进行算法补丁或单变量降级。证据命令：

```bash
./scripts/wsl/model-env.sh moirai_2_small python environments/moirai_2_small/probe.py \
  --checkpoint weights/moirai-2.0-R-small \
  --output artifacts/model_probes/moirai_2_small.json
```

Toto-2.0-313m代码与权重均采用Apache-2.0许可。运行时固定到Datadog官方仓库提交`c079fd7e32424c5d33f9e267c5d32a5630242eb5`，权重revision为`a7bab288f5e95f8606f8306f86659357e1c001ef`，本地树哈希为`be24d1a406954cbfbf2b61470ce3e45de5dca9842bc9b85a2974bf8c4347fe3b`。模型要求上下文长度整除32；适配器仅在左侧加入掩码为未观测的占位值，仍只提供730天真实历史。证据命令：

```bash
./scripts/wsl/model-env.sh toto_2_313m python environments/toto_2_313m/probe.py \
  --checkpoint weights/toto-2.0-313m \
  --output artifacts/model_probes/toto_2_313m.json
```
