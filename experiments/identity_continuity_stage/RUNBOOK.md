# 身份连续性正式实验手动运行手册

日期：2026-09-06。状态：`HOLDOUT_QA_PASS_READY_FOR_FORMAL_TRAINING`。两份冻结测试集已经采集并通过[质量核查](HOLDOUT_QA_2026-09-06.md)；稳定ID性能假设仍未判定。下列正式训练长任务由用户手动启动。

## 1. 先采集，完成数据 QA，再训练

在项目根目录依次运行：

```powershell
python code/collect_pilot.py --config config/identity_holdout_merge_collect.json
python code/collect_pilot.py --config config/identity_holdout_roundabout_collect.json
```

两场景分别采集 1,000 pair。输出为：

- `data/identity_continuity_stage/merge_holdout_1000/object_twinworld_holdout_1000.npz`
- `data/identity_continuity_stage/roundabout_holdout_1000/object_twinworld_holdout_1000.npz`

采集默认自动恢复；纯文本每 20 accepted pair 显示接纳量、尝试次数、终止分类、elapsed 和 ETA，每 100 accepted pair 保存 checkpoint。恢复边界及最终 NPZ/stats 原子完成校验见 [HOLDOUT_COLLECTION.md](HOLDOUT_COLLECTION.md)。不要并发运行同一场景，不要在采集中更改配置。

采集完成后执行以下只读完成校验，不创建仿真环境，也不训练：

```powershell
python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'code'); from collect_pilot import validate_completed_collection; configs=[json.loads(Path('config/identity_holdout_'+s+'_collect.json').read_text(encoding='utf-8')) for s in ('merge','roundabout')]; [validate_completed_collection(Path(c['paths']['data']),Path(c['paths']['data']).with_suffix('.stats.json'),c) for c in configs]; print('COLLECTION_COMPLETENESS_OK')"
```

检查 `.stats.json` 的 requested/accepted 均为 1,000、完整配置绑定、对象字段和全部终止原因计数；记录接受率和终止原因分布。任何缺失、文件不完整或种子冲突均须先解决，不得启动正式训练。只检查采集完整性、输入属性与哈希，不计算或查看正式测试误差，不据测试数据调整参数。

## 2. 只读配置与运行前保护

配置 schema 校验不要求冻结文件存在。正式配置在此阶段就强制固定训练种子及顺序为 `[11,22,33,44,55]`、`bootstrap_resamples` 为整数 `10000`、`confidence_level=0.95`；缺失或修改这些设置会在训练前拒绝。smoke 可保留单种子和较少重采样：

```powershell
python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'code'); from run_identity_continuity_comparison import validate_identity_config; [validate_identity_config(json.loads(Path('config/identity_continuity_'+s+'.json').read_text(encoding='utf-8'))) for s in ('merge','roundabout')]; print('FORMAL_SCHEMA_OK')"
```

数据采集完成后执行正式 preflight。该函数只读，不创建模型、训练或写输出：

```powershell
python -c "import sys,json; from pathlib import Path; sys.path.insert(0,'code'); from run_identity_continuity_comparison import preflight_identity_run; [print(s,preflight_identity_run(json.loads(Path('config/identity_continuity_'+s+'.json').read_text(encoding='utf-8')))) for s in ('merge','roundabout')]"
```

当前未采集时，预期明确报 `冻结数据缺失 (frozen test data missing)`。正式配置必须填写 `frozen_test_data`，不会回退 `development_test`。preflight 还检查：

- 源数据和冻结数据的采集种子不同；所有源 episode（包括原 split 2）与冻结 episode 的环境 seed 集合不重叠。两文件均由 `collect_pilot` 生成，episode seed 都直接传入环境 reset，因此可直接比较。
- `object_features_json` 都精确匹配 8 维字段：`presence, is_ego, x, y, vx, vy, cos_heading, sin_heading`。
- 对象容量、历史长度、未来 horizon 与内嵌元数据和数组维度一致，且两文件一致；场景与正式配置一致。
- 正式输出目录为空，或包含与本次完整配置、两份数据 SHA256、实现文件哈希匹配的 `experiment_manifest.json`。实现签名包含 `ac_twin/statistics.py`，只改统计方法也会拒绝复用旧结果。每个种子的签名在任何训练开始前再次核对；不完整签名、旧签名均拒绝覆盖。

旧 collector 将 JSON 标量保存为 object dtype；元数据兼容读取只适用于本项目可信的本地采集 NPZ，不应提供来历不明的 NPZ。

正式源数据仅 `split==0` 用于优化、`split==1` 用于验证早停和模型选择；归一化仅拟合 source train。源 `split==2` 不作为正式测试。冻结档案全部 1,000 episode 用于最终测试，不使用它内部的 split 标签选择子集。

## 3. 用户手动运行正式训练

在采集和数据 QA 均通过、配置与实现冻结后运行：

```powershell
python code/run_identity_continuity_comparison.py --config config/identity_continuity_merge.json --device cuda
python code/run_identity_continuity_comparison.py --config config/identity_continuity_roundabout.json --device cuda
```

仅当 CUDA 不可用时，将相应命令的 `--device cuda` 改为 `--device cpu`。不要因测试表现更改配置、容量、训练种子、epoch 或选用某个种子。每场景为四条件 × 两容量 × 五训练种子，共 40 个模型运行。

默认启用 resume，无需加参数。进度为纯文本，含 scene、tier、condition、seed、epoch、train/val、elapsed 和 ETA。正常休眠恢复后进程通常继续；若进程退出，重新运行同一命令。已完成且签名一致的种子显示 `resumed=complete`；签名匹配但未完成的种子显示 `resumed=incomplete restart_seed=true` 并从该种子开始重新训练，不从 epoch 中间恢复。不要并发运行同一输出目录。

正式运行期间终端不显示测试 ADE/FDE。各模型只用训练/验证数据完成选择，之后产生固定测试预测；整个场景完成后才生成总 `report.json`。训练未全部结束前不要打开逐种子 `*_run.json`、预测或测试指标，也不要用已完成场景的测试结果改变另一个场景的训练。恢复只复用固定预测，不增选测试模型。实现文件变化也会改变签名；请使用独立新实验目录保留旧证据，不要改写签名或删除旧证据绕过保护。

## 4. 返回报告并核对全部 QA

两场景最终报告位置：

- `experiments/identity_continuity_stage/merge/report.json`
- `experiments/identity_continuity_stage/roundabout/report.json`

同时保留 `experiment_manifest.json`、`normalization.npz`、`predictions.npz`、逐种子权重、日志、预测和 run metadata。报告保留源/冻结文件哈希、训练/验证/测试索引、每种子每 episode 的 ADE/FDE、两层配对 Bootstrap 和准确容量。

单场景报告固定为 `PENDING_SECOND_SCENE`，绝不单独输出 PASS。程序仅把已实际完成的 preflight `data_consistency` 标为 `true`；其他 QA 默认 `PENDING`，必须由主任务在收到两个报告后依据原始产物和检查结果完成审查。需要明确通过的全部项目为：

`fairness, object_multiset, ego_slot, permutation, data_consistency, capacity, finite_values, resume, holdout_collection`

这些项目分别对应四条件完全相同输入/训练约束、对象多重集合、自车槽位、排列性质、数据一致性与种子隔离、容量误差 ≤1.5%、有限数值、签名匹配的完成复用、采集完整性。对每项保留证据、运行命令、检查结果和对应数据/实现哈希；历史 smoke QA 不能自动证明新正式产物通过。`resume` 可在全部训练完成后用相同命令确认所有 40 个运行均 `resumed=complete`，不得借此重训或调整模型。首次生成的原报告保留不动；完成审查后将报告复制到内存，在副本 `qa` 中填写经证据确认的布尔值，并保存独立的 QA 审查记录。不要为消除 PENDING 批量填写 true。

主任务调用同一 runner 中的纯函数即可汇总，无需额外 CLI 或脚本：

```python
from run_identity_continuity_comparison import decide_identity_continuity
decision = decide_identity_continuity(merge_report, roundabout_report)
```

任一 QA 缺失、非布尔 true、失败，或并非两个不同场景的独立冻结测试报告，输出 `QA_FAIL`。统计缺失、单层 CI、种子不对齐、非有限数值等输出 `INCONCLUSIVE`，不把缺失统计解释为“无差异”。

## 5. 冻结的最终判定规则

依据 [PROTOCOL.md](PROTOCOL.md)，以下保守的操作规则在正式训练前固定。主指标为 FDE；ADE 保留为辅助信息，不用于替代主指标或挑选结论。差值方向是对照减 stable，正值表示 stable 更好。

两场景、65k/83k 两容量均需完整数据；每个比较必须是五个对齐训练种子 `[11,22,33,44,55]` × 冻结 episode 的两层配对 Bootstrap，10,000 resamples、95% CI。可靠正向证据要求 CI 下界 >0、平均差 >0、至少四个种子均值 >0，且删除任一个种子后其余四个均值仍 >0。正向 CI 但种子稳定性不满足时输出 `INCONCLUSIVE`；该规则是对协议“非单一种子驱动”的保守量化，不得在看到结果后放宽。

在全部 QA/统计完整后依次判定：

1. 两场景两个容量的 `set_pool - stable` 都没有可靠正向差异（CI 下界 ≤0，包括 set_pool 更优）时优先 `PREFER_SET_POOL`。不跨容量挑结果。
2. 两场景两个容量的两个主比较（`permuted - stable`、`distance - stable`）全部可靠正向，输出 `PASS_IDENTITY_CONTINUITY`。
3. 仅一个场景在两个容量、两个主比较均满足时，输出 `SCENE_LIMITED_IDENTITY_EVIDENCE`。
4. 两场景两个容量仅 distance 比较全部满足，且无完整场景身份证据时，输出 `DISTANCE_ONLY_DIFFERENCE_NOT_IDENTITY`。
5. 其余完整但不一致或不足的证据输出 `INCONCLUSIVE`。

当前工程交付不能声称稳定身份贡献通过。只有两报告与全部 QA 返回并完成审查后，才能形成上述科研判定、追加正式结果日志；任何状态均不能自动推出因果可识别性、多车响应、安全收益或跨场景泛化。
