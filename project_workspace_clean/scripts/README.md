# 辅助诊断脚本说明

> 本目录包含项目中的一次性诊断脚本、消融辅助脚本和机制探索脚本。  
> 这些脚本不在主线流程中使用，但在特定实验或问题排查时有价值。

---

## 脚本清单

| 脚本 | 用途 | 对应主线 |
|------|------|---------|
| `align_road_support_for_ablation.py` | 对齐路网特征支持度，消融实验预处理 | 消融辅助 |
| `apply_causal_context_calibration.py` | 应用因果上下文校准（Stage2 校准） | 机制探索 |
| `apply_h3_roadcell_projection.py` | H3 r8 路网格投影 | ⚠️ 旧路线（H3 消融用） |
| `diagnose_stage2_chain.py` | Stage2 链路诊断，排查上下文生成问题 | 诊断 |

---

## 使用说明

- `apply_h3_roadcell_projection.py`：仅在 H3 消融实验时使用，**不在主线数据流中调用**
- `diagnose_stage2_chain.py`：当 Stage2 输出异常时运行诊断
- 其余脚本按需运行，不影响主线训练/评测流程
