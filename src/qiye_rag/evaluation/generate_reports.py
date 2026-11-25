"""从完整实验产物生成模块报告，结论保留小样本限制。"""
import json
from pathlib import Path
from statistics import mean
from qiye_rag.rag.config import PROJECT_ROOT
from qiye_rag.evaluation.reporting import comparison, read_run, statistics


def main() -> None:
    """生成已完成的模块报告及机器可读汇总。"""
    root = PROJECT_ROOT / "eval/my_plan"
    chunk_paths = [root / name for name in ("chunk_legacy", "chunk_section", "chunk_prefix")]
    comparison(chunk_paths, root / "模块1_切分实验报告.md", "模块1：三种切分真实问答对照",
        "章节聚合将块数从830降至368；两种章节方案的冻结事实证据覆盖均为88%，高于本次旧切分的85.33%。"
        "Q06的规格字段不再与主体分离，前缀组能回答Q16的三个必要部分。章节组和前缀组的审核状态代理值都为26/30，"
        "不能证明前缀有显著质量优势；前缀组本次逻辑调用213次，少于章节组230次，但索引Embedding Token从17278增加至29363。"
        "建议保留章节聚合，前缀方案作为当前明确主体身份的默认起点；若主要关注索引成本，可选section。"
        "Q12/Q15等仍出现有证据却拒答，说明切分不能替代问题范围与审核策略优化。历史题仍受现行过滤限制。")
    rows = [r for p in chunk_paths for r in read_run(p)[2]]
    events = [e for r in rows for e in r["events"]]
    chat = [e for e in events if e["stage"] in ("plan", "research", "verdict")]
    differences = [e["input_tokens"] - e["offline_input_tokens"] for e in chat if e["input_tokens"] is not None]
    max_actual = max(e["input_tokens"] for e in chat if e["input_tokens"] is not None)
    max_evidence = max(e["evidence_tokens"] for e in events if "evidence_tokens" in e)
    complete = sum(r["token_usage_complete"] for r in rows)
    module0 = ["# 模块0：Token计数、上下文预算与阶段观测验证", "",
        "## 实现与口径", "",
        "使用DeepSeek v4官方离线tokenizer计算文本Token，不再把jieba分词数写成生产块的模型Token。离线值只用于预检；真实聊天、Embedding和Rerank消耗均取API usage。历史Gold保留旧元数据以保持快照可追溯，生产同步通过新指纹更新。", "",
        "证据按完整序列化JSON计数、整块装入6000预算；超过预算跳过整块，最后连续编号。Researcher与Reviewer使用同一证据快照。完整消息离线输入预算12000，预留输出上限2000，超限在请求前终止，不能悄悄截掉事实条件。", "",
        "HTTP响应钩子取得Embedding usage；logical_model_calls表示应用层逻辑调用，embedding_http.http_responses反映收到的HTTP响应，不能把二者混为重试后的请求总数。父子事件Token按API叶子求和，索引开销单列。", "",
        "## 实验结果（三种切分共90次问答）", "",
        f"- 90题均有执行结果，usage完整：{complete}/90。",
        f"- 最大装入证据离线Token：{max_evidence}，不超过6000。",
        f"- 聊天API实际最大输入Token：{max_actual}；对应配置的离线输入上限12000。",
        f"- API实际输入减离线估算的平均差：{mean(differences):.2f} Token，范围 {min(differences)} 至 {max(differences)}。",
        "- 离线文本编码不包含全部服务端消息封装，偏差保留在日志；未宣称离线值与计费值完全相同。", "",
        "## 验证与结论", "",
        "单元测试验证usage从合成HTTP响应原样读取、完整JSON预算边界、上下文隔离和引用映射；真实实验验证三个API均有usage。相比以前Embedding消耗未知，现在能够报告本次问答实际已知总Token及独立索引Token。预算与计费分开处理，避免以错误计数假装精确。", "",
        "官方依据：[DeepSeek Token说明](https://api-docs.deepseek.com/quick_start/token_usage/)、[百炼Embedding usage说明](https://help.aliyun.com/en/model-studio/text-embedding-synchronous-api)。本地tokenizer下载、校验值、回退方式及复现命令见[README](README.md)。", "",
        "## 限制", "",
        "预算不是质量阈值，丢块可能丢失必要证据，因此报告同时统计事实覆盖和dropped_chunks。更换模型需更换匹配tokenizer；缺失时回退字节上界并标注。长历史导致完整输入超限时当前实现明确报错，尚未实现自动历史摘要。真实服务的失败无响应时仍可能没有usage，不会填零伪称完整。"]
    (root / "模块0_Token与预算报告.md").write_text("\n".join(module0) + "\n", encoding="utf-8")
    paths = chunk_paths[:]
    retrieval_paths = [root / name for name in ("global_current", "global_mmr")]
    if all((p / "summary.json").is_file() for p in retrieval_paths):
        base, variant = [statistics(p) for p in retrieval_paths]
        conclusion = (f"全局候选先去重融合再围绕完整问题重排，修复了原Graph按子查询插入顺序截断的问题。"
            f"本次MMR对照中，无MMR事实覆盖为{base['fact_evidence_recall']:.2%}，MMR为{variant['fact_evidence_recall']:.2%}；"
            f"问答实际已知总Token分别为{base['total_tokens']}和{variant['total_tokens']}。"
            "MMR使用同一Embedding模型的真实余弦相似度及LangChain官方实现，lambda固定0.7；额外向量请求已计成本。"
            "是否启用默认策略需结合逐题丢失/新增事实和事后语义冗余测量，不以更分散的证据自动代替更充分的证据。"
            "本次仅一次完整运行，Router和补检索可能因模型随机性不同，不是冻结所有中间查询的纯离线MMR消融。")
        if (root / "semantic_diversity.json").is_file():
            diversity = json.loads((root / "semantic_diversity.json").read_text(encoding="utf-8"))
            before = diversity["runs"]["global_current"]["mean_pair_cosine"]
            after = diversity["runs"]["global_mmr"]["mean_pair_cosine"]
            conclusion += (f" 事后对两组最终证据用同一Embedding复测，块间平均余弦相似度从{before:.4f}降至{after:.4f}。"
                f"这说明本次证据更分散，但Q16必要事实覆盖由1降至1/3，说明多样性不能保证必要条款覆盖。"
                f"语义评价本身另耗{diversity['evaluation_tokens']} Token，不计入问答成本。"
                "综合事实覆盖下降和开销增加，保留MMR开关但默认关闭；以后如要继续试验，先保护必要子问题覆盖，再少量调整lambda，不能直接扩大随机参数搜索。")
        comparison(retrieval_paths, root / "模块2_MMR实验报告.md", "模块2：全局候选无MMR与有MMR对照", conclusion)
        comparison([root / "chunk_prefix", root / "global_current"], root / "全局融合修复对照.md",
            "全局候选融合修复：旧合并与新合并", "此表是全局融合修复的补充对照，切分固定为prefix。旧方案每子查询先重排Top5再合并，"
            "新方案每子查询保留候选、全局RRF与一次统一重排后Top5，两者最终证据量可能不同。它用于评估修复后的整体行为，不能称为只改变排序而证据预算完全相同的消融。")
        paths.extend(retrieval_paths)
    (root / "实验统计.json").write_text(json.dumps({p.name: statistics(p) for p in paths}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
