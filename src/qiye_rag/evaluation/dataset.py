"""人工设计问题与证据锚点；Gold 从真实切块中定位，绝不从检索结果反推。"""
import json
import hashlib
from pathlib import Path
from qiye_rag.rag.loader import load_directory
from qiye_rag.rag.splitter import split_documents
from qiye_rag import setting

KB = "eval_v5"
# 每个锚点来自原文；一题可以要求多个文档/事实同时成立。
SEEDS = [
    ("事实查询", "员工试用期一般多久？", [("HR001", "试用期一般为3个月")]),
    ("事实查询", "每月最多申请几个远程办公日？", [("HR002", "每月最多申请4个远程")]),
    ("事实查询", "差旅结束后多少天内提交票据？", [("FIN001", "15个自然日内提交")]),
    ("事实查询", "申请晋升至少需要在当前职级满多久？", [("PR001", "满2个季度")]),
    ("事实查询", "客户逾期35天应如何处理新订单发货？", [("FIN007", "超过30天暂停新订单发货并上报")]),
    ("关键词精确匹配", "BT-210的规格与尺寸是什么？", [("PRD002", "额定5W；蓝牙连接；USB-C充电")]),
    ("关键词精确匹配", "SS-330的材质是什么？", [("PRD003", "304不锈钢内胆")]),
    ("关键词精确匹配", "TX-620的规格与尺寸是什么？", [("PRD006", "38×42cm；提带70cm")]),
    ("关键词精确匹配", "OF-420的参考生产交期是多少？", [("PRD004", "标准工期22工作日")]),
    ("语义改写", "新人通常要考察几个月才能转为正式员工？", [("HR001", "试用期一般为3个月")]),
    ("语义改写", "没收到客户的钱，能先算销售奖金吗？", [("FIN004", "只有客户实际回款并完成财务对账后才计入提成基数")]),
    ("语义改写", "老板口头同意我延长考察期就够了吗？", [("HR001", "试用期延期不得由直属领导口头决定")]),
    ("语义改写", "合作方邮件说换收款账户了，照着改就行吗？", [("FIN002", "电话回拨与双人复核")]),
    ("多跳推理", "30万美元合同按组织矩阵和收付款制度应由谁审批？", [("HR007", "经营委员会集体审议"), ("FIN002", ">20万美元经营委员会审批")]),
    ("多跳推理", "低于15%毛利的报价需谁审批，毛利率如何计算？", [("TR003", "低于目标须财务审批"), ("FIN006", "报价毛利=(销售收入-产品采购成本-直接物流及检验成本)/销售收入")]),
    ("多跳推理", "BT-210涉及哪些电池风险，出口单证和市场准入应检查什么？", [("PRD002", "含锂电池"), ("TR007", "商业发票、装箱单、运输单据、合同/PI"), ("TR010", "目的国进口限制、产品适用认证、最终用途和贸易制裁风险")]),
    ("多跳推理", "销售季度绩效和晋升资格分别有什么要求？", [("HR005", "有效回款35%"), ("PR001", "最近两个季度绩效至少B")]),
    ("版本冲突", "2026年5月工龄2年的员工内部福利年假几天？", [("HIST001", "满1年未满3年为3天")]),
    ("版本冲突", "2026年10月工龄2年的员工内部福利年假几天？", [("HR003", "工龄满1年未满3年每年5天")]),
    ("版本冲突", "2026年7月20日上海住宿600元是否超出当时标准？", [("HIST002", "上海住宿600元超出旧版500元，需要例外审批")]),
    ("版本冲突", "2026年8月25日签发的普通产品报价有效期多久？", [("HIST003", "一般报价有效期30个自然日")]),
    ("权限隔离", "周芷晴负责哪个市场？", [("EMP002", "负责欧洲渠道与重点客户分配")]),
    ("权限隔离", "周芷晴负责哪个市场？", []),
    ("权限隔离", "沈博宇离职后，未完结出货交给谁？", [("EMP011", "未完结出货交接至赵明哲")]),
    ("权限隔离", "沈博宇离职后，未完结出货交给谁？", []),
    ("无法回答", "公司2027年的住宿报销额度是多少？", []),
    ("无法回答", "给我周芷晴的真实身份证号码。", []),
    ("无法回答", "BT-999的额定功率是多少？", []),
    ("多轮指代", "那它的标准生产交期呢？", [("PRD002", "标准工期25工作日")]),
    ("多轮指代", "那超过额度需要谁签字？", [("FIN001", "部门经理和财务经理双签")]),
]


def corpus(root: Path) -> tuple[list[dict], str]:
    """解析真实资料并生成内容身份和快照摘要。

    Args:
        root: 数据目录。
    Returns:
        真实 Chunk 列表及内容摘要。
    """
    docs, errors = load_directory(root, KB)
    if errors:
        raise ValueError(f"评测语料解析失败：{errors}")
    parts = [{"id": d.metadata["chunk_id"], "text": d.page_content, "metadata": d.metadata}
             for d in split_documents(docs)]
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True).encode()
    return parts, hashlib.sha256(payload).hexdigest()


def build(root: Path, target: Path) -> dict:
    """将人工事实锚点绑定到实际 Chunk 并写入评测集。

    Args:
        root: 企业资料目录。
        target: 输出 JSON 文件。
    Returns:
        含快照和三十道题的数据集。
    """
    parts, digest = corpus(root)
    cases = []
    for index, (category, question, anchors) in enumerate(SEEDS, 1):
        citations = []
        for document, fact in anchors:
            matches = [p for p in parts if p["metadata"]["document_id"] == document
                       and "".join(fact.split()) in "".join(p["text"].split())]
            if not matches:
                raise ValueError(f"原文锚点不存在：{document} {fact}")
            part = matches[0]
            citations.append({"document_id": document, "chunk_id": part["id"],
                              "source_path": part["metadata"]["source_path"],
                              "location": part["metadata"]["location"], "fact": fact})
        history = []
        if index == 29:
            history = [{"question": "BT-210是什么产品？", "answer": "BT-210是蓝牙音箱。"}]
        if index == 30:
            history = [{"question": "当前差旅住宿额度是多少？", "answer": "一线城市650元，其他城市450元。"}]
        cases.append({"id": f"Q{index:02}", "category": category, "question": question,
                      "role": "hr" if index in (22, 24) else "user", "history": history,
                      "answerable": bool(anchors), "gold_document_ids": sorted({d for d, _ in anchors}),
                      "gold_chunk_ids": sorted({c["chunk_id"] for c in citations}),
                      "reference_facts": [f for _, f in anchors], "expected_citations": citations,
                      "forbidden_document_ids": ["EMP002"] if index == 23 else ["EMP011"] if index == 25 else [],
                      "reference_answer": "；".join(f for _, f in anchors) if anchors else "当前授权知识库证据不足，应拒答。"})
    dataset = {"schema_version": 1, "kb_id": KB, "corpus_sha256": digest,
               "chunk_id_kind": "content_id（非带索引版本的存储主键）", "chunk_size": setting.CHUNK_SIZE,
               "chunk_overlap": setting.CHUNK_OVERLAP, "cases": cases}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(dataset, ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset
