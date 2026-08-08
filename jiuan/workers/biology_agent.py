"""Biology scenario Agent: annotated data -> KB -> 3 training iterations."""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

from .. import annotation, registry, store
from ..common import DATA, ROOT
from ..pipeline import breadth, dataprep, rag_backend
from ..schemas import Stage, TaskStatus
from . import runner

BIO_SYSTEM = (
    "你是专业生物学知识助手，回答要准确区分概念、机制、证据与应用；"
    "涉及实验判断时说明控制变量、可验证证据和常见误区。"
)

BIO_PROBE = "请从分子与细胞层面解释线粒体氧化磷酸化为什么需要内膜质子梯度。"

_lock = threading.RLock()
_thread: threading.Thread | None = None
_agent_task_id: str | None = None
_state: dict[str, Any] = {
    "active": False,
    "status": "idle",
    "message": "未启动",
    "logs": [],
    "summaries": [],
    "steps": [],
}


def status() -> dict:
    with _lock:
        if not _state.get("active") and not _state.get("task_id"):
            latest = _latest_persisted_state()
            if latest:
                return latest
        out = dict(_state)
        out["logs"] = list(_state.get("logs", []))
        out["summaries"] = list(_state.get("summaries", []))
        out["steps"] = list(_state.get("steps", []))
        return out


def _latest_persisted_state() -> dict | None:
    try:
        for task in store.list_tasks(Stage.AGENT):
            if task.params.get("mode") == "biology" and task.result:
                return task.result
    except Exception:
        return None
    return None


def _set(**kw: Any) -> None:
    with _lock:
        _state.update(kw)


def _log(message: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {message}"
    with _lock:
        logs = _state.setdefault("logs", [])
        logs.append(line)
        del logs[:-160]
        task_id = _agent_task_id
    if task_id:
        store.update_task(task_id, log=message)


def _progress(text: str) -> None:
    _set(progress=text, message=text)
    if _agent_task_id:
        store.update_task(_agent_task_id, progress=text)


def _add_step(name: str, status_text: str, summary: str, artifacts: dict | None = None) -> None:
    step = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "step": name,
        "status": status_text,
        "summary": summary,
        "artifacts": artifacts or {},
    }
    with _lock:
        _state.setdefault("steps", []).append(step)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _topic_rows() -> list[dict]:
    topics: list[tuple[str, str, str, str, str]] = [
        ("细胞生物学", "细胞膜流动镶嵌模型", "细胞膜由磷脂双层、膜蛋白、胆固醇和糖类共同构成，膜蛋白可在平面内侧向移动", "疏水相互作用维持双层结构，选择性通透性来自脂质环境与转运蛋白", "解释药物跨膜、受体定位和细胞识别时，不能把膜看成静止屏障"),
        ("细胞生物学", "线粒体氧化磷酸化", "线粒体内膜电子传递链把还原当量的能量转化为质子电化学梯度", "ATP 合酶利用质子回流驱动 ADP 磷酸化，氧是末端电子受体", "判断能量代谢异常时要同时看耗氧、膜电位和 ATP 生成"),
        ("细胞生物学", "粗面内质网蛋白合成", "带信号肽的新生多肽经 SRP 引导到粗面内质网并进入分泌途径", "共翻译转运、折叠伴侣和二硫键形成共同决定分泌蛋白质量", "分泌蛋白、膜蛋白缺陷常与折叠压力或转运信号异常有关"),
        ("细胞生物学", "高尔基体蛋白修饰", "高尔基体负责蛋白糖基化、剪切、分选和囊泡运输", "顺面到反面形成加工梯度，不同酶定位决定修饰顺序", "分析定位错误蛋白时要追踪 ER-Golgi-膜系统而非只看转录量"),
        ("细胞生物学", "溶酶体与自噬", "溶酶体含酸性水解酶，自噬把受损细胞器或蛋白聚集体送入溶酶体降解", "自噬体形成、融合和酸化是连续步骤，任何一步异常都会导致底物累积", "神经退行性疾病和饥饿适应常涉及自噬通量变化"),
        ("细胞生物学", "细胞周期检查点", "G1/S、G2/M 和纺锤体检查点确保 DNA 完整、复制完成和染色体正确连接", "Cyclin-CDK 活性受磷酸化、降解和抑制蛋白调控", "肿瘤细胞常因检查点失灵而积累突变"),
        ("细胞生物学", "有丝分裂纺锤体", "纺锤体微管捕获动粒并把姐妹染色单体分离到两个子细胞", "微管动态不稳定性和马达蛋白产生拉力，张力不足会激活检查点", "抗微管药物可阻断快速分裂细胞，但也会影响正常增殖组织"),
        ("细胞生物学", "细胞凋亡", "凋亡是受调控的程序性细胞死亡，通常不引发强烈炎症", "内源性线粒体通路和外源性死亡受体通路都可激活 caspase 级联", "发育塑形、免疫清除和肿瘤治疗都依赖凋亡调控"),
        ("细胞生物学", "干细胞分化", "干细胞兼具自我更新和向特定谱系分化的潜能", "转录因子网络、表观遗传状态和微环境信号共同限制分化方向", "评价干细胞实验要看功能性分化证据而不只看标志物表达"),
        ("细胞生物学", "细胞信号转导", "信号转导把外界配体、机械或代谢信号转化为细胞内响应", "受体激活后通过第二信使、激酶级联和转录调控放大并整合信号", "同一通路在不同细胞类型中可能产生不同输出"),
        ("分子遗传学", "DNA 半保留复制", "DNA 复制后每个子代双链都含一条亲代链和一条新合成链", "DNA 聚合酶只能 5' 到 3' 延伸，领先链连续、滞后链形成冈崎片段", "复制保真性来自碱基配对、校对活性和错配修复"),
        ("分子遗传学", "PCR 扩增", "PCR 通过变性、退火和延伸循环指数扩增目标 DNA 片段", "引物特异性、退火温度、模板质量和聚合酶保真性决定结果", "阴性对照和熔解曲线可帮助识别污染或非特异扩增"),
        ("分子遗传学", "转录调控", "转录调控决定基因在何时何地以何种强度表达", "启动子、增强子、转录因子、染色质开放状态共同影响 RNA 聚合酶招募", "解释表达变化时要区分转录调控和 mRNA 稳定性变化"),
        ("分子遗传学", "mRNA 剪接", "真核前体 mRNA 通过剪接去除内含子并连接外显子", "剪接体识别保守位点，可变剪接可产生多个蛋白异构体", "剪接位点突变可能不改变编码区却显著影响蛋白产物"),
        ("分子遗传学", "翻译起始", "翻译起始把核糖体、起始 tRNA 和 mRNA 正确组装在起始密码子处", "起始因子和 Kozak 序列影响起始效率，能量状态也会调节翻译", "蛋白表达量不一定与 mRNA 水平完全一致"),
        ("分子遗传学", "基因突变类型", "突变包括点突变、插入缺失、拷贝数变异和染色体结构变异", "是否改变蛋白功能取决于位置、读框、保守性和调控区域影响", "致病性判读需要结合群体频率、功能证据和遗传分离"),
        ("分子遗传学", "DNA 甲基化", "DNA 甲基化通常发生在 CpG 位点并与染色质沉默相关", "甲基转移酶和去甲基化过程改变基因可及性但不改变碱基序列", "表观遗传变化可受发育和环境影响，不能简单等同于永久突变"),
        ("分子遗传学", "CRISPR-Cas9 编辑", "CRISPR-Cas9 利用向导 RNA 引导核酸酶在目标位点造成双链断裂", "细胞通过 NHEJ 或 HDR 修复，分别倾向产生敲除或精确替换", "实验设计必须评估脱靶、递送效率和伦理边界"),
        ("分子遗传学", "连锁遗传与交换", "位于同一染色体上距离较近的基因倾向共同遗传", "减数分裂交换会打破连锁，重组率可反映遗传距离", "遗传图谱分析需要足够样本量和明确表型分类"),
        ("分子遗传学", "Hardy-Weinberg 平衡", "在无选择、突变、迁移、漂变且随机交配的大群体中等位基因频率保持稳定", "基因型频率可用 p²、2pq、q² 预测", "偏离平衡提示可能存在进化压力或抽样/分型问题"),
        ("生物化学", "酶动力学 Km 与 Vmax", "Km 反映达到半最大反应速度所需底物浓度，Vmax 反映酶饱和时最大速度", "米氏方程适用于简单稳态条件，抑制剂会改变参数表现", "比较酶活要控制温度、pH、底物浓度和酶量"),
        ("生物化学", "变构调节", "变构调节是效应分子结合非活性中心改变酶构象和活性", "协同性可让代谢通路对底物或产物浓度更敏感", "反馈抑制常出现在代谢通路末端产物调控中"),
        ("生物化学", "糖酵解", "糖酵解在细胞质中把葡萄糖分解为丙酮酸并产生 ATP 与 NADH", "己糖激酶、PFK-1 和丙酮酸激酶是重要调节点", "缺氧时丙酮酸可转为乳酸以再生 NAD+"),
        ("生物化学", "三羧酸循环", "三羧酸循环在线粒体基质中氧化乙酰辅酶 A 并产生 NADH、FADH2 和 GTP", "循环中间体也是氨基酸和脂质代谢交汇点", "通量受能量状态、底物供给和氧化磷酸化需求影响"),
        ("生物化学", "脂肪酸 β 氧化", "β 氧化在线粒体中逐轮切下乙酰辅酶 A 并生成还原当量", "长链脂肪酸进入线粒体需要肉碱穿梭", "禁食或耐力运动时脂肪酸氧化对供能更重要"),
        ("生物化学", "氨基酸脱氨", "氨基酸脱氨把氨基转移或释放，为碳骨架进入能量代谢做准备", "转氨酶和谷氨酸脱氢酶连接氮代谢与尿素循环", "高氨血症会影响神经系统，说明氮清除很关键"),
        ("生物化学", "蛋白质二级结构", "α 螺旋和 β 折叠由主链氢键稳定，是蛋白折叠的重要局部结构", "氨基酸性质、溶剂环境和分子伴侣影响最终构象", "突变可能通过破坏折叠稳定性而非活性位点直接致病"),
        ("生物化学", "ATP 能量偶联", "ATP 水解释放自由能，可与热力学不利反应偶联", "磷酸基团转移、构象变化和离子梯度都可承载能量转换", "不能把 ATP 简化成储能越多越好，细胞更关注能量通量"),
        ("生物化学", "NAD+/NADH 氧化还原", "NAD+ 接受电子和质子形成 NADH，是分解代谢中的关键电子载体", "NAD+/NADH 比值影响糖酵解、TCA 和乳酸生成方向", "代谢状态分析常需要同时看氧化还原平衡和底物水平"),
        ("生物化学", "蛋白磷酸化", "蛋白磷酸化通过激酶和磷酸酶可逆调节蛋白活性、定位或相互作用", "磷酸化级联能快速放大信号并形成反馈", "磷酸化检测要注意位点特异性和时间动态"),
        ("微生物学", "革兰氏染色", "革兰氏染色根据细胞壁结构把细菌大致分为阳性和阴性", "厚肽聚糖层保留结晶紫，外膜存在会影响药物通透和免疫识别", "染色结果受培养时间和操作影响，不能替代分子鉴定"),
        ("微生物学", "细菌生长曲线", "细菌批量培养经历延滞期、对数期、稳定期和衰亡期", "营养、代谢废物和空间限制决定群体增长速率", "抗生素敏感性实验通常关注对数生长期细胞"),
        ("微生物学", "质粒水平转移", "质粒可通过接合、转化或转导在细菌间传播", "耐药基因和代谢基因常借质粒快速扩散", "监测耐药传播要关注移动遗传元件而不只看物种"),
        ("微生物学", "噬菌体", "噬菌体是感染细菌的病毒，可进行裂解或溶原生命周期", "溶原转换可能赋予宿主毒力因子", "噬菌体治疗需要考虑宿主范围、免疫反应和耐受演化"),
        ("微生物学", "生物膜", "生物膜是微生物附着表面并被胞外基质包裹形成的群体结构", "基质限制药物进入并形成代谢异质性", "慢性感染和管道污染常因生物膜而难以清除"),
        ("微生物学", "抗生素耐药", "耐药可由药物靶点改变、药物外排、酶降解或通透性下降导致", "选择压力促进耐药菌株扩增，水平转移加速传播", "合理用药和药敏检测是控制耐药的核心"),
        ("微生物学", "灭菌与消毒", "灭菌要求杀灭所有微生物及芽孢，消毒主要降低病原体数量", "高压蒸汽灭菌、过滤、化学消毒适用对象不同", "选择方法要考虑材料耐受性、目标微生物和暴露时间"),
        ("微生物学", "病毒复制周期", "病毒复制包括吸附、进入、脱壳、基因组复制、装配和释放", "不同病毒依赖不同宿主酶和细胞器", "抗病毒药物常针对进入、聚合酶或蛋白酶等关键步骤"),
        ("微生物学", "人体微生物组", "微生物组是定植在人体不同部位的微生物群落及其基因集合", "菌群通过代谢、免疫教育和屏障竞争影响宿主健康", "相关性研究需要谨慎解释因果"),
        ("微生物学", "群体感应", "群体感应通过自诱导分子感知群体密度并协调基因表达", "当信号达到阈值，可诱导生物膜、毒力或发光等群体行为", "阻断群体感应是抗感染策略之一"),
        ("免疫学", "先天免疫与适应性免疫", "先天免疫快速识别模式分子，适应性免疫通过特异性受体产生记忆", "两者通过抗原呈递、细胞因子和共刺激信号相互连接", "免疫反应强弱取决于病原特征和宿主状态"),
        ("免疫学", "MHC I 与 MHC II 抗原呈递", "MHC I 主要呈递内源性抗原给 CD8 T 细胞，MHC II 呈递外源性抗原给 CD4 T 细胞", "抗原加工路径不同，决定了激活的 T 细胞类型", "疫苗设计需要考虑抗原进入哪条呈递通路"),
        ("免疫学", "TCR/BCR 克隆选择", "少数能识别抗原的淋巴细胞克隆被激活并扩增", "受体多样性来自 V(D)J 重排，耐受机制减少自身反应", "克隆扩增解释了免疫应答的特异性和记忆形成"),
        ("免疫学", "抗体类别转换", "B 细胞在辅助 T 细胞和细胞因子作用下改变抗体恒定区类别", "类别转换改变效应功能但不改变抗原特异性", "IgG、IgA、IgE 等类别对应不同组织环境和免疫任务"),
        ("免疫学", "补体系统", "补体通过经典、旁路和凝集素途径激活并促进裂解、调理和炎症", "级联反应需要严格调控以避免损伤自身组织", "补体缺陷会增加感染或自身免疫风险"),
        ("免疫学", "炎症细胞因子", "细胞因子协调免疫细胞募集、激活和分化", "TNF、IL-1、IL-6 等可诱导发热和急性期反应", "过度炎症可能造成组织损伤，抗炎治疗需平衡清除病原"),
        ("免疫学", "免疫记忆", "初次免疫后形成记忆 B/T 细胞，再次遇到抗原时反应更快更强", "亲和力成熟和长寿浆细胞提高抗体质量和持续性", "加强针的目的通常是提升记忆反应的广度和强度"),
        ("免疫学", "疫苗免疫原性", "疫苗通过安全暴露抗原诱导保护性免疫", "佐剂、递送方式和抗原构象影响免疫原性", "评价疫苗不能只看抗体滴度，还要看保护相关指标"),
        ("免疫学", "超敏反应", "超敏反应是免疫反应对机体造成损伤的病理状态", "I 到 IV 型分别涉及 IgE、抗体/补体、免疫复合物和 T 细胞机制", "治疗策略要依据机制选择抗组胺、免疫抑制或脱敏等方法"),
        ("免疫学", "免疫耐受", "免疫耐受防止免疫系统攻击自身抗原", "中枢耐受删除高亲和自身反应克隆，外周耐受依赖调节性 T 细胞等机制", "耐受破坏可导致自身免疫，过强耐受可能削弱抗肿瘤免疫"),
        ("生理学", "稳态负反馈", "稳态是机体维持内部环境相对稳定的能力", "传感器、整合中枢和效应器构成负反馈环路", "体温、血糖和血压调节都体现反馈控制"),
        ("生理学", "动作电位", "动作电位是可兴奋细胞膜电位快速去极化和复极化过程", "电压门控钠通道和钾通道的时序开放产生全或无信号", "髓鞘和轴突直径会影响传导速度"),
        ("生理学", "突触传递", "突触传递把神经元电信号转化为化学信号再影响下游细胞", "钙离子触发囊泡释放递质，受体类型决定兴奋或抑制效应", "药物可通过改变递质释放、降解或受体结合影响神经功能"),
        ("生理学", "内分泌激素", "激素由内分泌细胞释放并通过血液调控远端靶组织", "受体表达决定靶细胞响应，反馈轴维持分泌节律", "激素异常要同时分析分泌水平、受体敏感性和反馈调节"),
        ("生理学", "肾小球滤过", "肾小球滤过把血浆中小分子滤入肾小囊形成原尿", "滤过屏障由内皮、基底膜和足细胞裂孔膜组成", "蛋白尿提示滤过屏障受损但需结合肾小管重吸收判断"),
        ("生理学", "呼吸气体交换", "肺泡和毛细血管之间通过分压差交换氧气和二氧化碳", "通气、灌注和扩散距离共同决定交换效率", "低氧可来自通气不足、弥散障碍或 V/Q 失衡"),
        ("生理学", "心输出量", "心输出量等于每搏输出量乘以心率", "前负荷、后负荷、心肌收缩力和自主神经调节共同影响泵血", "评价循环功能要结合血压、灌注和氧输送"),
        ("生理学", "消化酶", "消化酶把大分子营养物分解为可吸收的小分子", "不同酶有特定 pH 和底物要求，胆汁促进脂肪乳化", "胰腺或肠黏膜功能异常会导致特定营养吸收障碍"),
        ("生理学", "肌肉收缩", "骨骼肌收缩依赖肌动蛋白和肌球蛋白滑动", "钙离子暴露结合位点，ATP 驱动横桥循环", "疲劳可由能量供应、离子平衡和神经驱动变化共同造成"),
        ("生理学", "昼夜节律", "昼夜节律由内源性生物钟与外界光暗周期同步", "视交叉上核、褪黑素和时钟基因参与调控", "轮班和光照紊乱会影响睡眠、代谢和免疫"),
        ("植物学", "光合作用光反应", "光反应在类囊体膜上把光能转化为 ATP 和 NADPH", "水裂解释放氧气，电子传递建立质子梯度", "光强、色素和电子传递效率影响光合速率"),
        ("植物学", "Calvin 循环", "Calvin 循环在叶绿体基质中固定二氧化碳并合成三碳糖", "Rubisco 催化羧化但也可能发生光呼吸", "C3、C4 和 CAM 植物对环境压力的适应不同"),
        ("植物学", "气孔调节", "气孔通过保卫细胞膨压变化调节气体交换和水分散失", "光、CO2、脱落酸和水分状态影响开闭", "干旱时关闭气孔能保水但会限制 CO2 进入"),
        ("植物学", "木质部与韧皮部运输", "木质部主要运输水和矿物质，韧皮部运输光合产物", "蒸腾拉力驱动木质部上升流，源库关系驱动韧皮部装载和卸载", "环割实验可用于区分两类运输通道"),
        ("植物学", "生长素", "生长素参与细胞伸长、顶端优势和向性生长", "极性运输造成浓度梯度，不同组织对浓度敏感性不同", "解释向光性时要关注生长素分布而非光直接拉动植物"),
        ("植物学", "光周期", "植物通过感知昼夜长度调控开花等发育过程", "光敏色素和生物钟共同解读夜长信息", "短日植物和长日植物的临界暗期不同"),
        ("植物学", "根瘤固氮", "豆科植物与根瘤菌共生，把大气氮转化为可利用氮化合物", "固氮酶对氧敏感，豆血红蛋白帮助维持低氧环境", "农业轮作可利用共生固氮减少氮肥依赖"),
        ("植物学", "植物防御", "植物通过结构屏障、次生代谢物和诱导免疫抵御病原或植食者", "模式识别受体和激素信号参与局部与系统性防御", "抗性育种要考虑病原变异和生长代价"),
        ("植物学", "世代交替", "植物生活史在二倍体孢子体和单倍体配子体之间转换", "减数分裂产生孢子，受精恢复二倍体", "不同植物类群中孢子体和配子体优势阶段不同"),
        ("植物学", "种子休眠", "种子休眠让萌发避开不利环境", "脱落酸、赤霉素、种皮限制和环境信号共同调控", "打破休眠可通过层积、光照或激素处理实现"),
        ("生态学", "食物网与营养级", "食物网描述生态系统中能量和物质通过捕食关系流动", "能量沿营养级传递时逐级损失，顶级消费者数量通常受限", "生态扰动可能通过级联效应影响多个营养级"),
        ("生态学", "环境容纳量", "环境容纳量是环境长期可支持的最大种群规模", "资源限制、竞争、捕食和疾病会限制增长", "种群超过容纳量可能导致资源枯竭和数量回落"),
        ("生态学", "生态位", "生态位包括物种利用资源、环境条件和生态功能的综合位置", "生态位重叠可能导致竞争，分化可促进共存", "保护物种要保护其功能环境而不只是个体数量"),
        ("生态学", "群落演替", "演替是群落组成随时间有方向变化的过程", "初生演替从无土壤环境开始，次生演替发生在扰动后的残留土壤上", "演替路径受物种扩散、干扰频率和环境过滤影响"),
        ("生态学", "生物多样性指数", "多样性同时包含物种丰富度和均匀度", "Shannon 或 Simpson 指数可量化群落结构但对稀有种敏感性不同", "监测多样性应统一采样强度和空间尺度"),
        ("生态学", "营养循环", "碳、氮、磷等元素在生物和非生物环境之间循环", "分解者和微生物转化决定许多元素的可利用性", "人类施肥和燃烧会改变全球生物地球化学循环"),
        ("生态学", "捕食者-猎物动态", "捕食和被捕食关系可导致种群数量周期性波动", "功能反应、繁殖延迟和替代猎物都会改变动态", "单一物种数量变化常需放到食物网中解释"),
        ("生态学", "共生关系", "共生包括互利、偏利和寄生等长期相互作用", "收益和成本可随环境条件变化而改变", "珊瑚与虫黄藻关系说明共生破裂会造成生态后果"),
        ("生态学", "入侵物种", "入侵物种在新环境中扩散并造成生态或经济影响", "天敌释放、高繁殖力和生态位空缺可促进入侵", "防控应优先早期监测和快速清除"),
        ("生态学", "生态系统服务", "生态系统服务包括供给、调节、文化和支持服务", "授粉、水源涵养、碳汇和土壤形成都是典型服务", "生态评估要把短期收益和长期系统稳定性一起考虑"),
        ("进化生物学", "自然选择", "自然选择使能提高适合度的遗传变异在群体中增加", "选择需要可遗传变异、适合度差异和繁殖传递", "适应不是个体主动进化，而是群体等位基因频率改变"),
        ("进化生物学", "遗传漂变", "遗传漂变是等位基因频率因随机抽样而变化", "小群体中漂变更强，可导致有益变异丢失或有害变异固定", "瓶颈效应和奠基者效应都是漂变的重要情形"),
        ("进化生物学", "基因流", "基因流是不同群体之间通过迁移和繁殖交换等位基因", "基因流可增加群体内变异并降低群体间分化", "保护遗传学中适度基因流可缓解近交但也需防止外源基因冲击"),
        ("进化生物学", "物种形成", "物种形成是生殖隔离逐渐建立并导致谱系分化的过程", "地理隔离、生态分化和性选择都可促进隔离", "判断物种边界需要结合形态、遗传和生殖证据"),
        ("进化生物学", "性选择", "性选择来自配偶竞争或择偶偏好导致的繁殖成功差异", "夸张性状可能提高繁殖机会但降低生存优势", "性二型和求偶行为常由性选择塑造"),
        ("进化生物学", "系统发育树", "系统发育树表示物种或基因之间的共同祖先关系", "分支顺序代表亲缘关系而非高低等级", "解读树时要区分节点、分支长度和外群设定"),
        ("进化生物学", "同源与同功", "同源结构源于共同祖先，同功结构功能相似但来源不同", "趋同进化可产生同功特征", "比较解剖和分子证据可帮助区分两者"),
        ("进化生物学", "适应辐射", "适应辐射是单一祖先谱系快速分化为多个生态型或物种", "生态机会、关键创新和地理隔离可促进辐射", "岛屿生物群常用于研究适应辐射"),
        ("进化生物学", "分子钟", "分子钟利用序列差异估计谱系分化时间", "突变率需校准，不同基因和谱系速率可能不同", "化石校准和置信区间对时间推断很重要"),
        ("进化生物学", "协同进化", "协同进化是相互作用物种之间相互施加选择压力并共同改变", "宿主-寄生物、植物-传粉者常出现协同适应", "军备竞赛和互利稳定都可能是协同进化结果"),
        ("实验与统计", "对照组设计", "对照组提供判断处理效应的基线", "阴性对照、阳性对照和空白对照回答不同质量问题", "缺少合适对照会让结果无法排除替代解释"),
        ("实验与统计", "随机化", "随机化把未知混杂因素平均分配到各组", "随机分组和随机采样解决的问题不同", "随机化不能替代足够样本量和盲法"),
        ("实验与统计", "生物学重复", "生物学重复来自独立样本，技术重复来自同一样本重复测量", "统计推断主要依赖生物学重复估计自然变异", "只增加技术重复不能弥补样本个体不足"),
        ("实验与统计", "qPCR Ct 值", "Ct 值是荧光信号超过阈值所需循环数，越低通常表示初始模板越多", "相对定量需内参基因和扩增效率校正", "比较表达量应报告 ΔΔCt 或等效方法而非只列 Ct"),
        ("实验与统计", "Western blot", "Western blot 用抗体检测特定蛋白大小和丰度", "样品上样量、转膜效率、抗体特异性和内参影响可信度", "条带强度需要在线性范围内定量"),
        ("实验与统计", "显微镜分辨率", "分辨率是区分两个相近点的能力，受波长和数值孔径限制", "放大倍数不能弥补分辨率不足", "共聚焦和超分辨技术适合不同尺度问题"),
        ("实验与统计", "ELISA", "ELISA 通过抗原抗体特异结合和酶促显色定量目标分子", "标准曲线、洗涤充分性和交叉反应决定准确度", "结果应落在标准曲线线性范围内"),
        ("实验与统计", "序列比对", "序列比对用于识别同源区域、保守位点和潜在功能域", "参数、缺口罚分和数据库选择会影响结果", "高相似性提示但不等同于功能完全相同"),
        ("实验与统计", "p 值与置信区间", "p 值表示在零假设下观察到当前或更极端数据的概率", "置信区间提供效应大小估计的不确定性范围", "统计显著不等于生物学意义重大"),
        ("实验与统计", "样本污染控制", "污染会把外源 DNA、蛋白或微生物信号误判为真实结果", "空间分区、阴性对照、无模板对照和独立复现实验可识别污染", "高灵敏方法尤其需要严格污染控制"),
        ("生物安全与伦理", "BSL 实验室等级", "BSL-1 到 BSL-4 根据病原风险和防护要求递增", "等级越高，对设施、个人防护、空气流向和废弃物处理要求越严格", "选择等级应基于病原危害、传播途径和实验操作风险"),
        ("生物安全与伦理", "转基因生物风险评估", "转基因风险评估关注基因流、生态影响、食品安全和管理可追溯性", "风险取决于性状、受体物种和释放环境", "不能只因使用转基因技术就判定高风险或无风险"),
        ("生物安全与伦理", "病原样本处理", "病原样本处理需防止暴露、泄漏和交叉污染", "个人防护、密闭离心、消毒和台账管理是基本要求", "不明风险样本应按更保守等级处理"),
        ("生物安全与伦理", "高压蒸汽灭菌", "高压蒸汽灭菌利用高温湿热使蛋白变性并杀灭微生物", "常用条件如 121 摄氏度、一定压力和足够时间，但需验证穿透效果", "装载过满或空气未排尽会导致灭菌失败"),
        ("生物安全与伦理", "锐器废弃物", "针头、刀片等锐器应进入防刺穿专用容器", "禁止回套针帽和徒手分拣可减少刺伤", "锐器伤需立即处理并按暴露流程上报"),
        ("生物安全与伦理", "人体研究伦理", "人体研究必须尊重知情同意、风险最小化、隐私保护和公平招募", "伦理审查评估科学价值与受试者风险收益", "数据再利用也需符合授权范围和隐私要求"),
        ("生物安全与伦理", "动物福利 3R", "3R 原则包括替代、减少和优化动物实验", "研究设计应减少动物数量并降低痛苦，同时保证科学有效性", "伦理合规不是形式审查，而是实验质量的一部分"),
        ("生物安全与伦理", "生物多样性保护", "保护生物多样性需要维护基因、物种和生态系统多层次多样性", "栖息地破碎化、过度利用、污染和气候变化是主要压力", "保护策略应结合就地保护、迁地保护和社区治理"),
        ("生物安全与伦理", "抗生素管理", "抗生素管理通过合理使用减少耐药选择压力", "适应证、剂量、疗程和窄谱优先原则都很重要", "农业和医疗场景都需要监测耐药传播"),
        ("生物安全与伦理", "双重用途研究", "双重用途研究可能同时带来科学收益和被滥用风险", "病原增强、传播性改变和合成生物学研究需特别评估", "治理应平衡开放科学、风险审查和责任沟通"),
    ]
    rows = []
    for idx, (category, topic, core, mechanism, application) in enumerate(topics, 1):
        rows.append({
            "id": f"bio-{idx:03d}",
            "category": category,
            "system_prompt": BIO_SYSTEM,
            "instruction": f"从专业生物学角度解释：{topic}。要求说明定义、关键机制和一个应用或判断要点。",
            "input": "",
            "output": (
                f"{topic}：{core}。关键机制：{mechanism}。应用或判断要点：{application}。"
                "回答时应把概念、机制、证据和应用分开，避免把相关性直接当作因果结论。"
            ),
        })
    return rows


def _eval_rows(train_rows: list[dict]) -> list[dict]:
    by_cat: dict[str, list[dict]] = {}
    for row in train_rows:
        by_cat.setdefault(row["category"], []).append(row)
    rows = []
    for category in sorted(by_cat):
        for src in by_cat[category][:3]:
            rows.append({
                "id": f"eval-{src['id']}",
                "category": category,
                "system_prompt": BIO_SYSTEM,
                "instruction": f"广度验证题：如果学生追问「{src['instruction'].split('：', 1)[-1].split('。', 1)[0]}」在真实研究或应用中的意义，应如何专业回答？",
                "input": "",
                "output": src["output"],
            })
    return rows


def _materials(source_name: str, run_id: str) -> dict:
    bio_dir = DATA / "biology"
    bio_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _topic_rows()
    eval_rows = _eval_rows(train_rows)
    source_path = bio_dir / f"{source_name}.jsonl"
    eval_path = bio_dir / f"{source_name}_广度验证.jsonl"
    kb_path = bio_dir / f"{source_name}库.md"
    _write_jsonl(source_path, train_rows)
    _write_jsonl(eval_path, eval_rows)
    categories: dict[str, list[dict]] = {}
    for row in train_rows:
        categories.setdefault(row["category"], []).append(row)
    lines = [
        f"# {source_name}库",
        "",
        "本知识库配套生物专家训练场景，覆盖细胞、遗传、生化、微生物、免疫、生理、植物、生态、进化、实验统计、生物安全与伦理。",
        "",
    ]
    for category, rows in sorted(categories.items()):
        lines.append(f"## {category}")
        for row in rows:
            topic = row["instruction"].split("：", 1)[-1].split("。", 1)[0]
            lines.append(f"- {topic}：{row['output']}")
        lines.append("")
    kb_path.write_text("\n".join(lines), encoding="utf-8")
    return {
        "run_id": run_id,
        "source_path": source_path,
        "eval_path": eval_path,
        "kb_path": kb_path,
        "train_rows": train_rows,
        "eval_rows": eval_rows,
    }


def _rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def _split_rows(rows: list[dict], parts: int) -> list[list[dict]]:
    base = len(rows) // parts
    rem = len(rows) % parts
    out = []
    start = 0
    for i in range(parts):
        size = base + (1 if i < rem else 0)
        out.append(rows[start:start + size])
        start += size
    return out


def _create_annotation_task(rows: list[dict], task_id: str, iteration_name: str) -> dict:
    ann_dir = DATA / "annotations"
    ann_dir.mkdir(parents=True, exist_ok=True)
    path = ann_dir / f"{task_id}.jsonl"
    ann_rows = []
    for row in rows:
        prompt = row["instruction"].strip()
        if row.get("input"):
            prompt += "\n" + row["input"].strip()
        ann_rows.append({
            "prompt": prompt,
            "reference": row["output"],
            "annotation": row["output"],
            "status": "annotated",
            "gap_type": "生物知识扩增/专业标注",
            "iteration": iteration_name,
            "category": row["category"],
            "source": "生物知识",
            "system_prompt": BIO_SYSTEM,
        })
    _write_jsonl(path, ann_rows)
    return {"task_id": task_id, "path": path, "count": len(ann_rows)}


def _wait_task(task_id: str, label: str, poll_interval: float) -> dict:
    last = ""
    while True:
        task = store.get_task(task_id)
        if not task:
            raise RuntimeError(f"{label} 子任务不存在: {task_id}")
        msg = f"{label}: {task.status.value}" + (f" {task.progress}" if task.progress else "")
        _progress(msg)
        if msg != last:
            _log(f"{label} 子任务 {task_id} -> {task.status.value}" + (f" ({task.progress})" if task.progress else ""))
            last = msg
        if task.status == TaskStatus.SUCCEEDED:
            return task.result
        if task.status == TaskStatus.FAILED:
            raise RuntimeError(f"{label} 子任务失败 {task_id}: {task.error[:800]}")
        time.sleep(poll_interval)


def _save_steps(run_id: str) -> dict:
    step_dir = DATA / "agent_steps"
    step_dir.mkdir(parents=True, exist_ok=True)
    snap = status()
    json_path = step_dir / f"biology-{run_id}.json"
    md_path = step_dir / f"biology-{run_id}.md"
    json_path.write_text(json.dumps(snap, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        f"# 生物专家 Agent 自动步骤 {run_id}",
        "",
        f"- 状态：{snap.get('status')}",
        f"- 消息：{snap.get('message')}",
        f"- 最终模型：{snap.get('final_model_id', '')}",
        f"- 最终数据集：{snap.get('final_dataset_id', '')}",
        f"- 知识库 collection：{snap.get('kb_collection', '')}",
        "",
        "## 步骤",
    ]
    for i, step in enumerate(snap.get("steps", []), 1):
        lines.append(f"{i}. {step['step']} [{step['status']}]：{step['summary']}")
        for k, v in (step.get("artifacts") or {}).items():
            lines.append(f"   - {k}: {v}")
    lines.append("")
    lines.append("## 三轮摘要")
    for item in snap.get("summaries", []):
        metrics = item.get("metrics") or {}
        breadth_report = (item.get("breadth") or {}).get("report_id", "")
        lines.append(
            f"- iter {item.get('iteration')}: dataset={item.get('dataset_id')}, "
            f"model={item.get('model_id')}, rouge={metrics.get('rouge_l_f')}, "
            f"judge={metrics.get('judge_avg')}, breadth={breadth_report}"
        )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    _set(step_summary_json=_rel(json_path), step_summary_md=_rel(md_path))
    return {"json": _rel(json_path), "markdown": _rel(md_path)}


def start(params: dict) -> dict:
    global _thread, _agent_task_id
    with _lock:
        if _thread and _thread.is_alive():
            return status()
        task = store.create_task(Stage.AGENT, {"mode": "biology", **params})
        _agent_task_id = task.id
        _state.clear()
        _state.update({
            "active": True,
            "status": "running",
            "message": "生物专家三轮 Agent 启动中",
            "progress": "starting",
            "task_id": task.id,
            "params": params,
            "started_at": time.time(),
            "logs": [],
            "summaries": [],
            "steps": [],
            "kb_collection": "biology",
        })
        store.update_task(task.id, status=TaskStatus.RUNNING, log="生物专家 Agent 自动流程启动")
        _thread = threading.Thread(target=_run, args=(task.id, params), name="jiuan-biology-agent", daemon=True)
        _thread.start()
        return status()


def _run(task_id: str, params: dict) -> None:
    run_id = time.strftime("%Y%m%d-%H%M%S")
    try:
        poll = float(params.get("poll_interval") or 5.0)
        max_iterations = int(params.get("max_iterations") or 3)
        source_name = params.get("source_name") or "生物知识"
        prefix = params.get("iteration_prefix") or "bio-v"
        train_backend = params.get("train_backend") or "hf"
        train_device = params.get("train_device") or "cpu"
        train_epochs = int(params.get("train_epochs") or 1)
        train_max_seq_len = int(params.get("train_max_seq_len") or 512)
        eval_max_samples = int(params.get("eval_max_samples") or 11)
        breadth_max_samples = int(params.get("breadth_max_samples") or 11)
        use_judge = params.get("use_judge") or "auto"

        _progress("生成生物场景材料")
        mat = _materials(source_name, run_id)
        _set(source=_rel(mat["source_path"]), breadth_source=_rel(mat["eval_path"]))
        _add_step(
            "生成场景材料",
            "succeeded",
            f"生成 {len(mat['train_rows'])} 条标注样本、{len(mat['eval_rows'])} 条广度验证样本和配套知识库",
            {"source": _rel(mat["source_path"]), "breadth_source": _rel(mat["eval_path"]), "kb": _rel(mat["kb_path"])},
        )
        _log(f"标注源 {source_name}: {len(mat['train_rows'])} 条")

        _progress("生成并标记 held-out 广度验证集")
        eval_dp = dataprep.run(
            {"source": _rel(mat["eval_path"]), "name": f"{prefix}广度验证", "valid_ratio": 0.9, "seed": 20260716},
            _log,
        )
        registry.mark_eval_dataset(eval_dp["dataset_id"])
        _set(eval_dataset_id=eval_dp["dataset_id"])
        _add_step(
            "生成固定广度验证集",
            "succeeded",
            f"创建 held-out 验证集 {eval_dp['dataset_id']}，并标记 role=eval 防止回灌泄漏",
            {"dataset_id": eval_dp["dataset_id"], "valid_count": eval_dp["valid_count"]},
        )

        _progress("写入 biology 知识库 collection")
        kb = rag_backend.ingest_document(_rel(mat["kb_path"]), chunk_size=700, collection="biology", log=_log)
        _add_step(
            "补全并切换知识库",
            "succeeded",
            f"知识库 collection=biology 入库 {kb['new_chunks']} 块，总块数 {kb['total_chunks']}",
            {"collection": "biology", "kb_source": kb["source"], "total_chunks": kb["total_chunks"]},
        )

        parent_dataset = None
        chunks = _split_rows(mat["train_rows"], max_iterations)
        for idx, rows in enumerate(chunks, 1):
            iteration_name = f"{prefix}{idx}"
            _set(iteration=idx)
            _progress(f"{iteration_name}: 自动生成预标注任务")
            ann_task_id = f"{iteration_name}-生物标注-{run_id}"
            ann_task = _create_annotation_task(rows, ann_task_id, iteration_name)
            _add_step(
                f"{iteration_name} 标注任务",
                "succeeded",
                f"从 {source_name} 自动抽取并预填 {ann_task['count']} 条标注，等待 Agent 回灌",
                {"annotation_task": ann_task_id, "file": _rel(ann_task["path"])},
            )

            _progress(f"{iteration_name}: 标注回灌生成训练数据集")
            dataset_name = f"{iteration_name}-生物学知识"
            ds = annotation.commit(ann_task_id, dataset_name, parent_dataset, hard_weight=1, log=_log)
            parent_dataset = ds["dataset_id"]
            _set(parent_dataset=parent_dataset, final_dataset_id=parent_dataset)
            _add_step(
                f"{iteration_name} 标注回灌",
                "succeeded",
                f"回灌 {ds['annotated']} 条标注，生成训练数据集 {parent_dataset}，累计 {ds['dataset_count']} 条",
                {"dataset_id": parent_dataset, "parent_dataset": ds.get("parent_dataset"), "added_count": ds.get("added_count")},
            )

            _progress(f"{iteration_name}: 提交训练")
            train_params = {
                "dataset_id": parent_dataset,
                "name": f"{iteration_name}-生物专家",
                "backend": train_backend,
                "method": "lora",
                "device": train_device,
                "epochs": train_epochs,
                "max_seq_len": train_max_seq_len,
                "lora_r": 4,
                "lora_alpha": 8,
                "lora_dropout": 0.05,
            }
            train_task_id = runner.submit(Stage.TRAIN, train_params)
            model = _wait_task(train_task_id, f"{iteration_name} 训练", poll)
            model_id = model["model_id"]
            _set(last_model_id=model_id, final_model_id=model_id)
            _add_step(
                f"{iteration_name} 模型训练",
                "succeeded",
                f"训练完成，模型 {model_id}，训练后端 {model.get('backend')}",
                {"train_task_id": train_task_id, "model_id": model_id, "train_loss": model.get("train_loss")},
            )

            _progress(f"{iteration_name}: 推理抽检")
            infer_task_id = runner.submit(
                Stage.INFER,
                {
                    "model_id": model_id,
                    "prompt": BIO_PROBE,
                    "backend": None,
                    "system_prompt": BIO_SYSTEM,
                    "max_new_tokens": 160,
                    "do_sample": False,
                },
            )
            infer_result = _wait_task(infer_task_id, f"{iteration_name} 推理", poll)
            _add_step(
                f"{iteration_name} 推理抽检",
                "succeeded",
                f"完成一次生物学问题推理，回答长度 {len(infer_result.get('answer', ''))} 字",
                {"infer_task_id": infer_task_id, "prompt": BIO_PROBE, "answer_preview": (infer_result.get("answer") or "")[:160]},
            )

            _progress(f"{iteration_name}: 固定验证集评测")
            eval_task_id = runner.submit(
                Stage.EVAL,
                {
                    "model_id": model_id,
                    "dataset_id": eval_dp["dataset_id"],
                    "split": "valid",
                    "use_judge": use_judge,
                    "max_samples": eval_max_samples,
                    "system_prompt": BIO_SYSTEM,
                },
            )
            ev = _wait_task(eval_task_id, f"{iteration_name} 评测", poll)
            _add_step(
                f"{iteration_name} 固定评测",
                "succeeded",
                f"评测完成，ROUGE-L={ev.get('metrics', {}).get('rouge_l_f')}，gap={ev.get('gap_count')}",
                {"eval_task_id": eval_task_id, "report_id": ev.get("report_id"), "metrics": ev.get("metrics", {})},
            )

            _progress(f"{iteration_name}: 判断模型广度分析")
            br = breadth.run(
                {
                    "model_id": model_id,
                    "source": _rel(mat["eval_path"]),
                    "max_samples": breadth_max_samples,
                    "use_judge": use_judge,
                    "system_prompt": BIO_SYSTEM,
                    "backend": None,
                },
                _log,
                progress=lambda p: _progress(f"{iteration_name} 广度分析 {p}"),
            )
            _add_step(
                f"{iteration_name} 广度分析",
                "succeeded",
                f"按类别验证 {br['samples']} 条，薄弱类别 {len(br.get('issues', []))} 个",
                {"breadth_report_id": br["report_id"], "issues": [x["category"] for x in br.get("issues", [])]},
            )

            summary = {
                "iteration": idx,
                "annotation_task": ann_task_id,
                "dataset_id": parent_dataset,
                "model_id": model_id,
                "metrics": ev.get("metrics", {}),
                "gap_count": ev.get("gap_count", len(ev.get("gaps") or [])),
                "train_task_id": train_task_id,
                "infer_task_id": infer_task_id,
                "eval_task_id": eval_task_id,
                "breadth": {
                    "report_id": br.get("report_id"),
                    "issues": br.get("issues", []),
                    "category_scores": br.get("category_scores", []),
                },
            }
            with _lock:
                _state.setdefault("summaries", []).append(summary)
            _save_steps(run_id)

        _finish("succeeded", f"生物专家三轮迭代完成，最终模型 {status().get('final_model_id')}", run_id)
    except Exception as exc:  # noqa: BLE001
        _add_step("异常终止", "failed", str(exc))
        _set(active=False, status="failed", message=str(exc), finished_at=time.time(), progress="done")
        paths = _save_steps(run_id)
        store.update_task(task_id, status=TaskStatus.FAILED, error=str(exc), result=status(), log=f"生物专家 Agent 失败: {exc}")


def _finish(status_text: str, message: str, run_id: str | None = None) -> None:
    _set(active=False, status=status_text, message=message, progress="done", finished_at=time.time())
    if run_id:
        paths = _save_steps(run_id)
        _set(step_summary_json=paths.get("json"), step_summary_md=paths.get("markdown"))
    _log(message)
    if _agent_task_id:
        store.update_task(_agent_task_id, status=TaskStatus.SUCCEEDED, progress="done", result=status(), log="生物专家 Agent 自动流程结束")
