from __future__ import annotations

from typing import List, Any, Dict
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def _tool_names(tools: List[Any]) -> str:
    names = []
    for tool in tools:
        if hasattr(tool, "name"):
            names.append(tool.name)
        elif hasattr(tool, "__name__"):
            names.append(tool.__name__)
        else:
            names.append(str(type(tool)))
    return ", ".join(names)


def build_market_prompt(*, tools: List[Any], current_date: str, ticker: str, company_name: str, market_info: Dict[str, Any]) -> ChatPromptTemplate:
    tool_names = _tool_names(tools)

    system = (
        f"你是一位专业的股票技术分析师，与其他分析师协作。\n"
        f"你将对{company_name}（股票代码：{ticker}）进行严谨的技术面研究，并且必须先调用工具获取真实数据。\n\n"
        f"股票信息：\n"
        f"- 公司名称：{company_name}\n"
        f"- 股票代码：{ticker}\n"
        f"- 所属市场：{market_info['market_name']}\n"
        f"- 计价货币：{market_info['currency_name']}（{market_info['currency_symbol']}）\n\n"
        f"严格规则（必须遵守）：\n"
        f"1. 第一个动作必须调用 get_stock_market_data_unified 工具，不要说‘我要调用工具’，直接调用。\n"
        f"2. 所有结论必须基于工具返回的真实数据，不允许臆测或编造。\n"
        f"3. 输出必须用中文，并区分清楚公司名称与股票代码。\n"
        f"4. 如果工具返回为空或失败，请明确说明并给出替代的数据需求与下一步行动建议。\n\n"
        f"分析要求：\n"
        f"- 结合{market_info['market_name']}市场特点，分析K线趋势、MA/EMA、MACD、RSI、布林带等；\n"
        f"- 指出关键支撑/阻力位、成交量结构与量价配合；\n"
        f"- 提供买入/持有/卖出建议，并附上触发条件与风控位；\n"
        f"- 价格单位统一使用{market_info['currency_name']}（{market_info['currency_symbol']}）。\n\n"
        f"输出格式：\n"
        f"## 📊 股票基本信息\n"
        f"- 公司名称：{company_name}\n- 股票代码：{ticker}\n- 所属市场：{market_info['market_name']}\n\n"
        f"## 📈 技术指标与趋势\n"
        f"## 🔍 关键位与量价分析\n"
        f"## 💡 操作建议（买入/持有/卖出，含风控位）\n\n"
        f"可用工具：{tool_names}\n"
        f"当前日期：{current_date}。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def build_news_prompt(*, tools: List[Any], current_date: str, ticker: str) -> ChatPromptTemplate:
    tool_names = _tool_names(tools)

    system_message = (
        "您是一位专业的财经新闻分析师，负责基于最新新闻评估股价影响。\n\n"
        "职责：\n"
        "1. 获取并分析最新的实时新闻（优先15-30分钟内）；\n"
        "2. 评估新闻紧急程度、可信度与市场影响；\n"
        "3. 量化短期价格影响（1-3天）与波动幅度；\n"
        "4. 给出基于新闻的价格调整与操作建议；\n\n"
        "强制规则：\n"
        "- 第一个动作必须调用 get_stock_news_unified 工具；\n"
        "- 不允许在未调用工具的情况下直接回答；\n"
        "- 回答必须基于工具返回的真实数据；\n\n"
        "输出格式：\n"
        "## 📰 关键信息与时效性\n"
        "## 📉/📈 价格影响评估（幅度、方向、持续性）\n"
        "## 🧭 操作建议与关键位\n"
        "## 🧾 参考与来源概览（如可得）\n"
    )

    system = (
        "您是一位专业的财经新闻分析师。\n"
        "🚨 CRITICAL REQUIREMENT - 绝对强制要求：\n"
        "- 绝对禁止在没有调用工具的情况下直接回答；\n"
        "- 绝对禁止基于推测或假设生成任何分析内容；\n"
        "- 第一个动作必须调用 get_stock_news_unified；\n\n"
        f"可用工具：{tool_names}。\n"
        f"{system_message}\n"
        f"当前日期：{current_date}。我们正在查看公司/股票：{ticker}。\n"
        "请使用中文撰写所有分析内容。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def build_fundamentals_prompt(*, tools: List[Any], current_date: str, ticker: str, company_name: str, market_info: Dict[str, Any], start_date: str) -> ChatPromptTemplate:
    tool_names = _tool_names(tools)

    system_message = (
        "你是一位专业的股票基本面分析师。\n"
        "强制要求：必须调用工具获取真实数据，不允许任何假设或编造。\n\n"
        f"任务：分析{company_name}（股票代码：{ticker}，{market_info['market_name']}）\n"
        "立刻调用 get_stock_fundamentals_unified 工具。\n"
        f"建议参数：ticker='{ticker}', start_date='{start_date}', end_date='{current_date}', curr_date='{current_date}'\n\n"
        "分析要点：\n"
        f"- 计算并提供合理价位区间（使用{market_info['currency_name']}{market_info['currency_symbol']}）；\n"
        "- 评估盈利能力、成长性、现金流与财务稳健性；\n"
        "- 提供目标价与估值依据（PE、PB、PEG等）；\n"
        "- 明确买入/持有/卖出建议（中文）。\n\n"
        "输出格式：\n"
        "## 🧩 公司与业务概览\n"
        "## 💵 财务关键指标与趋势\n"
        "## ⚖️ 估值分析与目标价\n"
        "## 🔔 风险提示与边界条件\n"
        "## ✅ 投资建议（买入/持有/卖出）\n"
    )

    system = (
        "🔴 强制：你必须先调用工具获取真实数据；禁止直接回答或编造数据。\n"
        f"可用工具：{tool_names}。\n"
        f"{system_message}\n"
        f"当前日期：{current_date}。\n"
        f"分析对象：{company_name}（股票代码：{ticker}）。\n"
        "请在分析中正确区分公司名称与股票代码，并全程使用中文。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt


def build_social_prompt(*, tools: List[Any], current_date: str, ticker: str, company_name: str) -> ChatPromptTemplate:
    tool_names = _tool_names(tools)

    system_message = (
        "您是一位专业的中国市场社交媒体与投资情绪分析师。\n\n"
        "职责：分析雪球、东方财富股吧、财经媒体报道、微博/知乎讨论等对投资情绪的影响。\n\n"
        "分析要点：\n"
        "- 情绪强弱与变化趋势（量化评分1-10）；\n"
        "- 关键KOL观点与影响力；\n"
        "- 事件/政策驱动与舆论预期；\n"
        "- 情绪与股价走势的相关性；\n\n"
        "输出必须包含：\n"
        "- 情绪指数评分；\n"
        "- 预期价格波动幅度与方向；\n"
        "- 基于情绪的交易时机建议；\n\n"
        "注意：若数据受限，请明确说明限制与替代方案。\n"
    )

    system = (
        "您是一位与其他分析师协作的AI助手。\n"
        "必须使用提供的工具来推进分析，不得凭空假设。\n"
        f"可用工具：{tool_names}。\n"
        f"{system_message}\n"
        f"当前日期：{current_date}。分析目标：{ticker}（公司：{company_name}）。\n"
        "请用中文撰写所有分析内容。"
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    return prompt
