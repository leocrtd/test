"""
零依赖 tool_call Agent 演示（只用 Python 标准库，无需 pip install）
多轮对话模式：用户问城市天气 → 模型自己决定调工具 → 拿真实数据回答

  大模型：DeepSeek  /chat/completions 接口，走 OpenAI 兼容协议
  天气源：Open-Meteo 公开免费接口（无需 Key）

运行：
  export DEEPSEEK_API_KEY=sk-xxxxxxxx
  python weather_agent.py
"""
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://api.deepseek.com/chat/completions"
API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
MODEL = "deepseek-chat"

SHOW_LLM_IO = True   # 打印大模型接口的完整入参/出参；不想看就改成 False

WMO = {  # 天气代码 → 中文（WMO 标准）
    0: "晴", 1: "晴间多云", 2: "多云", 3: "阴", 45: "雾", 48: "雾凇",
    51: "毛毛雨", 53: "小雨", 55: "中雨", 61: "小雨", 63: "中雨", 65: "大雨",
    71: "小雪", 73: "中雪", 75: "大雪", 77: "雪粒", 80: "阵雨", 81: "强阵雨",
    82: "暴雨", 85: "阵雪", 86: "强阵雪", 95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨",
}


# ---------- 两个小工具函数：发 GET / 发 POST ----------
def http_get_json(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=15) as r:
        return json.load(r)


def http_post_json(url: str, payload: dict) -> dict:
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.load(r)


def call_llm(payload: dict) -> dict:
    """调用大模型接口的唯一出口：把请求体（入参）和响应体（出参）原样打印出来。"""
    if SHOW_LLM_IO:
        print("\n" + "=" * 72)
        print(f"▶ 请求 POST {API_URL}")
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    data = http_post_json(API_URL, payload)
    if SHOW_LLM_IO:
        print("-" * 72)
        print("◀ 响应")
        print(json.dumps(data, ensure_ascii=False, indent=2))
        print("=" * 72)
    return data


# ---------- 1. 工具：Agent 真正能"动手"做的事 ----------
def get_weather(city: str) -> dict:
    """查城市实时天气：地理编码拿经纬度 → 查当前天气。"""
    q = urllib.parse.urlencode({"name": city, "count": 1, "language": "zh", "format": "json"})
    results = http_get_json(f"https://geocoding-api.open-meteo.com/v1/search?{q}").get("results")
    if not results:
        return {"error": f"找不到城市：{city}"}

    loc = results[0]
    q = urllib.parse.urlencode({
        "latitude": loc["latitude"], "longitude": loc["longitude"],
        "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
        "timezone": "auto",
    })
    cur = http_get_json(f"https://api.open-meteo.com/v1/forecast?{q}")["current"]
    return {
        "城市": f"{loc['name']}（{loc.get('country', '')}）",
        "天气": WMO.get(cur["weather_code"], f"代码{cur['weather_code']}"),
        "温度": f"{cur['temperature_2m']}℃",
        "湿度": f"{cur['relative_humidity_2m']}%",
        "风速": f"{cur['wind_speed_10m']} km/h",
    }


# ---------- 2. 工具的"说明书"，随请求一起发给大模型 ----------
TOOLS = [{
    "type": "function",
    "function": {
        "name": "get_weather",
        "description": "查询指定城市的实时天气（天气状况、温度、湿度、风速）",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "城市名，如 北京、上海、Tokyo"}},
            "required": ["city"],
        },
    },
}]

TOOL_IMPL = {"get_weather": get_weather}  # 名字 → 真实函数

SYSTEM = (
    "你是天气助手。凡涉及天气，必须先调用 get_weather 拿真实数据，不许编造，再用中文简洁作答。"
    "对话是多轮的：用户说“那XX呢”“这里”“那边”这类省略城市的追问时，结合上文判断指的是哪座城市。"
)


# ---------- 3. 一轮问答：模型 → 工具 → 模型 → …… → 最终回答 ----------
def chat(messages: list, question: str, verbose: bool = True) -> str:
    """把用户问题追加进历史，驱动模型直到它给出纯文本回答。返回回答内容。"""
    snapshot = len(messages)          # 记下断点，出错时回滚，避免污染多轮历史
    messages.append({"role": "user", "content": question})

    try:
        for _ in range(6):            # 最多来回 6 轮，防止死循环
            msg = call_llm({"model": MODEL, "messages": messages, "tools": TOOLS})["choices"][0]["message"]

            # 模型没要求调工具 → 这就是最终回答
            if not msg.get("tool_calls"):
                messages.append({"role": "assistant", "content": msg["content"]})
                return msg["content"]

            # 模型要求调工具 → 原样回填这条 assistant 消息（必须带 tool_calls）
            messages.append(msg)

            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                args = json.loads(tc["function"]["arguments"] or "{}")
                if verbose:
                    print(f"  → 调用 {name}({json.dumps(args, ensure_ascii=False)})")
                try:
                    result = TOOL_IMPL[name](**args)
                except Exception as e:   # 工具失败也要回喂，模型可自行补救
                    result = {"error": f"{type(e).__name__}: {e}"}
                if verbose:
                    print(f"  ← 返回 {json.dumps(result, ensure_ascii=False)}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": json.dumps(result, ensure_ascii=False),
                })

        return "（超过最大轮次，已中止）"
    except Exception:
        del messages[snapshot:]       # 本轮作废，保持历史干净
        raise


# ---------- 4. 主程序：多轮对话循环 ----------
def main() -> None:
    if not API_KEY:
        sys.exit("请先设置环境变量 DEEPSEEK_API_KEY")

    messages = [{"role": "system", "content": SYSTEM}]
    print("多轮天气对话（输入 exit 退出）")
    print("示例：北京天气怎么样？ / 那上海呢？ / 明天适合出门吗？\n")

    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            return

        if question.lower() in {"exit", "quit", "q", "退出"}:
            print("再见。")
            return
        if not question:
            continue

        try:
            print(f"助手：{chat(messages, question)}\n")
        except urllib.error.HTTPError as e:
            print(f"[接口错误] HTTP {e.code}：{e.read().decode('utf-8', 'ignore')[:300]}\n")
        except Exception as e:
            print(f"[出错] {type(e).__name__}: {e}\n")


if __name__ == "__main__":
    main()
