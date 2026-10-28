"""
Phase 11: the prompt format qwen3.5 is served with, in Python.

Ollama does not use a template file for qwen3.5 ("TEMPLATE {{ .Prompt }}",
"RENDERER qwen3.5"): the prompt is built by Go code compiled into Ollama,
model/renderers/qwen35.go. render_ollama() is a port of that file at
v0.40.12, the version installed here, for the settings Phase 10 used: tools
given, think false. Standard library only, so the Windows side (raw
requests to Ollama) and the WSL side (training) build the same text.

It differs from the HF chat template in two ways:

1. With think off, Ollama writes earlier assistant turns WITHOUT a think
   block and adds "<think>\\n\\n</think>\\n\\n" only after the final
   "<|im_start|>assistant\\n", where the model writes. The HF template puts
   the empty block in every assistant turn after the user's message.
2. Tool schemas are Go's json.Marshal of Ollama's structs plus spaces:
   "parameters" lists "required" before "properties" and leaves "required"
   out when it is empty. HF writes them in the order they were given.

render_train() is the training text: Ollama's tool block, and every assistant
turn written the way the model sees it at serving, right after the empty
think block. So each target follows exactly the text it follows when served;
only the history differs from Ollama (it keeps the empty blocks, as the HF
template does). It also returns the character spans of the targets.
"""

import json

IM_START, IM_END = "<|im_start|>", "<|im_end|>"
NO_THINK = "<think>\n\n</think>\n\n"

TOOL_POSTAMBLE = """
</tools>

If you choose to call a function ONLY reply in the following format with NO suffix:

<tool_call>
<function=example_function_name>
<parameter=example_parameter_1>
value_1
</parameter>
<parameter=example_parameter_2>
This is the value for the second parameter
that can span
multiple lines
</parameter>
</function>
</tool_call>

<IMPORTANT>
Reminder:
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags
- Required parameters MUST be specified
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls
</IMPORTANT>"""

# Field order of Ollama's api structs; omitempty fields drop out when empty.
_PROPERTY = [("anyOf", True), ("type", True), ("items", True), ("description", True),
             ("enum", True), ("properties", True), ("required", True)]


def _go_json(v):
    """json.Marshal: compact, and <, >, & escaped as Go does by default."""
    s = json.dumps(v, separators=(",", ":"), ensure_ascii=False)
    return s.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")


def _ordered(d, fields):
    out = {}
    for key, omitempty in fields:
        if key in d and not (omitempty and d[key] in (None, "", [], {})):
            out[key] = d[key]
    return out


def _go_tool(tool):
    fn, params = tool["function"], tool["function"].get("parameters", {})
    props = {k: _ordered(p, _PROPERTY) for k, p in (params.get("properties") or {}).items()}
    return _ordered({
        "type": tool["type"],
        "items": tool.get("items"),
        "function": _ordered({
            "name": fn["name"],
            "description": fn.get("description"),
            "parameters": _ordered({"type": params.get("type", ""), "$defs": params.get("$defs"),
                                    "items": params.get("items"),
                                    "required": params.get("required"),
                                    "properties": props},
                                   [("type", False), ("$defs", True), ("items", True),
                                    ("required", True), ("properties", False)]),
        }, [("name", False), ("description", True), ("parameters", False)]),
    }, [("type", False), ("items", True), ("function", False)])


def _add_json_spaces(s):
    """AddJSONSpaces: a space after each ':' and ',' outside strings."""
    out, in_str, esc = [], False, False
    for c in s:
        out.append(c)
        if in_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                in_str = False
        elif c == '"':
            in_str = True
        elif c in ":,":
            out.append(" ")
    return "".join(out)


def _arg(v):
    """formatToolCallArgument: strings raw, containers as compact JSON."""
    if v is None:
        return "null"
    if isinstance(v, str):
        return v
    if isinstance(v, (dict, list)):
        return _go_json(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    return str(v)


def _calls(m):
    """Tool calls in either shape: Ollama's {function: {name, arguments}}
    (as in transcripts) or mailbox's flat {name, arguments}."""
    for c in m.get("tool_calls") or []:
        f = c.get("function", c)
        yield f["name"], f.get("arguments") or {}


def _system(messages, tools):
    sb = []
    if tools:
        sb.append(IM_START + "system\n# Tools\n\nYou have access to the following functions:\n\n<tools>")
        for t in tools:
            sb.append("\n" + _add_json_spaces(_go_json(_go_tool(t))))
        sb.append(TOOL_POSTAMBLE)
        if messages and messages[0]["role"] == "system":
            sys_text = messages[0]["content"].strip()
            if sys_text:
                sb.append("\n\n" + sys_text)
        sb.append(IM_END + "\n")
    elif messages and messages[0]["role"] == "system":
        sb.append(IM_START + "system\n" + messages[0]["content"].strip() + IM_END + "\n")
    return "".join(sb)


def _assistant_body(m):
    content = (m.get("content") or "").strip()
    sb = [content]
    for j, (name, args) in enumerate(_calls(m)):
        if j == 0:
            if content:
                sb.append("\n\n")
        else:
            sb.append("\n")
        sb.append("<tool_call>\n<function=" + name + ">\n")
        for k, v in args.items():
            sb.append("<parameter=" + k + ">\n" + _arg(v) + "\n</parameter>\n")
        sb.append("</function>\n</tool_call>")
    return "".join(sb)


def _render(messages, tools, history_think, finished=False):
    """Shared body. Returns (text, spans of assistant targets). finished: a
    final assistant message is a complete turn (training), not a prefill."""
    sb, spans, pos = [_system(messages, tools)], [], None
    n = len(messages)
    for i, m in enumerate(messages):
        role = m["role"]
        if role == "user" or (role == "system" and i != 0):
            sb.append(IM_START + role + "\n" + (m.get("content") or "").strip() + IM_END + "\n")
        elif role == "assistant":
            sb.append(IM_START + "assistant\n" + (NO_THINK if history_think else ""))
            pos = sum(map(len, sb))
            sb.append(_assistant_body(m))
            if i != n - 1 or finished:
                sb.append(IM_END)
                spans.append((pos, sum(map(len, sb))))
                sb.append("\n")
            else:  # a final assistant message is a prefill: no end tag
                spans.append((pos, sum(map(len, sb))))
        elif role == "tool":
            if i == 0 or messages[i - 1]["role"] != "tool":
                sb.append(IM_START + "user")
            sb.append("\n<tool_response>\n" + (m.get("content") or "").strip() + "\n</tool_response>")
            if i == n - 1 or messages[i + 1]["role"] != "tool":
                sb.append(IM_END + "\n")
    if messages and messages[-1]["role"] != "assistant":
        sb.append(IM_START + "assistant\n" + NO_THINK)
    return "".join(sb), spans


def render_ollama(messages, tools):
    """The prompt Ollama 0.40.12 builds for qwen3.5 with tools and think false."""
    return _render(messages, tools, history_think=False)[0]


def render_train(messages, tools):
    """Training text for a finished episode, and the (start, end) character
    span of each assistant turn's target, end tag included so it learns to
    stop. The conversation ends with the assistant, so nothing is appended."""
    assert messages[-1]["role"] == "assistant", "a training episode ends with the summary"
    return _render(messages, tools, history_think=True, finished=True)
