"""工具注册表：把模型的每一次工具调用当作不可信输入来校验。

核心信条：**永远不要假设模型会乖乖按 schema 调用工具。**

四道校验：
1. 工具名必须在白名单内（拒绝未知工具，防工具发现攻击）
2. 参数必须是 JSON 对象（拒绝字符串/数组/None）
3. 未声明字段一律拒绝（防参数走私，如伪造 approved=true）
4. 必填、类型、枚举、长度、数值范围逐项校验

校验失败**不抛异常炸掉循环**，而是产出一条 rejected 观察结果回灌给模型，
让它知道错在哪、换个做法。这既保住了可用性，又守住了安全边界。
"""

from __future__ import annotations

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Literal

from agentloom.core.errors import ToolNotFound, ToolValidationError

JsonType = Literal["string", "integer", "number", "boolean", "object", "array"]

_TYPE_MAP: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "object": (dict,),
    "array": (list,),
}


@dataclass
class ParamSpec:
    """单个参数的声明。"""

    name: str
    type: JsonType
    description: str = ""
    required: bool = True
    enum: list[Any] | None = None
    minimum: float | None = None
    maximum: float | None = None
    min_length: int | None = None
    max_length: int | None = None
    default: Any = None

    def to_json_schema(self) -> dict[str, Any]:
        schema: dict[str, Any] = {"type": self.type}
        if self.description:
            schema["description"] = self.description
        if self.enum:
            schema["enum"] = list(self.enum)
        if self.minimum is not None:
            schema["minimum"] = self.minimum
        if self.maximum is not None:
            schema["maximum"] = self.maximum
        if self.min_length is not None:
            schema["minLength"] = self.min_length
        if self.max_length is not None:
            schema["maxLength"] = self.max_length
        return schema


@dataclass
class ToolResult:
    """工具执行结果。ok=False 表示业务失败（例如订单不存在），不是系统异常。"""

    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> dict[str, Any]:
        """产出回灌给模型的观察结果。

        **必须保证结果可 JSON 序列化**：它会被 json.dumps 后放进 tool 消息。
        工具在 data 里放了自定义对象时，这里统一降级成可序列化形式，
        否则整条 Agent 循环会在序列化处炸掉（而不是在工具里）。
        """
        if self.ok:
            payload: dict[str, Any] = {"ok": True, "result": _jsonable(self.data)}
            if self.meta:
                payload["meta"] = _jsonable(
                    {k: v for k, v in self.meta.items() if k != "citation_objects"}
                )
            return payload
        return {"ok": False, "error": self.error or "unknown error"}


@dataclass
class Tool:
    """一个可被模型调用的工具。"""

    name: str
    description: str
    params: list[ParamSpec]
    handler: Callable[..., Any]
    dangerous: bool = False  # 写操作：需要人工确认
    side_effect: bool = False

    # ---- schema ----
    def to_json_schema(self) -> dict[str, Any]:
        """产出给模型的 function 定义。

        additionalProperties=False 是防参数走私的关键：模型不能塞未声明字段。
        """
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {p.name: p.to_json_schema() for p in self.params},
                    "required": [p.name for p in self.params if p.required],
                    "additionalProperties": False,
                },
            },
        }

    # ---- 校验 ----
    def validate(self, raw: Any) -> tuple[dict[str, Any], list[str]]:
        """校验并归一化参数。返回 (clean_args, errors)。

        注意：真实模型给的 `arguments` 是 **JSON 字符串**，不是对象。
        所以字符串要先解析成对象再校验；解析失败也算校验失败。
        """
        errors: list[str] = []

        if isinstance(raw, str):
            try:
                raw = json.loads(raw) if raw.strip() else {}
            except json.JSONDecodeError as exc:
                return {}, [f"arguments is not valid JSON: {exc.msg}"]

        if not isinstance(raw, dict):
            return {}, [f"arguments must be a JSON object, got {type(raw).__name__}"]

        spec_by_name = {p.name: p for p in self.params}
        declared = set(spec_by_name)

        # 1) 未声明字段：直接拒绝，这是参数走私的主通道
        for key in raw:
            if key not in declared:
                errors.append(
                    f"unknown argument {key!r} (allowed: {sorted(declared)})"
                )

        clean: dict[str, Any] = {}
        for name, spec in spec_by_name.items():
            if name not in raw:
                if spec.required:
                    errors.append(f"missing required argument {name!r}")
                elif spec.default is not None:
                    clean[name] = spec.default
                continue

            value = raw[name]
            # bool 是 int 的子类，必须单独挡掉，否则 True 会被当成 1 通过 integer 校验
            expected = _TYPE_MAP[spec.type]
            if spec.type in ("integer", "number") and isinstance(value, bool):
                errors.append(f"{name!r} must be {spec.type}, got boolean")
                continue
            if not isinstance(value, expected):
                errors.append(
                    f"{name!r} must be {spec.type}, got {type(value).__name__}"
                )
                continue

            if spec.enum is not None and value not in spec.enum:
                errors.append(f"{name!r} must be one of {spec.enum}, got {value!r}")
                continue

            if spec.type in ("integer", "number"):
                if spec.minimum is not None and value < spec.minimum:
                    errors.append(f"{name!r} must be >= {spec.minimum}, got {value}")
                    continue
                if spec.maximum is not None and value > spec.maximum:
                    errors.append(f"{name!r} must be <= {spec.maximum}, got {value}")
                    continue

            if spec.type == "string":
                if spec.min_length is not None and len(value) < spec.min_length:
                    errors.append(
                        f"{name!r} must be at least {spec.min_length} chars, got {len(value)}"
                    )
                    continue
                if spec.max_length is not None and len(value) > spec.max_length:
                    errors.append(
                        f"{name!r} must be at most {spec.max_length} chars, got {len(value)}"
                    )
                    continue

            if spec.type == "array" and spec.max_length is not None and len(value) > spec.max_length:
                errors.append(
                    f"{name!r} must have at most {spec.max_length} items, got {len(value)}"
                )
                continue

            clean[name] = value

        return clean, errors


class ToolRegistry:
    """工具白名单注册表。"""

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} already registered")
        self._tools[tool.name] = tool
        return tool

    def tool(
        self,
        name: str,
        description: str,
        params: Iterable[ParamSpec] | None = None,
        *,
        dangerous: bool = False,
        side_effect: bool = False,
    ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
        """装饰器式注册。"""

        def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
            self.register(
                Tool(
                    name=name,
                    description=description,
                    params=list(params or []),
                    handler=fn,
                    dangerous=dangerous,
                    side_effect=side_effect,
                )
            )
            return fn

        return deco

    # ---- 查询 ----
    def get(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise ToolNotFound(f"tool {name!r} is not in the whitelist", name=name)
        return tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return sorted(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    def __iter__(self):
        return iter(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [t.to_json_schema() for t in self._tools.values()]

    # ---- 调用 ----
    def invoke(self, name: str, raw_args: Any) -> ToolResult:
        """校验并执行。校验失败返回 ok=False 的结果，不抛异常。"""
        try:
            tool = self.get(name)
        except ToolNotFound as exc:
            return ToolResult(ok=False, error=str(exc.message), meta={"rejected": True})

        clean, errors = tool.validate(raw_args)
        if errors:
            return ToolResult(
                ok=False,
                error="argument validation failed",
                meta={"rejected": True, "errors": errors},
            )

        try:
            result = tool.handler(**clean)
        except Exception as exc:  # 工具内部异常也要变成观察结果，不能炸掉循环
            return ToolResult(
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
                meta={"rejected": False, "exception": True},
            )

        if isinstance(result, ToolResult):
            return result
        return ToolResult(ok=True, data=result)

    def describe(self) -> str:
        """给模型看的工具清单（人类可读版，用于 system prompt）。"""
        lines: list[str] = []
        for t in self._tools.values():
            req = ", ".join(
                f"{p.name}:{p.type}" + ("" if p.required else "?") for p in t.params
            )
            lines.append(f"- {t.name}({req}): {t.description}")
        return "\n".join(lines)


def spec_from_signature(fn: Callable[..., Any], type_map: dict[type, JsonType] | None = None) -> list[ParamSpec]:
    """（可选）从函数签名推断 ParamSpec，便于快速注册。"""
    tmap = type_map or {str: "string", int: "integer", float: "number", bool: "boolean"}
    out: list[ParamSpec] = []
    for pname, param in inspect.signature(fn).parameters.items():
        jtype = tmap.get(param.annotation, "string")
        out.append(
            ParamSpec(
                name=pname,
                type=jtype,
                required=param.default is inspect.Parameter.empty,
                default=None if param.default is inspect.Parameter.empty else param.default,
            )
        )
    return out


def dumps(obj: Any) -> str:
    """稳定序列化，便于测试断言与轨迹比对。"""
    return json.dumps(obj, ensure_ascii=False, sort_keys=True)


def _jsonable(value: Any) -> Any:
    """把任意值降级成可 JSON 序列化的形式。

    带 to_dict() 的对象（如 Citation）用它自己的表示；
    set/tuple 转 list；其余不可序列化的降级为字符串，
    绝不让序列化异常把整条 Agent 循环炸掉。
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "to_dict"):
        return _jsonable(value.to_dict())
    return str(value)
