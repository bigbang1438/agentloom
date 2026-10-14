"""工具注册表测试：白名单、参数校验、参数走私防御。

这一层是「模型输出不可信」的第一道闸门，测不严等于没设防。
"""

from __future__ import annotations

import pytest

from agentloom.tools.registry import ParamSpec, Tool, ToolRegistry, ToolResult


@pytest.fixture
def registry() -> ToolRegistry:
    reg = ToolRegistry()

    @reg.tool(
        "lookup_order",
        "查询订单",
        [ParamSpec("order_no", "string", min_length=3, max_length=32)],
    )
    def lookup_order(order_no: str):
        return {"order_no": order_no, "status": "shipped"}

    @reg.tool(
        "check_stock",
        "查询库存",
        [
            ParamSpec("sku", "string", min_length=3),
            ParamSpec("top_k", "integer", required=False, default=5, minimum=1, maximum=12),
        ],
    )
    def check_stock(sku: str, top_k: int = 5):
        return {"sku": sku, "top_k": top_k}

    @reg.tool(
        "set_status",
        "设置状态",
        [ParamSpec("status", "string", enum=["open", "closed"])],
    )
    def set_status(status: str):
        return {"status": status}

    @reg.tool(
        "boom",
        "总是抛异常的工具",
        [],
    )
    def boom():
        raise RuntimeError("internal failure")

    return reg


# ---------------------------------------------------------------- 白名单

def test_unknown_tool_is_rejected(registry: ToolRegistry) -> None:
    result = registry.invoke("delete_everything", {})
    assert result.ok is False
    assert "whitelist" in (result.error or "")
    assert result.meta.get("rejected") is True


def test_registry_exposes_only_registered_tools(registry: ToolRegistry) -> None:
    assert registry.names() == ["boom", "check_stock", "lookup_order", "set_status"]
    assert len(registry) == 4


def test_duplicate_registration_is_refused(registry: ToolRegistry) -> None:
    with pytest.raises(ValueError):
        registry.register(
            Tool(name="lookup_order", description="dup", params=[], handler=lambda: None)
        )


# ---------------------------------------------------------------- 参数走私

def test_undeclared_field_is_rejected(registry: ToolRegistry) -> None:
    """典型攻击：往只读查询里塞 approved=true，试图绕过审批。"""
    result = registry.invoke("lookup_order", {"order_no": "A1024", "approved": True})
    assert result.ok is False
    assert result.meta.get("rejected") is True
    assert any("approved" in e for e in result.meta.get("errors", []))


def test_undeclared_field_rejected_even_when_otherwise_valid(registry: ToolRegistry) -> None:
    result = registry.invoke("check_stock", {"sku": "SKU-1001", "top_k": 3, "admin": 1})
    assert result.ok is False
    assert any("admin" in e for e in result.meta.get("errors", []))


def test_schema_declares_additional_properties_false(registry: ToolRegistry) -> None:
    schema = registry.get("lookup_order").to_json_schema()
    assert schema["function"]["parameters"]["additionalProperties"] is False
    assert schema["type"] == "function"


# ---------------------------------------------------------------- 类型

@pytest.mark.parametrize(
    "args,reason",
    [
        (["a"], "must be a JSON object"),
        (123, "must be a JSON object"),
        (None, "must be a JSON object"),
        ("{not json", "not valid JSON"),
        ('"a string"', "must be a JSON object"),
    ],
)
def test_arguments_must_be_object(registry: ToolRegistry, args, reason: str) -> None:
    """模型给的 arguments 可能是各种形态，全都要挡住。"""
    result = registry.invoke("lookup_order", args)
    assert result.ok is False
    errors = result.meta.get("errors") or [result.error]
    assert any(reason in e for e in errors), errors


def test_json_string_arguments_are_accepted(registry: ToolRegistry) -> None:
    """真实模型给的是 JSON 字符串，必须能正常解析执行。"""
    result = registry.invoke("lookup_order", '{"order_no": "A1024"}')
    assert result.ok is True
    assert result.data["order_no"] == "A1024"


def test_empty_string_arguments_treated_as_no_args(registry: ToolRegistry) -> None:
    assert registry.invoke("boom", "").ok is False  # boom 自己抛异常，但参数解析通过
    assert registry.invoke("boom", "").meta.get("exception") is True


def test_missing_required_argument(registry: ToolRegistry) -> None:
    result = registry.invoke("lookup_order", {})
    assert result.ok is False
    assert any("missing required" in e for e in result.meta.get("errors", []))


def test_wrong_type_is_rejected(registry: ToolRegistry) -> None:
    result = registry.invoke("lookup_order", {"order_no": 12345})
    assert result.ok is False
    assert any("must be string" in e for e in result.meta.get("errors", []))


def test_boolean_is_not_accepted_as_integer(registry: ToolRegistry) -> None:
    """bool 是 int 的子类，不单独挡掉的话 True 会被当成 1 通过校验。"""
    result = registry.invoke("check_stock", {"sku": "SKU-1001", "top_k": True})
    assert result.ok is False
    assert any("boolean" in e for e in result.meta.get("errors", []))


def test_numeric_range_is_enforced(registry: ToolRegistry) -> None:
    assert registry.invoke("check_stock", {"sku": "SKU-1001", "top_k": 0}).ok is False
    assert registry.invoke("check_stock", {"sku": "SKU-1001", "top_k": 99}).ok is False
    assert registry.invoke("check_stock", {"sku": "SKU-1001", "top_k": 12}).ok is True


def test_string_length_is_enforced(registry: ToolRegistry) -> None:
    assert registry.invoke("lookup_order", {"order_no": "AB"}).ok is False
    assert registry.invoke("lookup_order", {"order_no": "A" * 33}).ok is False
    assert registry.invoke("lookup_order", {"order_no": "A1024"}).ok is True


def test_enum_is_enforced(registry: ToolRegistry) -> None:
    assert registry.invoke("set_status", {"status": "deleted"}).ok is False
    assert registry.invoke("set_status", {"status": "open"}).ok is True


def test_optional_default_is_applied(registry: ToolRegistry) -> None:
    result = registry.invoke("check_stock", {"sku": "SKU-1001"})
    assert result.ok is True
    assert result.data["top_k"] == 5


# ---------------------------------------------------------------- 异常与业务失败

def test_tool_exception_becomes_observation_not_crash(registry: ToolRegistry) -> None:
    """工具内部异常不能炸掉 Agent 循环，要变成一条观察结果。"""
    result = registry.invoke("boom", {})
    assert result.ok is False
    assert result.meta.get("exception") is True
    assert "RuntimeError" in (result.error or "")


def test_toolresult_is_passed_through() -> None:
    reg = ToolRegistry()

    @reg.tool("business_fail", "业务失败", [])
    def business_fail():
        return ToolResult(ok=False, error="order not found")

    result = reg.invoke("business_fail", {})
    assert result.ok is False
    assert result.error == "order not found"
    assert result.meta.get("rejected") is not True  # 业务失败不是被拒绝


def test_observation_shape_for_success_and_failure(registry: ToolRegistry) -> None:
    ok = registry.invoke("lookup_order", {"order_no": "A1024"})
    assert ok.to_observation() == {"ok": True, "result": {"order_no": "A1024", "status": "shipped"}}

    bad = registry.invoke("lookup_order", {})
    obs = bad.to_observation()
    assert obs["ok"] is False
    assert "error" in obs


# ---------------------------------------------------------------- schema 输出

def test_schemas_list_all_tools(registry: ToolRegistry) -> None:
    schemas = registry.schemas()
    assert len(schemas) == 4
    names = {s["function"]["name"] for s in schemas}
    assert names == {"lookup_order", "check_stock", "set_status", "boom"}


def test_required_list_only_contains_required_params(registry: ToolRegistry) -> None:
    schema = registry.get("check_stock").to_json_schema()
    assert schema["function"]["parameters"]["required"] == ["sku"]


def test_describe_is_human_readable(registry: ToolRegistry) -> None:
    text = registry.describe()
    assert "lookup_order(order_no:string)" in text
    assert "check_stock(sku:string, top_k:integer?)" in text
