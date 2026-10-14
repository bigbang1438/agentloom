"""AgentLoom 异常体系。

所有对外抛出的异常都继承自 AgentLoomError，并携带机器可读的 code，
便于 HTTP 层映射为稳定错误码，不泄漏堆栈与内部细节。
"""

from __future__ import annotations

from typing import Any


class AgentLoomError(Exception):
    """所有 AgentLoom 异常的基类。"""

    code = "internal_error"
    http_status = 500

    def __init__(self, message: str, **detail: Any) -> None:
        super().__init__(message)
        self.message = message
        self.detail: dict[str, Any] = detail

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "detail": self.detail}


class ConfigError(AgentLoomError):
    """配置缺失或非法。"""

    code = "config_error"
    http_status = 400


class BudgetExceeded(AgentLoomError):
    """预算耗尽。注意：这不是「失败」，而是「被截断」。"""

    code = "budget_exceeded"
    http_status = 200

    def __init__(self, reason: str, **detail: Any) -> None:
        super().__init__(f"budget exceeded: {reason}", reason=reason, **detail)
        self.reason = reason


class ToolNotFound(AgentLoomError):
    code = "tool_not_found"
    http_status = 400


class ToolValidationError(AgentLoomError):
    """工具参数未通过 schema 校验。

    这是「模型输出不可信」的第一道闸门：校验失败不抛给调用方，
    而是转成拒绝观察结果回灌给模型，让它自己纠正。
    """

    code = "tool_validation_error"
    http_status = 400

    def __init__(self, tool: str, errors: list[str]) -> None:
        super().__init__(f"tool {tool!r} arguments rejected", tool=tool, errors=errors)
        self.tool = tool
        self.errors = errors


class ModelError(AgentLoomError):
    code = "model_error"
    http_status = 502


class ModelTimeout(ModelError):
    code = "model_timeout"
    http_status = 504


class RateLimited(ModelError):
    code = "rate_limited"
    http_status = 429


class GuardrailBlocked(AgentLoomError):
    """护栏拦截。"""

    code = "guardrail_blocked"
    http_status = 400


class NotFound(AgentLoomError):
    code = "not_found"
    http_status = 404
