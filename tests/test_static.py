"""静态资源端点测试：路径穿越必须被挡住。"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentloom.api import app as app_module
from agentloom.api.app import AppState, app


@pytest.fixture
def client(monkeypatch) -> TestClient:
    fresh = AppState()
    monkeypatch.setattr(app_module, "STATE", fresh)
    monkeypatch.setenv("AGENTLOOM_MODEL_BACKEND", "scripted")
    with TestClient(app, base_url="http://127.0.0.1") as c:
        yield c


def test_index_is_served(client: TestClient) -> None:
    resp = client.get("/")
    assert resp.status_code == 200
    assert "AgentLoom" in resp.text


def test_css_is_served_with_correct_type(client: TestClient) -> None:
    resp = client.get("/static/styles.css")
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/css")


def test_js_is_served_with_correct_type(client: TestClient) -> None:
    resp = client.get("/static/app.js")
    assert resp.status_code == 200
    assert "javascript" in resp.headers["content-type"]


def test_missing_static_file_returns_404(client: TestClient) -> None:
    assert client.get("/static/nope.css").status_code == 404


def test_hidden_file_is_refused(client: TestClient) -> None:
    assert client.get("/static/.env").status_code == 422


def test_path_traversal_with_slash_is_refused(client: TestClient) -> None:
    resp = client.get("/static/..%2F..%2Fpyproject.toml")
    assert resp.status_code in (404, 422)


def test_frontend_uses_textcontent_not_innerhtml() -> None:
    """前端不得用 innerHTML 写入服务端数据（XSS 防线）。

    检查的是**代码里真的调用了** `.innerHTML =`，而不是注释里提到这个词。
    """
    import os
    import re

    from agentloom.paths import WEB_DIR

    js = open(os.path.join(WEB_DIR, "app.js"), encoding="utf-8").read()
    # 去掉块注释与行注释后再找赋值用法
    stripped = re.sub(r"/\*.*?\*/", "", js, flags=re.S)
    stripped = re.sub(r"^\s*//.*$", "", stripped, flags=re.M)

    assert ".innerHTML" not in stripped, "前端出现了 .innerHTML 赋值，存在 XSS 风险"
    assert "document.write" not in stripped
    assert "eval(" not in stripped


def test_frontend_reads_csrf_from_api_not_hardcoded() -> None:
    import os

    from agentloom.paths import WEB_DIR

    js = open(os.path.join(WEB_DIR, "app.js"), encoding="utf-8").read()
    assert "/api/state" in js
    assert "csrf" in js.lower()
