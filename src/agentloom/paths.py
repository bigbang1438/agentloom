"""路径解析：集中定义项目根目录，避免各处手数 dirname 层数出错。

`__file__` 是 `<root>/src/agentloom/paths.py`，因此上溯三级到 `<root>`。
"""

from __future__ import annotations

import os

SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # <root>/src
PROJECT_ROOT = os.path.dirname(SRC_DIR)  # <root>

SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")
WEB_DIR = os.path.join(PROJECT_ROOT, "web")
EVAL_DIR = os.path.join(PROJECT_ROOT, "eval")
DOCS_DIR = os.path.join(PROJECT_ROOT, "docs")
