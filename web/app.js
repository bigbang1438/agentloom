/* AgentLoom 工作台前端。原生 JS，无框架、无 CDN。
 *
 * 安全要点：
 * - 所有来自服务端的文本一律用 textContent 写入，绝不拼 innerHTML。
 * - CSRF token 每次从 /api/state 读取，不写死在页面里。
 */

(function () {
  "use strict";

  var state = { csrf: "", pending: [] };

  var $ = function (id) { return document.getElementById(id); };

  function api(path, options) {
    var opts = options || {};
    var headers = { "Content-Type": "application/json" };
    if (opts.method && opts.method !== "GET") {
      headers["X-CSRF-Token"] = state.csrf;
    }
    return fetch(path, {
      method: opts.method || "GET",
      headers: headers,
      body: opts.body ? JSON.stringify(opts.body) : undefined,
    }).then(function (resp) {
      return resp.json().then(function (data) {
        if (!resp.ok) {
          var msg = (data && data.error && data.error.message) || ("HTTP " + resp.status);
          throw new Error(msg);
        }
        return data;
      });
    });
  }

  /* ---------------- 渲染：工具 ---------------- */

  function renderTools(tools) {
    var ul = $("tools");
    ul.textContent = "";
    $("tool-count").textContent = tools.length + " 个";
    tools.forEach(function (t) {
      var li = document.createElement("li");
      var name = document.createElement("div");
      name.className = "tname";
      name.textContent = t.name;
      if (t.dangerous) {
        var d = document.createElement("span");
        d.className = "danger";
        d.textContent = "写操作 · 需人工确认";
        name.appendChild(d);
      }
      var desc = document.createElement("div");
      desc.className = "tdesc";
      desc.textContent = t.description;
      li.appendChild(name);
      li.appendChild(desc);
      ul.appendChild(li);
    });
  }

  /* ---------------- 渲染：轨迹 ---------------- */

  function renderTrace(steps) {
    var ol = $("trace");
    ol.textContent = "";
    if (!steps || !steps.length) {
      var li = document.createElement("li");
      li.className = "empty";
      li.textContent = "暂无轨迹";
      ol.appendChild(li);
      return;
    }
    steps.forEach(function (s) {
      var li = document.createElement("li");

      var head = document.createElement("div");
      head.className = "head";

      var idx = document.createElement("span");
      idx.className = "idx";
      idx.textContent = "#" + s.index;

      var kind = document.createElement("span");
      kind.className = "kind " + s.kind;
      kind.textContent = s.kind;

      var name = document.createElement("span");
      name.className = "name";
      name.textContent = s.name;

      var badge = document.createElement("span");
      badge.className = "badge " + s.status;
      badge.textContent = s.status;

      var dur = document.createElement("span");
      dur.className = "dur";
      dur.textContent = (s.duration_ms || 0).toFixed(1) + " ms";

      head.appendChild(idx);
      head.appendChild(kind);
      head.appendChild(name);
      head.appendChild(badge);
      head.appendChild(dur);
      li.appendChild(head);

      if (s.detail && Object.keys(s.detail).length) {
        var det = document.createElement("details");
        var sum = document.createElement("summary");
        sum.textContent = "查看详情";
        var pre = document.createElement("pre");
        pre.textContent = JSON.stringify(s.detail, null, 2);
        det.appendChild(sum);
        det.appendChild(pre);
        li.appendChild(det);
      }
      ol.appendChild(li);
    });
  }

  /* ---------------- 渲染：引用 ---------------- */

  function renderCitations(citations) {
    var box = $("citations");
    box.textContent = "";
    if (!citations || !citations.length) return;

    var title = document.createElement("div");
    title.className = "hint";
    title.textContent = "来源（编号由服务端分配）";
    box.appendChild(title);

    citations.forEach(function (c) {
      var div = document.createElement("div");
      div.className = "cite";

      var head = document.createElement("div");
      head.className = "head";
      var left = document.createElement("span");
      left.textContent = "[" + c.index + "] " + c.filename + " : " + c.start_line + "-" + c.end_line;
      var right = document.createElement("span");
      right.textContent = "score " + c.score;
      head.appendChild(left);
      head.appendChild(right);

      var body = document.createElement("div");
      body.className = "body";
      body.textContent = (c.preview || "").slice(0, 400);

      div.appendChild(head);
      div.appendChild(body);
      box.appendChild(div);
    });
  }

  /* ---------------- 渲染：指标 ---------------- */

  function metricCard(label, value) {
    var d = document.createElement("div");
    d.className = "metric";
    var k = document.createElement("div");
    k.className = "k";
    k.textContent = label;
    var v = document.createElement("div");
    v.className = "v";
    v.textContent = value;
    d.appendChild(k);
    d.appendChild(v);
    return d;
  }

  function renderMetrics(data) {
    var box = $("metrics");
    box.textContent = "";
    var p = data.process || {};
    var lat = p.latency_ms || {};
    box.appendChild(metricCard("运行次数", p.runs_total || 0));
    box.appendChild(metricCard("完成 / 截断", (p.runs_completed || 0) + " / " + (p.runs_truncated || 0)));
    box.appendChild(metricCard("工具调用", p.tool_calls_total || 0));
    box.appendChild(metricCard("被拒调用", p.tool_rejections_total || 0));
    box.appendChild(metricCard("P50 延迟", (lat.p50 || 0) + " ms"));
    box.appendChild(metricCard("P95 延迟", (lat.p95 || 0) + " ms"));
    box.appendChild(metricCard("累计 token", (p.prompt_tokens_total || 0) + (p.completion_tokens_total || 0)));
    box.appendChild(metricCard("估算成本", "$" + (p.estimated_cost_usd_total || 0).toFixed(4)));
    if (p.injection_hits_total) {
      box.appendChild(metricCard("注入命中", p.injection_hits_total));
    }
  }

  /* ---------------- 渲染：审批与任务 ---------------- */

  function renderApprovals(pending) {
    state.pending = pending || [];
    var box = $("approvals");
    box.textContent = "";
    $("approval-count").textContent = state.pending.length ? state.pending.length + " 条待处理" : "";

    if (!state.pending.length) {
      var p = document.createElement("p");
      p.className = "empty";
      p.textContent = "没有待确认的请求";
      box.appendChild(p);
      return;
    }

    state.pending.forEach(function (item) {
      var div = document.createElement("div");
      div.className = "approval";

      var title = document.createElement("div");
      title.className = "title";
      title.textContent = (item.args && item.args.title) || item.tool;

      var meta = document.createElement("div");
      meta.className = "meta";
      meta.textContent = item.tool + " · " + item.approval_id;

      var acts = document.createElement("div");
      acts.className = "acts";

      var yes = document.createElement("button");
      yes.className = "yes";
      yes.textContent = "批准";
      yes.addEventListener("click", function () { decide(item.approval_id, true); });

      var no = document.createElement("button");
      no.className = "no";
      no.textContent = "拒绝";
      no.addEventListener("click", function () { decide(item.approval_id, false); });

      acts.appendChild(yes);
      acts.appendChild(no);
      div.appendChild(title);
      div.appendChild(meta);
      div.appendChild(acts);
      box.appendChild(div);
    });
  }

  function decide(id, approved) {
    api("/api/approvals/" + encodeURIComponent(id), {
      method: "POST",
      body: { approved: approved, note: approved ? "工作台批准" : "工作台拒绝" },
    }).then(function (res) {
      if (res.task) {
        appendTask(res.task);
      }
      refresh();
    }).catch(function (err) {
      alert("处理失败：" + err.message);
    });
  }

  function renderTasks(tasks) {
    var ul = $("tasks");
    ul.textContent = "";
    if (!tasks || !tasks.length) {
      var li = document.createElement("li");
      li.className = "empty";
      li.textContent = "暂无任务";
      ul.appendChild(li);
      return;
    }
    tasks.forEach(function (t) { ul.appendChild(taskLi(t)); });
  }

  function taskLi(t) {
    var li = document.createElement("li");
    li.textContent = "[" + t.priority + "] " + t.title;
    return li;
  }

  function appendTask(t) {
    var ul = $("tasks");
    var empty = ul.querySelector(".empty");
    if (empty) empty.remove();
    ul.insertBefore(taskLi(t), ul.firstChild);
  }

  /* ---------------- 提问 ---------------- */

  function setBusy(busy) {
    $("ask").disabled = busy;
    $("ask").textContent = busy ? "运行中…" : "发送";
  }

  function ask() {
    var q = $("question").value.trim();
    if (!q) return;

    setBusy(true);
    $("answer").textContent = "";
    $("answer").classList.remove("empty");
    $("citations").textContent = "";
    $("trace").textContent = "";
    $("answer-meta").textContent = "";

    if ($("use-stream").checked) {
      askStream(q);
    } else {
      askPlain(q);
    }
  }

  function askPlain(q) {
    api("/api/ask", { method: "POST", body: { question: q } })
      .then(function (res) { showResult(res); })
      .catch(function (err) { showError(err.message); })
      .then(function () { setBusy(false); refresh(); });
  }

  function askStream(q) {
    fetch("/api/ask/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": state.csrf },
      body: JSON.stringify({ question: q }),
    }).then(function (resp) {
      if (!resp.ok) throw new Error("HTTP " + resp.status);
      var reader = resp.body.getReader();
      var decoder = new TextDecoder();
      var buffer = "";
      var steps = [];
      var answerText = "";

      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) return;
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split("\n\n");
          buffer = parts.pop();

          parts.forEach(function (part) {
            var line = part.trim();
            if (!line.startsWith("data:")) return;
            var evt;
            try { evt = JSON.parse(line.slice(5).trim()); } catch (e) { return; }

            if (evt.event === "start") {
              $("answer-meta").textContent = "运行 " + evt.run_id;
            } else if (evt.event === "step") {
              steps.push(evt);
              renderTrace(steps);
            } else if (evt.event === "token") {
              answerText += evt.text;
              $("answer").textContent = answerText;
            } else if (evt.event === "done") {
              steps = evt.steps || steps;
              renderTrace(steps);
              $("answer").textContent = evt.answer || answerText;
              renderCitations(evt.citations);
              $("answer-meta").textContent =
                evt.stopped_reason + " · " + evt.tool_calls + " 次工具调用 · " +
                evt.elapsed_seconds + "s" + (evt.truncated ? " · 被预算截断" : "");
            } else if (evt.event === "error") {
              showError(evt.message);
            }
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (err) {
      showError(err.message);
    }).then(function () { setBusy(false); refresh(); });
  }

  function showResult(res) {
    $("answer").textContent = res.answer || "(无答案)";
    renderTrace(res.steps);
    renderCitations(res.citations);
    $("answer-meta").textContent =
      res.stopped_reason + " · " + res.tool_calls + " 次工具调用 · " +
      res.elapsed_seconds + "s" + (res.truncated ? " · 被预算截断" : "");
    if (res.injection_hits && res.injection_hits.length) {
      $("answer-meta").textContent += " · 检测到疑似提示注入";
    }
  }

  function showError(msg) {
    $("answer").textContent = "出错了：" + msg;
  }

  /* ---------------- 刷新 ---------------- */

  function refresh() {
    api("/api/state").then(function (s) {
      state.csrf = s.csrf_token;
      renderTools(s.tools);
      renderApprovals(s.pending_approvals);
      renderTasks(s.tasks);
      $("backend-pill").textContent = "模型：" + s.model_backend;
      $("backend-pill").className = "pill " + (s.model_backend === "scripted" ? "warn" : "ok");
      $("doc-pill").textContent = "资料 " + s.documents.length + " 篇";
    }).catch(function (err) {
      $("backend-pill").textContent = "连接失败";
      $("backend-pill").className = "pill";
      console.error(err);
    });

    api("/metrics").then(renderMetrics).catch(function () {});
  }

  /* ---------------- 事件绑定 ---------------- */

  document.addEventListener("DOMContentLoaded", function () {
    $("ask").addEventListener("click", ask);
    $("question").addEventListener("keydown", function (e) {
      if (e.key === "Enter") ask();
    });
    Array.prototype.forEach.call(document.querySelectorAll(".ex"), function (b) {
      b.addEventListener("click", function () {
        $("question").value = b.getAttribute("data-q");
        ask();
      });
    });
    refresh();
    setInterval(refresh, 8000);
  });
})();
