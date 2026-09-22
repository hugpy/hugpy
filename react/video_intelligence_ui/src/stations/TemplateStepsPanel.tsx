/**
 * Template steps — the ordered task outline of a registry task-template
 * (settings NS `task_templates`), rendered so the operator can SEE every step
 * and CHOOSE the model for each one. This is the missing cockpit: the
 * template existed only as API data with no screen behind Generate.
 *
 * Reads: GET /llm/templates (open). Saves one slot at a time via
 * POST /llm/templates/<id>/tasks/<task> {model} (operator-gated) — a 401/403
 * or any refusal is rendered VERBATIM on the row it belongs to; the panel
 * itself never hides. Model options come from the assist discovery endpoint,
 * which already excludes operator-blocked and mistagged models, so a choice
 * made here is a choice the resolver will actually serve.
 */
import { useCallback, useEffect, useState } from "react";

import { request, okValue, errorOf, describeAppError } from "../transport/client";
import { hugpyConfig } from "../config";

type TemplateTask = { name: string; desc: string; model: string | null };
type TemplateRec = {
  id: string;
  name: string;
  active: boolean;
  groups: string[];
  tasks: TemplateTask[];
};

const TEMPLATES_URL = `${hugpyConfig.apiBase}/llm/templates`;

export function TemplateStepsPanel({ templateId }: { templateId?: string }) {
  const [templates, setTemplates] = useState<TemplateRec[]>([]);
  const [picked, setPicked] = useState<string | null>(templateId ?? null);
  const [models, setModels] = useState<string[]>([]);
  const [rowBusy, setRowBusy] = useState<string | null>(null);
  const [rowError, setRowError] = useState<Record<string, string>>({});
  const [loadError, setLoadError] = useState<string | null>(null);

  const load = useCallback(async () => {
    const res = await request<unknown>(TEMPLATES_URL, {
      method: "GET",
      timeoutMs: 15_000,
      meta: { operation: "templates.list" },
    });
    if (!res.ok) {
      setLoadError(describeAppError(errorOf(res)));
      return;
    }
    const value = okValue(res) as { templates?: TemplateRec[] } | null;
    const rows = Array.isArray(value?.templates) ? value.templates : [];
    setTemplates(rows);
    setLoadError(null);
    // Keep the caller's pick when it still exists; otherwise prefer the
    // active template, then the first one. Never invent an id.
    setPicked((cur) =>
      cur && rows.some((t) => t.id === cur)
        ? cur
        : rows.find((t) => t.active)?.id ?? rows[0]?.id ?? null,
    );
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  // Offerable generators — same feed the assist picker uses (blocked models
  // are excluded server-side). A discovery failure leaves the dropdowns with
  // just the current value; saving is still possible via "unset".
  useEffect(() => {
    let alive = true;
    (async () => {
      const res = await request<unknown>(hugpyConfig.promptAssistModelsUrl, {
        method: "GET",
        timeoutMs: 15_000,
        meta: { operation: "templates.models" },
      });
      if (!alive || !res.ok) return;
      const value = okValue(res) as { models?: { model: string }[] } | null;
      setModels(
        (Array.isArray(value?.models) ? value.models : []).map((m) => m.model),
      );
    })();
    return () => {
      alive = false;
    };
  }, []);

  const tpl = templates.find((t) => t.id === picked) ?? null;

  const saveSlot = useCallback(
    async (taskName: string, model: string | null) => {
      if (!tpl) return;
      setRowBusy(taskName);
      const url = `${TEMPLATES_URL}/${encodeURIComponent(
        tpl.id,
      )}/tasks/${encodeURIComponent(taskName)}`;
      const res = await request<unknown>(url, {
        method: "POST",
        body: JSON.stringify({ model }),
        headers: { "Content-Type": "application/json" },
        timeoutMs: 15_000,
        meta: { operation: "templates.fill" },
      });
      setRowBusy(null);
      if (!res.ok) {
        setRowError((e) => ({ ...e, [taskName]: describeAppError(errorOf(res)) }));
        return;
      }
      setRowError((e) => {
        const { [taskName]: _gone, ...rest } = e;
        return rest;
      });
      void load();
    },
    [tpl, load],
  );

  return (
    <section className="vi-script-panel">
      <h3>
        Template steps
        <span className="vi-script-sub">
          the ordered plan — every step, and the model it calls
        </span>
      </h3>

      {loadError && <p className="vi-script-note">{loadError}</p>}

      <div className="vi-script-row">
        {templates.map((t) => (
          <button
            key={t.id}
            type="button"
            className={`vi-btn vi-btn-sm${t.id === picked ? " vi-btn-on" : ""}`}
            onClick={() => setPicked(t.id)}
          >
            {t.name}
            {t.active ? " · active" : ""}
          </button>
        ))}
        {templates.length === 0 && !loadError && (
          <span className="vi-script-note">No templates in the registry.</span>
        )}
      </div>

      {tpl && (
        <>
          <p className="vi-script-note">
            groups (stage order): <code>{tpl.groups.join(" → ") || "—"}</code>
          </p>
          <ol className="vi-template-steps">
            {tpl.tasks.map((task) => (
              <li key={task.name} className="vi-template-step">
                <div className="vi-template-step-head">
                  <code>{task.name}</code>
                  <select
                    className="vi-knob-input"
                    value={task.model ?? ""}
                    disabled={rowBusy === task.name}
                    onChange={(e) =>
                      void saveSlot(task.name, e.target.value || null)
                    }
                  >
                    <option value="">unset — router picks from groups</option>
                    {/* Keep a pinned model visible even when discovery
                        doesn't offer it right now (worker offline). */}
                    {task.model && !models.includes(task.model) && (
                      <option value={task.model}>{task.model}</option>
                    )}
                    {models.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                  {rowBusy === task.name && (
                    <span className="vi-script-note">saving…</span>
                  )}
                </div>
                <p className="vi-template-step-desc">{task.desc}</p>
                {rowError[task.name] && (
                  <p className="vi-script-note">{rowError[task.name]}</p>
                )}
              </li>
            ))}
          </ol>
        </>
      )}
    </section>
  );
}
