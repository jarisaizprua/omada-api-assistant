"""Turn raw evaluation results into a report a non-specialist can read.

Outputs:
* ``summarize``         - numbers (one dict, easy to compare between runs)
* ``render_markdown``   - the written report saved next to the raw JSON
* ``render_console``    - a short version printed at the end of ``manage.py eval``
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from typing import Any

ICON = {"ok": "✅", "partial": "🟡", "bad": "❌", "skipped": "⏭️"}
LABEL = {"ok": "Correcta", "partial": "Parcial", "bad": "Incorrecta", "skipped": "Omitida"}
TYPE_NAMES = {
    "buscar": "Encontrar un endpoint",
    "configurar": "Configurar algo (campos, permisos)",
    "errores": "Entender un código de error",
    "varios_pasos": "Tareas de varios pasos",
    "autenticacion": "Autenticación y tokens",
    "referencia": "Datos de referencia",
    "fuera_de_alcance": "Cosas que la API no hace",
}
DIFFICULTY_NAMES = {"facil": "Fácil", "media": "Media", "dificil": "Difícil"}


# ------------------------------------------------------------------------------ numbers
def pct(x: float | None) -> str:
    return "—" if x is None else f"{round(x * 100)}%"


def seconds(ms: float) -> str:
    return f"{ms / 1000:.1f} s".replace(".", ",")


def money(x: float | None) -> str:
    if x is None:
        return "—"
    return f"${x:.4f}" if x < 0.1 else f"${x:.2f}"


def _group(records: list[dict], key: str) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        groups[r[key]].append(r)
    rows = []
    for name, items in groups.items():
        n = len(items)
        rows.append(
            {
                "name": name,
                "n": n,
                "ok": sum(r["status"] == "ok" for r in items),
                "partial": sum(r["status"] == "partial" for r in items),
                "bad": sum(r["status"] == "bad" for r in items),
            }
        )
    return rows


def summarize_answers(records: list[dict]) -> dict[str, Any]:
    scored = [r for r in records if r["status"] != "skipped"]
    n = len(scored)
    ok = sum(r["status"] == "ok" for r in scored)
    partial = sum(r["status"] == "partial" for r in scored)
    bad = sum(r["status"] == "bad" for r in scored)

    with_expectation = [r for r in scored if r["checks"]["endpoint_ok"] is not None]
    judged = [r["judge"]["score"] for r in scored if r.get("judge") and "score" in r["judge"]]
    times = [r["elapsed_ms"] for r in scored]
    costs = [r["cost"] for r in scored if r.get("cost") is not None]
    in_tok = [r["usage"].get("input_tokens", 0) for r in scored]
    out_tok = [r["usage"].get("output_tokens", 0) for r in scored]
    answerable = [r for r in scored if r["type"] != "fuera_de_alcance"]

    return {
        "n": n,
        "skipped": len(records) - n,
        "ok": ok,
        "partial": partial,
        "bad": bad,
        "endpoint_found": sum(1 for r in with_expectation if r["checks"]["endpoint_ok"]),
        "endpoint_total": len(with_expectation),
        "search_found": sum(1 for r in with_expectation if r["checks"]["retrieval_ok"]),
        "invented_responses": sum(1 for r in scored if r["checks"]["invented_paths"]),
        "suspicious_responses": sum(1 for r in scored if r["checks"]["suspicious_names"]),
        "looked_up": sum(1 for r in answerable if r["checks"]["looked_up"]),
        "read_detail": sum(1 for r in with_expectation if r["checks"]["read_detail"]),
        "answerable": len(answerable),
        "judge_avg": statistics.mean(judged) if judged else None,
        "judge_n": len(judged),
        "avg_ms": statistics.mean(times) if times else 0,
        "p90_ms": sorted(times)[int(0.9 * (len(times) - 1))] if times else 0,
        "avg_in": statistics.mean(in_tok) if in_tok else 0,
        "avg_out": statistics.mean(out_tok) if out_tok else 0,
        "total_cost": sum(costs) if costs else None,
        "avg_cost": statistics.mean(costs) if costs else None,
        "by_type": sorted(_group(scored, "type"), key=lambda g: g["name"]),
        "by_topic": sorted(_group(scored, "topic"), key=lambda g: (g["ok"] / g["n"], g["name"])),
        "by_difficulty": sorted(_group(scored, "difficulty"), key=lambda g: ("facil", "media", "dificil").index(g["name"])),
    }


def effective_retrieval(retrieval: dict) -> dict:
    """The most realistic search figures available: everyday-wording questions beat literal ones."""
    return retrieval.get("natural") or retrieval


def overall_score(retrieval: dict | None, answers: dict | None) -> tuple[int | None, str]:
    """0-100 grade. 60% answers, 20% search (top-3), 20% no invented paths."""
    ret = effective_retrieval(retrieval)["top3"] if retrieval else None
    ans = grounded = None
    if answers and answers["n"]:
        ans = (answers["ok"] + 0.5 * answers["partial"]) / answers["n"]
        grounded = 1 - answers["invented_responses"] / answers["n"]
    if ans is not None and ret is not None:
        score = 0.6 * ans + 0.2 * ret + 0.2 * grounded
    elif ans is not None:
        score = 0.75 * ans + 0.25 * grounded
    elif ret is not None:
        score = ret
    else:
        return None, "Sin datos"
    value = round(score * 100)
    label = "Excelente" if value >= 90 else "Bueno" if value >= 75 else "Aceptable" if value >= 60 else "Necesita mejoras"
    return value, label


# ------------------------------------------------------------------------------ advice
def recommendations(run: dict) -> list[str]:
    tips: list[str] = []
    ret, ans = run.get("retrieval"), (run.get("answers") or {}).get("summary")
    meta = run["meta"]

    if ret:
        if ret["mode"] == "keyword":
            tips.append(
                "La búsqueda se midió **solo con palabras clave** (sin vectores). Ejecuta de nuevo con la clave de OpenAI "
                "activa para medir la búsqueda real del portal."
            )
        real = effective_retrieval(ret)
        if ret.get("natural") is None:
            tips.append(
                "La prueba de búsqueda usa preguntas que repiten las palabras de la documentación, y por eso es optimista. "
                "Ejecuta `python manage.py eval --only retrieval --natural` una vez para medir con preguntas en lenguaje cotidiano."
            )
        if real["top3"] < 0.85:
            tips.append(
                f"La búsqueda coloca la operación correcta entre las 3 primeras solo en el {pct(real['top3'])} de los casos. "
                "Prueba un modelo de embeddings mayor (`OPENAI_EMBEDDING_MODEL=text-embedding-3-large`, luego "
                "`build-index --full`) o sube `RAG_TOP_K`."
            )
        elif ret["top8"] < 0.95:
            tips.append("Sube `RAG_TOP_K` (por ejemplo a 10) para que el modelo reciba más candidatos.")
        weak = [c for c in ret["categories"] if c["n"] >= 3 and c["top3"] < 0.6][:4]
        if weak:
            tips.append(
                "Categorías donde la búsqueda más falla: " + ", ".join(f"{c['tag']} ({pct(c['top3'])})" for c in weak) + "."
            )

    if ans and ans["n"]:
        if ans["invented_responses"]:
            tips.append(
                f"**{ans['invented_responses']} respuesta(s) inventaron rutas que no existen.** Es el fallo más grave: "
                "usa un modelo más capaz o baja `OPENAI_TEMPERATURE` a 0."
            )
        if ans["answerable"] and ans["looked_up"] / ans["answerable"] < 0.9:
            tips.append(
                "A veces el modelo responde **sin consultar la documentación**. Un modelo más capaz suele cumplir mejor "
                "la regla de buscar primero."
            )
        if ans["endpoint_total"] and ans["search_found"] > ans["endpoint_found"]:
            lost = ans["search_found"] - ans["endpoint_found"]
            tips.append(
                f"En {lost} caso(s) la búsqueda encontró el endpoint pero el modelo **no lo usó** en la respuesta: "
                "el límite está en el modelo, no en el RAG."
            )
        if ans["endpoint_total"] and ans["search_found"] < 0.9 * ans["endpoint_total"]:
            tips.append("En varias preguntas la búsqueda ni siquiera encontró el endpoint: mejora primero el RAG.")
        bad_types = [g for g in ans["by_type"] if g["n"] >= 2 and g["ok"] / g["n"] < 0.5]
        if bad_types:
            tips.append(
                "Tipos de pregunta más débiles: " + ", ".join(TYPE_NAMES.get(g["name"], g["name"]) for g in bad_types) + "."
            )
        if ans["avg_ms"] > 15000:
            tips.append("Las respuestas tardan más de 15 s de media: reduce `OPENAI_REASONING_EFFORT` o usa un modelo más rápido.")
        if ans["skipped"]:
            tips.append(
                f"Se omitieron {ans['skipped']} preguntas que necesitan la guía de acceso. Impórtala con "
                "`python manage.py import-guide` y vuelve a evaluar."
            )
    if not tips:
        tips.append("Sin acciones urgentes. Guarda este informe y compáralo la próxima vez con `--baseline latest`.")
    if meta.get("judge_model") and meta["judge_model"] == meta["model"]:
        tips.append(
            "El mismo modelo responde y califica. Un modelo pequeño no sabe más que sí mismo, así que sus "
            "calificaciones son poco fiables (suele marcar como error respuestas correctas). Aquí su nota solo cuenta "
            "como aviso. Para una medición firme define `OPENAI_JUDGE_MODEL` con un modelo más capaz."
        )
    return tips


# ------------------------------------------------------------------------------ comparison
def comparison(current: dict, baseline: dict) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []

    def add(label: str, new: float | None, old: float | None, fmt, higher_better: bool = True) -> None:
        if new is None or old is None:
            return
        diff = new - old
        better = (diff > 0) == higher_better
        arrow = "＝" if abs(diff) < 1e-9 else ("🟢 mejora" if better else "🔴 empeora")
        rows.append((label, fmt(old), fmt(new), arrow))

    add("Nota global", current["overall"], baseline.get("overall"), lambda v: f"{v:.0f}/100")
    cr, br = current.get("retrieval"), baseline.get("retrieval")
    if cr and br:
        add("Búsqueda: acierta entre los 3 primeros", effective_retrieval(cr)["top3"], effective_retrieval(br)["top3"], pct)
    ca = (current.get("answers") or {}).get("summary")
    ba = ((baseline.get("answers") or {}).get("summary")) or None
    if ca and ba and ca["n"] and ba["n"]:
        add("Respuestas correctas", ca["ok"] / ca["n"], ba["ok"] / ba["n"], pct)
        add("Respuestas con rutas inventadas", ca["invented_responses"], ba["invented_responses"], lambda v: f"{v:.0f}", False)
        add("Tiempo medio", ca["avg_ms"], ba["avg_ms"], seconds, False)
        add("Costo medio por pregunta", ca["avg_cost"], ba["avg_cost"], money, False)
    return rows


# ------------------------------------------------------------------------------ markdown
def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "| " + " | ".join(headers) + " |\n|" + "|".join("---" for _ in headers) + "|\n"
    return head + "\n".join("| " + " | ".join(r) + " |" for r in rows) + "\n"


def _share(g: dict) -> str:
    return f"{g['ok']} de {g['n']} ({pct(g['ok'] / g['n'])})"


def render_markdown(run: dict, baseline: dict | None = None) -> str:
    meta, ret = run["meta"], run.get("retrieval")
    answers = run.get("answers")
    ans = answers["summary"] if answers else None
    records = answers["records"] if answers else []
    score, label = run["overall"], run["verdict"]

    out: list[str] = []
    out.append("# Informe de evaluación — Omada API Assistant\n")
    out.append(
        f"**Fecha:** {meta['started_at'][:16].replace('T', ' ')} · **Duración:** {meta['seconds']:.0f} s  \n"
        f"**Modelo que responde:** `{meta['model']}` · **Embeddings:** `{meta.get('embedding_model') or '—'}` · "
        f"**Controller:** {meta.get('controller_version') or '—'}  \n"
        f"**Calificador (juez):** {'`' + meta['judge_model'] + '`' if meta.get('judge_enabled') else 'desactivado'}\n"
    )

    # ---- headline
    out.append("## Resumen en una mirada\n")
    out.append(f"### Nota global: {score}/100 — {label}\n" if score is not None else "### Sin nota (no se ejecutó nada)\n")
    bullets: list[str] = []
    if ans and ans["n"]:
        bullets.append(
            f"Respondió **correctamente {ans['ok']} de {ans['n']} preguntas ({pct(ans['ok'] / ans['n'])})**; "
            f"{ans['partial']} fueron parciales y {ans['bad']} incorrectas."
        )
        if ans["endpoint_total"]:
            bullets.append(
                f"Nombró el endpoint correcto en **{ans['endpoint_found']} de {ans['endpoint_total']}** preguntas "
                f"({pct(ans['endpoint_found'] / ans['endpoint_total'])})."
            )
        bullets.append(
            "No inventó ninguna ruta." if not ans["invented_responses"]
            else f"**Inventó rutas que no existen en {ans['invented_responses']} respuesta(s).**"
        )
        if ans["answerable"]:
            bullets.append(f"Consultó la documentación antes de responder en el {pct(ans['looked_up'] / ans['answerable'])} de los casos.")
        bullets.append(
            f"Tardó {seconds(ans['avg_ms'])} de media por respuesta"
            + (f" y costó unos {money(ans['avg_cost'])} por pregunta." if ans["avg_cost"] is not None else ".")
        )
    if ret:
        bullets.append(
            f"La búsqueda puso la operación correcta **en el primer lugar el {pct(effective_retrieval(ret)['top1'])}** de las veces y "
            f"**entre las 3 primeras el {pct(effective_retrieval(ret)['top3'])}** ({effective_retrieval(ret)['n']} casos de prueba"
            + (", con palabras de la vida real)." if ret.get("natural") else ", con preguntas «de manual»; es una cifra optimista).")
        )
    out += [f"- {b}" for b in bullets]
    out.append("")
    if score is not None:
        out.append(
            "> **Cómo se calcula la nota:** 60 % respuestas correctas (las parciales cuentan la mitad), 20 % qué tan bien "
            "encuentra la búsqueda y 20 % que no invente rutas. "
            "90+ Excelente · 75–89 Bueno · 60–74 Aceptable · menos de 60 Necesita mejoras.\n"
        )

    if baseline:
        rows = comparison(run, baseline)
        if rows:
            out.append(f"### Comparación con la evaluación anterior ({baseline['meta']['started_at'][:10]})\n")
            out.append(_table(["Medida", "Antes", "Ahora", "Cambio"], [list(r) for r in rows]))

    # ---- search
    if ret:
        out.append("## 1. ¿Encuentra los documentos correctos? (la búsqueda)\n")
        out.append(
            f"Se tomaron **{ret['n']} operaciones al azar**, repartidas entre todas las categorías, y se hizo la pregunta que "
            "haría un desarrollador (por ejemplo, «How do I get site list?»). Se midió en qué posición aparece la operación correcta "
            f"entre los resultados. Modo de búsqueda: **{'combinada (significado + palabras clave)' if ret['mode'] == 'hybrid' else 'solo palabras clave'}**.\n"
        )
        nat, cur = ret.get("natural"), ret.get("curated")
        header = ["Medida", "Pregunta «de manual»"]
        keys = [ret]
        if nat:
            header.append("Palabras de la vida real")
            keys.append(nat)
        if cur:
            header.append("Tus preguntas curadas")
            keys.append(cur)
        header.append("Qué significa")
        meanings = [
            ("En el 1.er lugar", "top1", "El resultado de arriba es justo la operación buscada."),
            ("Entre los 3 primeros", "top3", "Está a la vista casi siempre; es la medida más importante."),
            ("Entre los 8 primeros", "top8", "Llega al modelo; si no está aquí, el modelo no puede usarla."),
        ]
        rows = [[label, *[pct(k[key]) for k in keys], text] for label, key, text in meanings]
        rows.append(["Posición media (MRR)", *[f"{k['mrr']:.2f}" for k in keys], "De 0 a 1. Más cerca de 1 = la operación aparece más arriba."])
        rows.append(["Casos", *[str(k["n"]) for k in keys], ""])
        out.append(_table(header, rows))
        out.append(
            "- **Pregunta «de manual»**: repite las palabras de la documentación; es fácil y por eso optimista.\n"
            + (
                "- **Palabras de la vida real**: las mismas operaciones, pedidas con lenguaje cotidiano "
                "(«reiniciar» en vez de *reboot*). Esta es la cifra que cuenta para la nota.\n"
                if nat
                else "- (Sin medir con palabras de la vida real todavía: usa `eval --only retrieval --natural` una vez.)\n"
            )
            + (
                "- **Tus preguntas curadas**: las preguntas escritas a mano del examen de respuestas, tal como las escribe una persona.\n"
                if cur
                else ""
            )
        )
        weak = ret["categories"][:8]
        if weak:
            out.append("**Categorías donde más cuesta encontrar la operación correcta:**\n")
            out.append(
                _table(
                    ["Categoría", "Casos", "Entre los 3 primeros"],
                    [[c["tag"], str(c["n"]), pct(c["top3"])] for c in weak],
                )
            )
        if ret["misses"]:
            out.append("<details><summary>Ejemplos donde falló</summary>\n")
            out.append(
                _table(
                    ["Pregunta", "Debía aparecer", "Apareció primero"],
                    [[m["query"], f"`{m['expected']}`", f"`{m['got']}`" if m["got"] else "—"] for m in ret["misses"][:15]],
                )
            )
            out.append("</details>\n")

    # ---- answers
    if ans and ans["n"]:
        out.append("## 2. ¿Responde bien? (las respuestas del asistente)\n")
        out.append(
            f"Se hicieron **{ans['n']} preguntas reales** (mezcla de español e inglés, fáciles y difíciles) al asistente, con el mismo modelo y "
            "las mismas herramientas que usa el portal. Cada respuesta se revisó de dos formas: reglas automáticas "
            "(¿nombró el endpoint correcto? ¿incluyó los datos clave? ¿todas las rutas existen?)"
            + (" y un segundo modelo que hace de juez." if meta.get("judge_enabled") else ".")
            + "\n"
        )
        out.append(
            _table(
                ["Resultado", "Preguntas", "Qué significa"],
                [
                    ["✅ Correcta", str(ans["ok"]), "Endpoint correcto, datos clave presentes y nada inventado."],
                    ["🟡 Parcial", str(ans["partial"]), "Va por buen camino pero omite algún dato importante."],
                    ["❌ Incorrecta", str(ans["bad"]), "Endpoint equivocado, rutas inventadas o no reconoce lo que no existe."],
                ],
            )
        )
        if ans["judge_avg"] is not None:
            out.append(f"**Nota media del juez:** {ans['judge_avg']:.1f} de 5 ({ans['judge_n']} respuestas calificadas).\n")

        out.append("**Por tipo de pregunta**\n")
        out.append(
            _table(
                ["Tipo", "Preguntas", "Correctas", "Parciales", "Incorrectas"],
                [[TYPE_NAMES.get(g["name"], g["name"]), str(g["n"]), _share(g), str(g["partial"]), str(g["bad"])] for g in ans["by_type"]],
            )
        )
        out.append("**Por dificultad**\n")
        out.append(
            _table(
                ["Dificultad", "Preguntas", "Correctas", "Parciales", "Incorrectas"],
                [[DIFFICULTY_NAMES.get(g["name"], g["name"]), str(g["n"]), _share(g), str(g["partial"]), str(g["bad"])] for g in ans["by_difficulty"]],
            )
        )
        out.append("**Por tema** (de más débil a más fuerte)\n")
        out.append(
            _table(
                ["Tema", "Preguntas", "Correctas", "Parciales", "Incorrectas"],
                [[g["name"], str(g["n"]), _share(g), str(g["partial"]), str(g["bad"])] for g in ans["by_topic"]],
            )
        )

        out.append("## 3. ¿Inventa cosas? (fiabilidad)\n")
        out.append(
            _table(
                ["Comprobación", "Resultado", "Por qué importa"],
                [
                    ["Rutas inventadas", f"{ans['invented_responses']} de {ans['n']} respuestas", "Una ruta que no existe rompe el código del usuario."],
                    ["Nombres de operación sospechosos", f"{ans['suspicious_responses']} de {ans['n']}", "Aviso suave: parecen operaciones, pero no están en la especificación."],
                    ["Consultó antes de responder", f"{ans['looked_up']} de {ans['answerable']}", "Responder sin buscar es la causa típica de errores."],
                    ["Leyó el detalle del endpoint", f"{ans['read_detail']} de {ans['endpoint_total']}", "Con el detalle, los nombres de campos salen de la documentación."],
                    ["La búsqueda encontró el endpoint", f"{ans['search_found']} de {ans['endpoint_total']}", "Si falla aquí, el problema es el RAG."],
                    ["El modelo nombró el endpoint", f"{ans['endpoint_found']} de {ans['endpoint_total']}", "Si falla aquí (y arriba no), el problema es el modelo."],
                ],
            )
        )

        out.append("## 4. Velocidad y costo\n")
        cost_rows = [
            ["Tiempo medio por respuesta", seconds(ans["avg_ms"])],
            ["El 90 % de las respuestas tarda menos de", seconds(ans["p90_ms"])],
            ["Tokens medios (entrada / salida)", f"{ans['avg_in']:,.0f} / {ans['avg_out']:,.0f}".replace(",", ".")],
            ["Costo medio por pregunta (estimado)", money(ans["avg_cost"])],
            ["Costo de esta evaluación (estimado)", money(ans["total_cost"])],
        ]
        out.append(_table(["Medida", "Valor"], cost_rows))
        out.append(
            "*El costo usa precios de lista de referencia (o los de `EVAL_PRICE_*` en tu `.env`). "
            "Compruébalos en la página de precios de OpenAI.*\n"
        )

        # ---- failures
        failing = [r for r in records if r["status"] in ("bad", "partial")]
        out.append(f"## 5. Dónde falló ({len(failing)} preguntas)\n" if failing else "## 5. Dónde falló\n\nNinguna pregunta falló. 🎉\n")
        for r in sorted(failing, key=lambda r: (r["status"] != "bad", r["id"])):
            out.append(f"### {ICON[r['status']]} {r['id']} · {r['question']}\n")
            out.append(f"*{r['topic']} · {DIFFICULTY_NAMES.get(r['difficulty'], r['difficulty'])} · {TYPE_NAMES.get(r['type'], r['type'])}*\n")
            for reason in r["reasons"]:
                out.append(f"- {reason}")
            if r.get("judge") and r["judge"].get("missing"):
                out.append("- Según el juez faltó: " + "; ".join(r["judge"]["missing"]))
            if r.get("judge") and r["judge"].get("wrong"):
                out.append("- Según el juez es incorrecto: " + "; ".join(r["judge"]["wrong"]))
            snippet = " ".join((r.get("answer") or "").split())[:420]
            if snippet:
                out.append(f"\n> **Respuesta (extracto):** {snippet}{'…' if len(r.get('answer', '')) > 420 else ''}\n")
            else:
                out.append("")

    # ---- recommendations
    out.append("## 6. Qué mejorar\n")
    out += [f"{i}. {tip}" for i, tip in enumerate(run["recommendations"], 1)]
    out.append("")

    # ---- annex
    if records:
        out.append("## Anexo: todas las preguntas\n")
        out.append(
            _table(
                ["", "ID", "Pregunta", "Tema", "Tiempo", "Nota juez"],
                [
                    [
                        ICON[r["status"]],
                        r["id"],
                        r["question"].replace("|", "\\|"),
                        r["topic"],
                        seconds(r["elapsed_ms"]) if r.get("elapsed_ms") else "—",
                        str(r["judge"]["score"]) if r.get("judge") and "score" in r["judge"] else "—",
                    ]
                    for r in records
                ],
            )
        )
    return "\n".join(out) + "\n"


# ------------------------------------------------------------------------------ console
def render_console(run: dict, console, report_path: str) -> None:
    from rich import box
    from rich.panel import Panel
    from rich.table import Table

    ans = (run.get("answers") or {}).get("summary")
    ret = run.get("retrieval")
    score, label = run["overall"], run["verdict"]
    color = "green" if (score or 0) >= 75 else "yellow" if (score or 0) >= 60 else "red"

    table = Table(box=box.SIMPLE, show_header=False, pad_edge=False)
    if ans and ans["n"]:
        table.add_row("Respuestas correctas", f"[bold]{ans['ok']}[/] de {ans['n']} ({pct(ans['ok'] / ans['n'])})   🟡 {ans['partial']} parciales   ❌ {ans['bad']} incorrectas")
        if ans["endpoint_total"]:
            table.add_row("Endpoint correcto", f"{ans['endpoint_found']} de {ans['endpoint_total']}")
        table.add_row("Rutas inventadas", "[green]ninguna[/]" if not ans["invented_responses"] else f"[red]{ans['invented_responses']} respuesta(s)[/]")
        if ans["answerable"]:
            table.add_row("Consultó la documentación", f"{pct(ans['looked_up'] / ans['answerable'])}")
        if ans["judge_avg"] is not None:
            table.add_row("Nota media del juez", f"{ans['judge_avg']:.1f} / 5")
        table.add_row("Tiempo / costo medio", f"{seconds(ans['avg_ms'])} · {money(ans['avg_cost'])} por pregunta")
        if ans["skipped"]:
            table.add_row("Omitidas", f"{ans['skipped']} (necesitan la guía de acceso)")
    if ret:
        table.add_row("Búsqueda «de manual» (1.º / top-3 / top-8)", f"{pct(ret['top1'])} / {pct(ret['top3'])} / {pct(ret['top8'])}  [dim]({ret['mode']})[/]")
        for key, label in (("natural", "Búsqueda vida real"), ("curated", "Búsqueda preguntas curadas")):
            extra = ret.get(key)
            if extra:
                table.add_row(label + " (1.º / top-3 / top-8)", f"{pct(extra['top1'])} / {pct(extra['top3'])} / {pct(extra['top8'])}")
    console.print(
        Panel(
            table,
            title=f"[bold {color}]Nota global: {score}/100 — {label}[/]" if score is not None else "Evaluación",
            border_style=color,
        )
    )
    for tip in run["recommendations"][:4]:
        console.print(f" • {tip}".replace("**", "").replace("`", ""))
    console.print(f"\nInforme completo: [bold]{report_path}[/]")
