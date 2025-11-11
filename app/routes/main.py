from flask import Blueprint, jsonify, request
import os
import json
from datetime import datetime, timedelta
import pandas as pd
from app.utils.llm import get_chat_model

main_bp = Blueprint("main", __name__)


def _safe_read_json(path: str):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _safe_read_csv(path: str):
    try:
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, encoding="utf-8")
        if df.empty:
            return None
        return df
    except Exception:
        return None


def _iso_now(offset_hours: int = 0) -> str:
    return (datetime.utcnow() + timedelta(hours=offset_hours)).isoformat()


@main_bp.route("/messages", methods=["GET"])
def get_user_messages():
    """
    Return practice suggestions for the dashboard messages panel.
    Response schema:
    {
      "items": [{ id, title, body, created_at, read? }, ...],
      "page": number,
      "total_pages": number
    }
    """
    user_id = request.args.get("user_id", "").strip()
    try:
        page = int(request.args.get("page", "1"))
        page_size = int(request.args.get("page_size", "5"))
    except ValueError:
        page, page_size = 1, 5

    # Default messages for new users or on failure
    default_items = [
        {
            "id": 1,
            "title": "Welcome to your dashboard",
            "body": "Explore Writing and Speaking tools to start practicing.",
            "created_at": _iso_now(-1),
            "read": False,
        },
        {
            "id": 2,
            "title": "Tip: Track your progress",
            "body": "Your recent practices will appear in charts below.",
            "created_at": _iso_now(-5),
            "read": True,
        },
        {
            "id": 3,
            "title": "New feature",
            "body": "We’ve improved feedback for coherence and fluency.",
            "created_at": _iso_now(-24),
            "read": True,
        },
    ]

    if not user_id:
        total_pages = 1
        return jsonify({"items": default_items, "page": 1, "total_pages": total_pages})

    # Build paths
    base_dir = os.path.dirname(os.path.dirname(__file__))
    user_dir = os.path.join(base_dir, "data", user_id)

    # Gather writing/speaking signals
    writing_csv = os.path.join(user_dir, "writing_dashboard.csv")
    oral_csv = os.path.join(user_dir, "oral_dashboard.csv")
    writing_hist = os.path.join(user_dir, "writing_history.json")
    oral_hist = os.path.join(user_dir, "oral_history.json")

    df_w = _safe_read_csv(writing_csv)
    df_s = _safe_read_csv(oral_csv)
    his_w = _safe_read_json(writing_hist) or []
    his_s = _safe_read_json(oral_hist) or []

    # If no real data at all, return defaults
    if (df_w is None and df_s is None) and (not his_w and not his_s):
        total_pages = 1
        return jsonify({"items": default_items, "page": 1, "total_pages": total_pages})

    # ==== Build a compact history summary for LLM ====
    def _truncate_text(text: str, limit: int = 400) -> str:
        if not isinstance(text, str):
            return ""
        t = text.strip()
        return t if len(t) <= limit else t[:limit] + "..."

    def _summarize_conversation(conv: list, max_items: int = 8) -> str:
        if not isinstance(conv, list):
            return ""
        lines = []
        for msg in conv[-max_items:]:
            role = msg.get("role", "bot")
            content = _truncate_text(msg.get("content", ""))
            part = msg.get("part", None)
            tag = f" [Part {part}]" if part is not None else ""
            lines.append(f"{role}{tag}: {content}")
        return "\n".join(lines)

    writing_blocks = []
    if isinstance(his_w, list):
        for sess in his_w[-5:]:
            conv = sess.get("conversation", [])
            title = sess.get("title", f"Practice {sess.get('conversation_id', '')}")
            writing_blocks.append(f"- {title}\n{_summarize_conversation(conv)}")

    speaking_blocks = []
    if isinstance(his_s, list):
        for sess in his_s[-5:]:
            conv = sess.get("conversation", [])
            title = sess.get("title", f"Session {sess.get('conversation_id', '')}")
            speaking_blocks.append(f"- {title}\n{_summarize_conversation(conv)}")

    # Include dashboard weak spots if available
    weak_summary = []
    try:
        if df_w is not None:
            w_cols = df_w.apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
            if not w_cols.empty:
                weak_summary.append(f"Writing weakest: {w_cols.idxmin()}")
        if df_s is not None:
            s_cols = df_s.apply(pd.to_numeric, errors="coerce").mean(numeric_only=True)
            if not s_cols.empty:
                weak_summary.append(f"Speaking weakest: {s_cols.idxmin()}")
    except Exception:
        pass

    history_prompt = (
        "You are an IELTS coach. Based on the user's recent Writing and Speaking history, "
        "generate a short list of actionable practice suggestions. "
        "Return STRICT JSON (array) with objects: {\"id\": string, \"title\": string, \"body\": string}. "
        "Do not include any other fields or text. Keep titles concise, bodies practical.\n\n"
        f"Weak summary: {', '.join(weak_summary) if weak_summary else 'N/A'}\n\n"
        "Recent Writing sessions:\n"
        + ("\n".join(writing_blocks) if writing_blocks else "(none)") +
        "\n\nRecent Speaking sessions:\n"
        + ("\n".join(speaking_blocks) if speaking_blocks else "(none)")
    )

    # ==== Call global chat model ====
    llm, init_error = get_chat_model()
    items = []
    if llm is None:
        # Fallback to defaults if LLM unavailable
        items = default_items
    else:
        try:
            resp = llm.invoke([
                # Use SystemMessage/HumanMessage is done inside llm helpers; here we pass a simple string if needed
                # But ChatOpenAI in our utils expects messages list of System/Human; pass as one Human message string.
                # We can just pass a single string; langchain will wrap it as HumanMessage.
                # For safety, keep it as simple content.
                # type: ignore
                # @ts-ignore
                # noqa
                # NOTE: ChatOpenAI.invoke accepts a list of messages; passing a string is also handled.
                history_prompt
            ])
            content = (getattr(resp, "content", "") or "").strip()
            # Try parse JSON array
            parsed = json.loads(content)
            if isinstance(parsed, list):
                for idx, it in enumerate(parsed):
                    if not isinstance(it, dict):
                        continue
                    sid = str(it.get("id", f"sugg-{idx+1}"))
                    title = str(it.get("title", "Practice suggestion")).strip()
                    body = str(it.get("body", "")).strip()
                    if not title or not body:
                        continue
                    items.append({
                        "id": sid,
                        "title": title,
                        "body": body,
                        "created_at": _iso_now(-2),
                        "read": False,
                    })
            # If parsing failed to produce items, fallback
            if not items:
                items = default_items
        except Exception:
            items = default_items

    # Sort by created_at desc
    try:
        items.sort(key=lambda x: x.get("created_at", ""), reverse=True)
    except Exception:
        pass

    # Pagination
    total = len(items)
    page = max(1, page)
    page_size = max(1, min(page_size, 50))
    total_pages = (total + page_size - 1) // page_size
    start = (page - 1) * page_size
    end = start + page_size
    paged = items[start:end]

    return jsonify({"items": paged, "page": page, "total_pages": max(1, total_pages)})