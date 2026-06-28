"""
app.py — Gradio Candidate Ranking Dashboard (Gradio 6.x compatible)
Run with: python app.py
"""

import sys
import os
import tempfile
from pathlib import Path

import pandas as pd
import numpy as np
import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Score computation ─────────────────────────────────────────────────────────

def compute_scores(df: pd.DataFrame, w_skill: float, w_domain: float,
                   w_traj: float, w_loc: float) -> pd.Series:
    total = w_skill + w_domain + w_traj + w_loc
    w_skill /= total; w_domain /= total; w_traj /= total; w_loc /= total
    base = (
        w_skill  * df["skill_evidence_score"]
        + w_domain * df["domain_fit_score"]
        + w_traj   * df["trajectory_score"]
        + w_loc    * df["location_fit"]
    )
    base = base + df["nth_bonus"].clip(0, 0.05)
    base = base * (1.0 - df["is_honeypot"].astype(float))
    base = base * df["consulting_penalty"]
    base = base * df["disqualifier_penalty"]
    return (base * df["behavioral_multiplier"]).clip(0.0, 1.0)


# ── Load features once ────────────────────────────────────────────────────────

_df_cache = None

def get_df():
    global _df_cache
    if _df_cache is None:
        _df_cache = pd.read_parquet("artifacts/features.parquet")
    return _df_cache


# ── Tab handlers ──────────────────────────────────────────────────────────────

def live_rankings(w_skill, w_domain, w_traj, w_loc, top_n,
                  title_filter, show_honeypots, show_consulting):
    try:
        df = get_df().copy()
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), f"❌ {e}", None

    df["score"] = compute_scores(df, w_skill, w_domain, w_traj, w_loc)

    if not show_honeypots:
        df = df[~df["is_honeypot"]]
    if not show_consulting:
        df = df[~df["is_pure_consulting"]]
    if title_filter and title_filter.strip():
        df = df[df["current_title"].str.contains(title_filter.strip(), case=False, na=False)]

    top = df.nlargest(int(top_n), "score").reset_index(drop=True)
    top.insert(0, "rank", range(1, len(top) + 1))

    cols = ["rank", "candidate_id", "current_title", "location", "yoe",
            "skill_evidence_score", "domain_fit_score", "trajectory_score",
            "score", "notice_period_days", "open_to_work", "is_honeypot"]
    result = top[[c for c in cols if c in top.columns]].copy()
    for col in ["yoe", "skill_evidence_score", "domain_fit_score", "trajectory_score", "score"]:
        if col in result.columns:
            result[col] = result[col].round(4)

    raw = get_df()
    scores_all = compute_scores(raw, w_skill, w_domain, w_traj, w_loc)
    summary = (
        f"**Pool:** {len(raw):,} total | "
        f"**Eligible:** {int((scores_all > 0).sum()):,} | "
        f"**Honeypots:** {int(raw['is_honeypot'].sum())} | "
        f"**Consulting Disq.:** {int(raw['is_pure_consulting'].sum()):,} | "
        f"**Showing top {int(top_n)}**"
    )

    # Write Excel-friendly CSV (UTF-8 BOM so Excel auto-detects encoding)
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix="_ranked_candidates.csv",
        prefix="redrob_top", mode="w", encoding="utf-8-sig"
    )
    result.to_csv(tmp.name, index=False)
    tmp.close()

    return result, summary, tmp.name


def kpi_summary():
    try:
        df = get_df()
    except Exception as e:
        return f"❌ {e}"

    scores = compute_scores(df, 0.30, 0.35, 0.15, 0.10)
    top100 = df.assign(score=scores).nlargest(100, "score")

    lines = [
        "## 📈 Pipeline KPIs\n",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total candidates | {len(df):,} |",
        f"| Eligible (score > 0) | {int((scores > 0).sum()):,} |",
        f"| Honeypots detected | {int(df['is_honeypot'].sum())} |",
        f"| Pure consulting (disqualified) | {int(df['is_pure_consulting'].sum()):,} |",
        f"| Title chasers flagged | {int(df['title_chaser_flag'].sum())} |",
        f"| Research-only flagged | {int(df['research_only_flag'].sum())} |",
        f"| Open to work | {int(df['open_to_work'].sum()):,} |",
        f"| Avg YOE — top 100 | {top100['yoe'].mean():.1f} yrs |",
        f"| Avg skill score — top 100 | {top100['skill_evidence_score'].mean():.3f} |",
        f"| Avg domain fit — top 100 | {top100['domain_fit_score'].mean():.3f} |",
        f"| Score range (top 100) | {scores.nlargest(100).min():.4f} – {scores.max():.4f} |",
        "\n---\n## 🥇 Top 10 Candidates\n",
        "| Rank | ID | Title | YOE | Domain | Skill | Score |",
        "|------|----|-------|-----|--------|-------|-------|",
    ]
    for i, row in top100.head(10).reset_index(drop=True).iterrows():
        lines.append(
            f"| #{i+1} | {row['candidate_id']} | {row['current_title']} | "
            f"{row['yoe']:.1f} | {row['domain_fit_score']:.3f} | "
            f"{row['skill_evidence_score']:.3f} | {row['score']:.4f} |"
        )
    return "\n".join(lines)


def get_honeypots():
    """Return honeypot candidates as a DataFrame."""
    try:
        df = get_df()
        hp = df[df["is_honeypot"]][
            ["candidate_id", "current_title", "location", "yoe",
             "honeypot_score", "domain_fit_score", "skill_evidence_score"]
        ].sort_values("honeypot_score", ascending=False).reset_index(drop=True)
        hp.insert(0, "rank", range(1, len(hp) + 1))
        hp = hp.round(4)
        return hp, f"🍯 **{len(hp)} honeypots detected** (sorted by honeypot_score)"
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), f"❌ {e}"


def get_consulting():
    """Return pure consulting disqualified as a DataFrame."""
    try:
        df = get_df()
        consult = df[df["is_pure_consulting"]][
            ["candidate_id", "current_title", "location", "yoe",
             "domain_fit_score", "skill_evidence_score", "trajectory_score"]
        ].reset_index(drop=True)
        consult.insert(0, "rank", range(1, len(consult) + 1))
        consult = consult.round(4)
        return consult, f"🏢 **{len(consult):,} pure consulting candidates disqualified**"
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), f"❌ {e}"


def get_flagged():
    """Return title chasers and research-only flagged candidates."""
    try:
        df = get_df()
        flagged = df[(df["title_chaser_flag"]) | (df["research_only_flag"])].copy()
        flagged["flags"] = (
            flagged["title_chaser_flag"].map({True: "Title Chaser", False: ""}) + " " +
            flagged["research_only_flag"].map({True: "Research Only", False: ""})
        ).str.strip()
        result = flagged[["candidate_id", "current_title", "yoe", "flags",
                           "trajectory_score", "domain_fit_score"]].reset_index(drop=True)
        result.insert(0, "rank", range(1, len(result) + 1))
        result = result.round(4)
        return result, f"🚩 **{len(result)} candidates flagged** (title chaser / research-only)"
    except Exception as e:
        return pd.DataFrame({"error": [str(e)]}), f"❌ {e}"


def submission_preview():
    if not os.path.exists("submission.csv"):
        return pd.DataFrame({"note": ["Run python src/rank.py to generate submission.csv"]}), None, "⚠️ No submission.csv found."
    sub = pd.read_csv("submission.csv")
    info = (
        f"✅ **submission.csv** — {len(sub)} rows | "
        f"Top score: **{sub['score'].max():.4f}** | "
        f"Bottom score: **{sub['score'].min():.4f}**"
    )
    # Write UTF-8 BOM CSV so Excel opens it without encoding issues
    tmp = tempfile.NamedTemporaryFile(
        delete=False, suffix="_submission.csv",
        prefix="redrob_submission", mode="w", encoding="utf-8-sig"
    )
    sub.to_csv(tmp.name, index=False)
    tmp.close()
    return sub, tmp.name, info



def upload_and_rank(file_obj):
    if file_obj is None:
        return None, "Please upload a candidates.jsonl file (≤500 rows for speed)."
    import subprocess
    file_path = file_obj.name
    out_features = "artifacts/sandbox_features.parquet"
    out_csv = "artifacts/sandbox_submission.csv"
    try:
        r1 = subprocess.run(
            ["python", "src/precompute/engineer_features.py",
             "--candidates", file_path,
             "--taxonomy", "artifacts/skill_taxonomy.json",
             "--jd", "job_requirements.yaml",
             "--out", out_features],
            capture_output=True, text=True, check=True
        )
        r2 = subprocess.run(
            ["python", "src/rank.py",
             "--candidates", file_path,
             "--features", out_features,
             "--out", out_csv],
            capture_output=True, text=True, check=True
        )
        return out_csv, f"✅ Done!\n\n{r2.stdout}"
    except subprocess.CalledProcessError as e:
        return None, f"❌ Pipeline error:\n{e.stderr or e.stdout}"


# ── Build Gradio App ──────────────────────────────────────────────────────────

with gr.Blocks(title="Candidate Ranking Dashboard") as demo:

    gr.HTML("""
    <div style="text-align:center; padding: 20px 0 8px 0;">
        <h1 style="font-size:2rem; font-weight:800; margin:0;
                   background: linear-gradient(90deg,#64b5f6,#ab47bc);
                   -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🏆 Candidate Ranking Dashboard
        </h1>
        <p style="color:#8892b0; font-size:0.9rem; margin:4px 0 0 0;">
            Intelligent Candidate Discovery &amp; Ranking — 100k Pool Explorer
        </p>
    </div>
    """)

    # ── Tab 1: Live Rankings ──────────────────────────────────────────────────
    with gr.Tab("🥇 Live Rankings"):
        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                w_skill  = gr.Slider(0.10, 0.60, value=0.30, step=0.05, label="🧠 Skill Evidence")
                w_domain = gr.Slider(0.10, 0.60, value=0.35, step=0.05, label="🏗️ Domain Fit")
                w_traj   = gr.Slider(0.05, 0.40, value=0.15, step=0.05, label="📈 Trajectory")
                w_loc    = gr.Slider(0.05, 0.30, value=0.10, step=0.05, label="📍 Location Fit")
                top_n    = gr.Slider(10, 100, value=50, step=10, label="Top N")
                title_filter    = gr.Textbox(label="Filter by Title", placeholder="e.g. ML Engineer")
                show_honeypots  = gr.Checkbox(label="Show Honeypots", value=False)
                show_consulting = gr.Checkbox(label="Show Pure Consulting", value=False)
                rank_btn = gr.Button("🔄 Recompute Rankings", variant="primary", size="lg")
            with gr.Column(scale=4):
                rank_summary = gr.Markdown(value="*Click **Recompute Rankings** to load.*")
                rank_table   = gr.Dataframe(interactive=False, wrap=False, label="Top Candidates")
                rank_download = gr.File(
                    label="⬇️ Download Ranked Results as CSV (Excel-ready)",
                    visible=True
                )

        rank_btn.click(
            fn=live_rankings,
            inputs=[w_skill, w_domain, w_traj, w_loc, top_n,
                    title_filter, show_honeypots, show_consulting],
            outputs=[rank_table, rank_summary, rank_download]
        )

    # ── Tab 2: KPI Summary ────────────────────────────────────────────────────
    with gr.Tab("📊 KPI Summary"):
        kpi_btn = gr.Button("🔄 Load KPIs", variant="primary")
        kpi_out = gr.Markdown()
        kpi_btn.click(fn=kpi_summary, inputs=[], outputs=[kpi_out])

    # ── Tab 3: Honeypot Inspector ─────────────────────────────────────────────
    with gr.Tab("🕵️ Honeypot Inspector"):
        gr.Markdown("Click the buttons below to load each category separately.")
        with gr.Row():
            hp_btn  = gr.Button("🍯 Load Honeypots", variant="primary")
            con_btn = gr.Button("🏢 Load Consulting Disqualified", variant="primary")
            fl_btn  = gr.Button("🚩 Load Flagged (Title Chaser / Research)", variant="secondary")

        hp_info  = gr.Markdown()
        hp_table = gr.Dataframe(interactive=False, wrap=False, label="Honeypots")

        con_info  = gr.Markdown()
        con_table = gr.Dataframe(interactive=False, wrap=False, label="Consulting Disqualified")

        fl_info  = gr.Markdown()
        fl_table = gr.Dataframe(interactive=False, wrap=False, label="Flagged Candidates")

        hp_btn.click(fn=get_honeypots,   inputs=[], outputs=[hp_table,  hp_info])
        con_btn.click(fn=get_consulting, inputs=[], outputs=[con_table, con_info])
        fl_btn.click(fn=get_flagged,     inputs=[], outputs=[fl_table,  fl_info])

    # ── Tab 4: Submission Preview ─────────────────────────────────────────────
    with gr.Tab("📋 Submission Preview"):
        sub_btn   = gr.Button("🔄 Load submission.csv", variant="primary")
        sub_info  = gr.Markdown()
        sub_table = gr.Dataframe(interactive=False, wrap=True, label="Submission")
        sub_file  = gr.File(label="⬇️ Download submission.csv (Excel-ready)")
        sub_btn.click(
            fn=submission_preview,
            inputs=[],
            outputs=[sub_table, sub_file, sub_info]
        )


    # ── Tab 5: Upload & Re-rank ───────────────────────────────────────────────
    with gr.Tab("📤 Upload & Re-rank"):
        gr.Markdown("### Upload a small `.jsonl` file (≤ 500 candidates) to test the full pipeline end-to-end")
        file_input = gr.File(label="Upload candidates.jsonl", file_types=[".jsonl"])
        run_btn    = gr.Button("🚀 Run Pipeline", variant="primary", size="lg")
        with gr.Row():
            dl_file = gr.File(label="⬇️ Download Ranked CSV")
            log_out = gr.Textbox(label="Pipeline Logs", lines=20, interactive=False)
        run_btn.click(fn=upload_and_rank, inputs=[file_input], outputs=[dl_file, log_out])

    gr.HTML("""
    <div style="text-align:center; color:#4a5568; font-size:0.78rem; padding:12px 0 4px 0; border-top:1px solid #1e2130;">
        Candidate Ranking Pipeline &bull; Phases 1–3 Complete &bull;
        <a href="https://github.com/rudrakshmala/Candidate_selection_system-" style="color:#64b5f6;">GitHub</a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(share=True, server_port=7860)
