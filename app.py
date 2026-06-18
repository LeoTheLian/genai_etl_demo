"""
Streamlit UI for the GenAI ETL Orchestrator.

Step-by-step pipeline with human review between each agent phase.
Artifacts stack vertically on a single scrollable page.

Run:
  streamlit run app.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make the project root importable regardless of cwd
sys.path.insert(0, str(Path(__file__).resolve().parent))

from agents.analyst_agent import analyze_requirements_and_data
from agents.developer_agent import generate_pipeline_code
from agents.tester_agent import run_tests
from agents.orchestrator import _run_pipeline_subprocess
from agents.activity_logger import AgentActivityLogger
from agents.prompt_logger import PromptResponseLogger

# ── Page config (must be the first Streamlit call) ────────────────────────────
st.set_page_config(
    page_title="GenAI ETL Data Curation",
    page_icon="⚙️",
    layout="wide",
)


# ── Session state ─────────────────────────────────────────────────────────────

_STATE_DEFAULTS = {
    "phase": "idle",           # see phase state machine below
    "attempt": 0,              # current developer/executor/tester iteration
    "config": None,            # sidebar config captured at run start
    "mapping": None,           # dict from analyst
    "requirements_text": None, # raw requirements text (needed for developer retry)
    "pipeline_path": None,     # str path to generated pipeline
    "processed_data_path": None,
    "executor_error": None,    # str error message from executor failure
    "executor_succeeded": False, # True only after executor runs successfully in this run
    "test_summary": None,      # dict from tester
    "test_feedback": None,     # dict passed to developer on retry
}

# Phase state machine:
#
#  idle
#   │ [Start Pipeline]
#   ▼
#  running_analyst ──► analyst_done
#                            │ [Continue to Developer →]
#                            ▼
#                      running_developer ──► developer_done
#                                                │ [Continue to Executor →]
#                                                ▼
#                                          running_executor ──► executor_done
#                                          │  (failure)               │ [Continue to Tester →]
#                                          ▼                          ▼
#                                     executor_failed          running_tester ──► tester_done
#                                          │                          │ (failure, retries left)
#                                          │ [Retry]                  ▼
#                                          └────────────────── tester_failed
#                                                                     │ [Retry with feedback]
#                                                                     └──► running_developer


def _init_session_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default
    if "logger" not in st.session_state:
        st.session_state.logger = AgentActivityLogger()


def _reset_state() -> None:
    for key, default in _STATE_DEFAULTS.items():
        st.session_state[key] = default


# ── Sidebar ───────────────────────────────────────────────────────────────────

def _render_sidebar() -> tuple[dict, bool]:
    """Render configuration sidebar. Returns (config_dict, run_clicked)."""
    with st.sidebar:
        st.header("Configuration")

        is_idle = st.session_state.phase == "idle"
        is_running = st.session_state.phase.startswith("running_")

        req_path = st.text_input(
            "Requirements doc",
            "data/raw/requirements_document.txt",
            disabled=not is_idle,
        )
        csv_path = st.text_input(
            "Source CSV",
            "data/raw/sample_transactions.csv",
            disabled=not is_idle,
        )
        framework = st.selectbox(
            "Framework", ["pandas", "pyspark"], disabled=not is_idle
        )
        validate = st.checkbox(
            "Test-driven retry", value=True, disabled=not is_idle
        )
        max_retries = st.slider(
            "Max retries", 1, 5, 3, disabled=not is_idle or not validate
        )
        skip_analyst = st.checkbox(
            "Skip analyst (use cached mapping)", disabled=not is_idle
        )

        st.markdown("**Model selection**")
        _analyst_models = ["gpt-4o-mini", "gpt-4o", "gpt-4.1"]
        _code_models = ["o4-mini", "gpt-4.1", "gpt-4o", "gpt-4o-mini"]
        analyst_model = st.selectbox(
            "Analyst model", _analyst_models, index=0, disabled=not is_idle
        )
        developer_model = st.selectbox(
            "Developer model", _code_models, index=0, disabled=not is_idle
        )
        tester_model = st.selectbox(
            "Tester model", _code_models, index=0, disabled=not is_idle
        )

        st.divider()

        run_clicked = False
        if is_idle:
            run_clicked = st.button(
                "Start Pipeline", type="primary", use_container_width=True
            )
        elif is_running:
            st.button("Running...", disabled=True, use_container_width=True)
        else:
            if st.button("Reset / New Run", use_container_width=True):
                _reset_state()
                st.rerun()

        # Status summary from last completed run
        phase = st.session_state.phase
        if phase == "tester_done":
            st.success("All tests passed")
        elif phase == "tester_failed":
            summary = st.session_state.test_summary or {}
            n_fail = sum(1 for t in summary.get("test_results", []) if not t.get("passed"))
            n_total = len(summary.get("test_results", []))
            st.error(f"{n_fail}/{n_total} tests failed")
        elif phase == "executor_failed":
            st.error("Pipeline execution failed")

        config = {
            "req_path": req_path,
            "csv_path": csv_path,
            "framework": framework,
            "validate": validate,
            "max_retries": max_retries if validate else 1,
            "skip_analyst": skip_analyst,
            "analyst_model": analyst_model,
            "developer_model": developer_model,
            "tester_model": tester_model,
        }

    return config, run_clicked


# ── Flow diagram ─────────────────────────────────────────────────────────────

_NODE_COLORS: dict[str, tuple[str, str]] = {
    "pending": ("#F5F5F5", "#BDBDBD"),
    "running": ("#FFF9C4", "#F9A825"),
    "active":  ("#E3F2FD", "#1976D2"),
    "done":    ("#E8F5E9", "#388E3C"),
    "failed":  ("#FFEBEE", "#C62828"),
}


def _get_node_states(phase: str) -> dict[str, str]:
    """Map current phase to a visual state for each of the 4 agent nodes."""
    s = {"analyst": "pending", "developer": "pending", "executor": "pending", "tester": "pending"}
    if phase in ("running_analyst",):
        s["analyst"] = "running"
    elif phase in ("analyst_done",):
        s["analyst"] = "active"
    elif phase in ("running_developer",):
        s["analyst"] = "done"; s["developer"] = "running"
    elif phase in ("developer_done",):
        s["analyst"] = "done"; s["developer"] = "active"
    elif phase in ("running_executor",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "running"
    elif phase in ("executor_done",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "active"
    elif phase in ("executor_failed",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "failed"
    elif phase in ("running_tester",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "done"; s["tester"] = "running"
    elif phase in ("tester_done",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "done"; s["tester"] = "done"
    elif phase in ("tester_failed",):
        s["analyst"] = "done"; s["developer"] = "done"; s["executor"] = "done"; s["tester"] = "failed"
    return s


def _render_flow_diagram() -> None:
    phase = st.session_state.phase
    attempt = st.session_state.attempt
    test_feedback = st.session_state.test_feedback

    ns = _get_node_states(phase)

    def node(state: str, label: str) -> str:
        fill, border = _NODE_COLORS[state]
        return f'fillcolor="{fill}" color="{border}" penwidth=2 label="{label}"'

    # Feedback edge highlight rules
    tester_retry_active = phase == "tester_failed" or (
        phase == "running_developer" and attempt > 1 and test_feedback is not None
    )
    executor_retry_active = phase == "executor_failed"

    tester_fb = 'color="#C62828" penwidth=2 style=dashed fontcolor="#C62828"' if tester_retry_active else 'color="#BDBDBD" penwidth=1 style=dashed fontcolor="#BDBDBD"'
    executor_fb = 'color="#E65100" penwidth=2 style=dashed fontcolor="#E65100"' if executor_retry_active else 'color="#BDBDBD" penwidth=1 style=dashed fontcolor="#BDBDBD"'

    dot = f"""
digraph etl_pipeline {{
    rankdir=LR
    splines=curved
    graph [pad="0.3" ranksep="1.0" nodesep="0.4" bgcolor="transparent"]
    node [shape=box style="filled,rounded" fontname="Helvetica" fontsize=11 width=2.0 height=0.85]
    edge [fontname="Helvetica" fontsize=9]

    analyst   [{node(ns["analyst"],  "Analyst\\nParse Requirements")}]
    developer [{node(ns["developer"], "Developer\\nGenerate Pipeline")}]
    executor  [{node(ns["executor"],  "Code Execution\\nRun Pipeline")}]
    tester    [{node(ns["tester"],    "Tester\\nValidate Output")}]

    analyst   -> developer [label=" mapping.json" color="#9E9E9E" fontcolor="#9E9E9E"]
    developer -> executor  [label=" pipeline.py"  color="#9E9E9E" fontcolor="#9E9E9E"]
    executor  -> tester    [label=" output.csv"   color="#9E9E9E" fontcolor="#9E9E9E"]

    tester   -> developer [label=" test feedback (retry)"  {tester_fb}  constraint=false tailport=s headport=s]
    executor -> developer [label=" exec error (retry)"     {executor_fb} constraint=false tailport=s headport=s]
}}
"""
    st.graphviz_chart(dot, use_container_width=True)
    st.caption(
        "⬜ Pending &nbsp; 🟡 Running &nbsp; 🔵 Waiting for review &nbsp; 🟢 Done &nbsp; 🔴 Failed",
        unsafe_allow_html=True,
    )


# ── Artifact sections ─────────────────────────────────────────────────────────

def _render_analyst_section() -> None:
    if st.session_state.mapping is None:
        return

    mapping = st.session_state.mapping
    target = mapping.get("target_table", "—")
    n_rules = len(mapping.get("source_to_target_mapping", []))

    st.subheader("Analyst — Source-to-Target Mapping")
    st.caption(f"Target table: `{target}` · {n_rules} transformation rules")

    # Raw source data preview
    cfg = st.session_state.config or {}
    csv_path = cfg.get("csv_path", "")
    if csv_path and Path(csv_path).exists():
        src_df = pd.read_csv(csv_path)
        with st.expander(
            f"Source data preview — {len(src_df):,} rows · {len(src_df.columns)} columns",
            expanded=False,
        ):
            st.dataframe(src_df.head(10), use_container_width=True)

    # Source-to-target mapping table
    mappings = mapping.get("source_to_target_mapping", [])
    if mappings:
        st.markdown("**Source-to-target mapping**")
        mapping_df = pd.DataFrame(mappings)
        display_cols = [c for c in ["source_column", "target_column", "target_type", "mapping_type", "transformation"] if c in mapping_df.columns]
        mapping_df = mapping_df[display_cols].rename(columns={
            "source_column": "Source Column",
            "target_column": "Target Column",
            "target_type": "Target Type",
            "mapping_type": "Mapping Type",
            "transformation": "Transformation",
        })
        st.dataframe(mapping_df, hide_index=True, use_container_width=True)

    with st.expander("Full mapping JSON", expanded=False):
        st.json(mapping)

    st.divider()


def _render_developer_section() -> None:
    if st.session_state.pipeline_path is None:
        return

    pipeline_path = st.session_state.pipeline_path
    attempt = st.session_state.attempt

    st.subheader("Developer — Generated Pipeline Code")
    st.caption(f"Attempt {attempt} · `{pipeline_path}`")

    if Path(pipeline_path).exists():
        code = Path(pipeline_path).read_text(encoding="utf-8")
        line_count = code.count("\n")
        with st.expander(f"Pipeline code ({line_count} lines)", expanded=True):
            st.code(code, language="python")
    else:
        st.warning(f"Pipeline file not found: {pipeline_path}")

    st.divider()


def _render_executor_section() -> None:
    if st.session_state.executor_error:
        st.subheader("Executor — Pipeline Execution Failed")
        st.error(st.session_state.executor_error[:600])
        st.divider()
        return

    # Only show output after the executor has actually run in this session
    if not st.session_state.executor_succeeded:
        return

    processed_path = st.session_state.processed_data_path
    if not processed_path or not Path(processed_path).exists():
        st.warning(f"Executor ran but output file not found: {processed_path}")
        return

    st.subheader("Executor — Pipeline Output")
    st.caption(f"`{processed_path}`")

    df = pd.read_csv(processed_path)
    st.metric("Total rows", f"{len(df):,}")

    with st.expander("Output data preview — first 10 rows", expanded=True):
        st.dataframe(df.head(10), use_container_width=True)

    st.divider()


def _render_tester_section() -> None:
    if st.session_state.test_summary is None:
        return

    summary = st.session_state.test_summary
    results = summary.get("test_results", [])
    attempt = st.session_state.attempt
    passed = sum(1 for r in results if r.get("passed"))
    total = len(results)

    st.subheader("Tester — Processed Data Testing Results")
    st.caption(f"Attempt {attempt} · {passed}/{total} tests passed")

    if results:
        df = pd.DataFrame(results)[["name", "passed", "details"]]
        df = df.rename(columns={"name": "Test", "passed": "Status", "details": "Details"})
        df["Test"] = df["Test"].map(_friendly_test_name)
        df["Status"] = df["Status"].map({True: "PASS", False: "FAIL"})

        def _color_status(val):
            return "color: green; font-weight: bold" if val == "PASS" else "color: red; font-weight: bold"

        st.dataframe(
            df.style.map(_color_status, subset=["Status"]),
            hide_index=True,
            use_container_width=True,
        )

    st.divider()


# ── Phase runner ──────────────────────────────────────────────────────────────

def _run_phase(config: dict) -> None:
    """Execute the current running_* phase. Updates session state and reruns."""
    phase = st.session_state.phase
    if not phase.startswith("running_"):
        return

    logger: AgentActivityLogger = st.session_state.logger

    if phase == "running_analyst":
        with st.spinner("Analyst is running — parsing requirements into mapping..."):
            if config["skip_analyst"]:
                mapping_path = Path("outputs/source_to_target_mapping.json")
                if not mapping_path.exists():
                    st.error(
                        "Cached mapping not found at outputs/source_to_target_mapping.json. "
                        "Uncheck 'Skip analyst' to generate it first."
                    )
                    st.session_state.phase = "idle"
                    st.stop()
                mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
            else:
                analysis = analyze_requirements_and_data(
                    requirements_path=config["req_path"],
                    output_mapping_path="outputs/source_to_target_mapping.json",
                    model=config["analyst_model"],
                )
                mapping = analysis["mapping"]

            requirements_text = Path(config["req_path"]).read_text(encoding="utf-8")

        target_table = mapping.get("target_table", "output_table")
        st.session_state.mapping = mapping
        st.session_state.requirements_text = requirements_text
        st.session_state.processed_data_path = f"data/processed/{target_table}.csv"
        st.session_state.phase = "analyst_done"
        logger.log_event(
            agent="analyst",
            stage="completed",
            output_paths={"mapping": "outputs/source_to_target_mapping.json"},
        )
        st.rerun()

    elif phase == "running_developer":
        attempt = st.session_state.attempt
        with st.spinner(f"Developer is running — generating pipeline code (attempt {attempt})..."):
            pipeline_path = generate_pipeline_code(
                mapping=st.session_state.mapping,
                output_path="outputs/generated_pipeline.py",
                framework=config["framework"],
                data_dictionary=None,
                test_feedback=st.session_state.test_feedback,
                requirements_text=st.session_state.requirements_text,
                model=config["developer_model"],
            )

        st.session_state.pipeline_path = pipeline_path
        st.session_state.phase = "developer_done"
        logger.log_event(
            agent="developer",
            stage="completed",
            iteration=attempt,
            output_paths={"pipeline": pipeline_path},
        )
        st.rerun()

    elif phase == "running_executor":
        with st.spinner("Executor is running — executing generated pipeline..."):
            try:
                _run_pipeline_subprocess(st.session_state.pipeline_path)
                st.session_state.executor_error = None
                st.session_state.executor_succeeded = True
                st.session_state.phase = "executor_done"
                logger.log_event(agent="executor", stage="completed", status="success")
            except RuntimeError as exc:
                err = str(exc)[:600]
                st.session_state.executor_error = err
                st.session_state.phase = "executor_failed"
                logger.log_event(
                    agent="executor", stage="completed", status="failure", details=err
                )
        st.rerun()

    elif phase == "running_tester":
        with st.spinner("Tester is running — validating output data quality..."):
            summary = run_tests(
                requirements_path=config["req_path"],
                source_data_path=config["csv_path"],
                processed_data_path=st.session_state.processed_data_path,
                report_path="data/processed/test_report.txt",
                model=config["tester_model"],
            )

        st.session_state.test_summary = summary
        failed = [t for t in summary.get("test_results", []) if not t.get("passed")]
        if summary.get("all_passed"):
            st.session_state.phase = "tester_done"
        else:
            st.session_state.test_feedback = {
                "test_results": failed,
                "generated_code": Path(st.session_state.pipeline_path).read_text(
                    encoding="utf-8"
                ),
            }
            st.session_state.phase = "tester_failed"
        logger.log_event(
            agent="tester",
            stage="completed",
            status="success" if summary.get("all_passed") else "failure",
            iteration=st.session_state.attempt,
        )
        st.rerun()


# ── Action buttons ────────────────────────────────────────────────────────────

def _render_action_buttons(config: dict) -> None:
    """Render Continue / Retry / Stop buttons appropriate to the current phase."""
    phase = st.session_state.phase
    attempt = st.session_state.attempt
    max_retries = config["max_retries"]

    def _stop():
        if st.button("Stop", use_container_width=True):
            _reset_state()
            st.rerun()

    if phase == "analyst_done":
        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button("Continue to Developer →", type="primary", use_container_width=True):
                st.session_state.attempt = 1
                st.session_state.phase = "running_developer"
                st.rerun()
        with c2:
            _stop()

    elif phase == "developer_done":
        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button("Continue to Executor →", type="primary", use_container_width=True):
                st.session_state.phase = "running_executor"
                st.rerun()
        with c2:
            _stop()

    elif phase == "executor_done":
        c1, c2 = st.columns([4, 1])
        with c1:
            if st.button("Continue to Tester →", type="primary", use_container_width=True):
                st.session_state.phase = "running_tester"
                st.rerun()
        with c2:
            _stop()

    elif phase in ("executor_failed", "tester_failed"):
        can_retry = config["validate"] and attempt < max_retries
        if can_retry:
            c1, c2 = st.columns([4, 1])
            with c1:
                label = f"Retry — regenerate pipeline (attempt {attempt + 1}/{max_retries})"
                if st.button(label, type="primary", use_container_width=True):
                    if phase == "executor_failed":
                        # Wrap execution error as test feedback for the developer
                        st.session_state.test_feedback = {
                            "test_results": [
                                {
                                    "name": "execution_error",
                                    "passed": False,
                                    "details": st.session_state.executor_error or "",
                                }
                            ],
                            "generated_code": Path(
                                st.session_state.pipeline_path
                            ).read_text(encoding="utf-8"),
                        }
                        st.session_state.executor_error = None
                    st.session_state.attempt += 1
                    st.session_state.test_summary = None
                    st.session_state.executor_succeeded = False
                    st.session_state.phase = "running_developer"
                    st.rerun()
            with c2:
                _stop()
        else:
            msg = (
                "No retries remaining."
                if config["validate"]
                else "Enable 'Test-driven retry' to allow reruns."
            )
            st.info(msg)
            _stop()

    elif phase == "tester_done":
        if st.button("Reset / New Run"):
            _reset_state()
            st.rerun()


# ── Test name display labels ──────────────────────────────────────────────────

_TEST_NAME_LABELS = {
    "expected_columns_present":         "Expected Columns Present",
    "rename_mapping_applied":           "Column Rename Applied",
    "timestamp_conversion_logic":       "Timestamp Conversion",
    "merchant_country_standardization": "Country Code Standardization",
    "merchant_name_blank_handling":     "Merchant Name Blank Handling",
    "type_standardization_checks":      "Type Standardization",
    "output_schema_rowcount_sanity":    "Row Count Sanity",
    "requirements_path_exists":         "Requirements File Found",
    "source_data_path_exists":          "Source Data File Found",
    "processed_data_path_exists":       "Processed Data File Found",
    "generated_tests_runtime":          "Test Execution Error",
}


def _friendly_test_name(name: str) -> str:
    return _TEST_NAME_LABELS.get(name, name.replace("_", " ").title())


# ── LLM Prompts & Responses tab ───────────────────────────────────────────────

_CALL_TYPE_LABELS = {
    "requirements_parsing": "Requirements Parsing",
    "data_profiling": "Data Profiling",
    "code_generation": "Code Generation",
    "test_generation": "Test Generation",
}

_RESPONSE_LANGUAGES = {
    "requirements_parsing": "json",
    "data_profiling": "json",
    "code_generation": "python",
    "test_generation": "python",
}

_AGENT_LABELS = {
    "analyst": "Analyst Agent",
    "developer": "Developer Agent",
    "tester": "Tester Agent",
}

_AGENT_ORDER = ["analyst", "developer", "tester"]


def _render_prompt_tab() -> None:
    entries = PromptResponseLogger.load()
    if not entries:
        st.info("Run the pipeline to see LLM prompts and responses here.")
        return

    by_agent: dict[str, list[dict]] = {a: [] for a in _AGENT_ORDER}
    for entry in entries:
        agent = entry.get("agent", "")
        if agent in by_agent:
            by_agent[agent].append(entry)

    for agent in _AGENT_ORDER:
        agent_entries = by_agent[agent]
        if not agent_entries:
            continue

        st.subheader(_AGENT_LABELS.get(agent, agent.title()))

        for i, entry in enumerate(agent_entries):
            call_type = entry.get("call_type", "unknown")
            label = _CALL_TYPE_LABELS.get(call_type, call_type.replace("_", " ").title())
            if len(agent_entries) > 1:
                label = f"{label} — Attempt {i + 1}"

            st.markdown(f"**{label}**")
            ts = entry.get("timestamp", "")
            if ts:
                st.caption(f"Called at {ts}")

            with st.expander("Prompt", expanded=False):
                st.markdown("**System:**")
                st.code(entry.get("system_prompt", ""), language="text")
                st.markdown("**Human:**")
                st.code(entry.get("human_prompt", ""), language="text")

            lang = _RESPONSE_LANGUAGES.get(call_type, "text")
            with st.expander("Raw Response", expanded=False):
                st.code(entry.get("raw_response", ""), language=lang)

            if i < len(agent_entries) - 1:
                st.markdown("---")

        st.divider()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    _init_session_state()
    config, run_clicked = _render_sidebar()

    st.title("⚙️ GenAI ETL Data Curation")
    st.caption("Step-by-step pipeline with human review at each stage.")

    if run_clicked:
        _reset_state()
        PromptResponseLogger().clear()
        st.session_state.config = config
        st.session_state.phase = "running_analyst"
        st.rerun()

    # Use the config captured at run start for the rest of the run
    active_config = st.session_state.config or config

    _render_flow_diagram()

    st.divider()
    tab_pipeline, tab_prompts = st.tabs(["Pipeline", "LLM Prompts & Responses"])

    with tab_pipeline:
        _render_analyst_section()
        _render_developer_section()
        _render_tester_section()
        _render_executor_section()

        # Execute the current running phase (triggers st.rerun() internally)
        _run_phase(active_config)

        # Render action buttons for non-running paused phases
        phase = st.session_state.phase
        if phase == "idle":
            st.info(
                "Configure the pipeline in the sidebar, then click **Start Pipeline** to begin."
            )
        elif not phase.startswith("running_"):
            _render_action_buttons(active_config)

    with tab_prompts:
        _render_prompt_tab()


if __name__ == "__main__":
    main()
