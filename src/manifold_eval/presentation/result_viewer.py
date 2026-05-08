"""Streamlit shell for comparing saved evaluation results.

Run from the project root:

    manifold-eval-viewer --results-dir output

For now the screen is split into two sections. Each section can select one
model folder from the first level under the configured results directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import streamlit as st


OUTPUT_DIR = Path()


def main() -> None:
    """Render the two-panel model selection screen."""
    global OUTPUT_DIR
    OUTPUT_DIR = parse_results_dir()

    st.set_page_config(page_title="Eval Comparison", layout="wide")
    render_app_styles()

    model_names = get_model_folder_names()

    st.markdown(
        '<section class="app-bar"><h1 class="app-title">Eval Comparison</h1>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2)
    with left:
        first_model = render_model_panel("first", model_names)

    with right:
        second_model = render_model_panel("second", model_names)
    st.markdown("</section>", unsafe_allow_html=True)

    benchmarks = get_matching_benchmark_names(first_model, second_model)
    if benchmarks:
        st.divider()
        st.markdown("**Benchmarks to Compare**")
        selected_benchmarks = render_benchmark_checkboxes(benchmarks)
        if selected_benchmarks:
            st.divider()
            render_selected_benchmark_scores(
                [model for model in (first_model, second_model) if model],
                selected_benchmarks,
            )


def render_app_styles() -> None:
    """Render CSS for the top app bar."""
    st.markdown(
        """
        <style>
        .block-container {
            padding-top: 0;
        }
        .app-bar {
            position: sticky;
            top: 0;
            z-index: 100;
            background: rgb(255, 255, 255);
            border-bottom: 1px solid rgba(49, 51, 63, 0.18);
            padding: 1rem 0 1.1rem 0;
            margin: 0 0 1rem 0;
        }
        .app-title {
            text-align: center;
            margin: 0 0 0.85rem 0;
            font-size: 1.6rem;
            font-weight: 650;
            line-height: 1.2;
        }
        .benchmark-title {
            text-align: left;
            margin: 0.15rem 0 0.9rem 0;
            font-size: 1.35rem;
            font-weight: 650;
            line-height: 1.25;
        }
        [data-testid="stVerticalBlockBorderWrapper"] {
            background: #f8fafc;
            border-color: rgba(49, 51, 63, 0.16);
            box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_model_folder_names() -> list[str]:
    """Return first-level model folders under the output directory."""
    if not OUTPUT_DIR.exists():
        return []

    return sorted(path.name for path in OUTPUT_DIR.iterdir() if path.is_dir())


def render_model_panel(label: str, model_names: list[str]) -> None:
    """Render one model selection section."""
    if not model_names:
        st.info("No model folders found under output/.")
        return None

    selected = st.selectbox(
        "Model",
        ["<select model>", *model_names],
        key=f"{label.lower()}_model",
        label_visibility="collapsed",
    )

    if selected == "<select model>":
        return None

    return selected


def get_matching_benchmark_names(
    first_model: str | None,
    second_model: str | None,
) -> list[str]:
    """Return benchmarks for selected model folders.

    One selected model returns all of its benchmarks. Two selected models return
    only benchmarks that exist for both models.
    """
    selected_models = [
        model for model in (first_model, second_model) if model is not None
    ]
    if not selected_models:
        return []

    benchmark_sets = [
        set(get_benchmark_folder_names(model)) for model in selected_models
    ]

    matching = set.intersection(*benchmark_sets)
    return sorted(matching)


def get_benchmark_folder_names(model_name: str) -> list[str]:
    """Return benchmark folders under one model output folder."""
    model_dir = OUTPUT_DIR / model_name
    if not model_dir.exists():
        return []

    return sorted(path.name for path in model_dir.iterdir() if path.is_dir())


def render_benchmark_checkboxes(benchmarks: list[str]) -> list[str]:
    """Render benchmark checkboxes in the row below model selectors."""
    selected = []
    columns = st.columns(min(len(benchmarks), 4))
    for index, benchmark in enumerate(benchmarks):
        with columns[index % len(columns)]:
            checked = st.checkbox(
                benchmark,
                key=f"benchmark_{benchmark}",
            )
            if checked:
                selected.append(benchmark)

    return selected


def render_selected_benchmark_scores(
    model_names: list[str],
    benchmark_names: list[str],
) -> None:
    """Render latest score summaries for selected model/benchmark pairs."""
    for benchmark_name in benchmark_names:
        with st.container(border=True):
            st.markdown(
                f'<h2 class="benchmark-title">{benchmark_name}</h2>',
                unsafe_allow_html=True,
            )

            selected_files = {}
            columns = st.columns(len(model_names))
            for index, model_name in enumerate(model_names):
                with columns[index]:
                    result_file = select_result_file(model_name, benchmark_name)
                    selected_files[model_name] = result_file
                    render_model_benchmark_score(
                        model_name,
                        benchmark_name,
                        result_file,
                        show_stderr=False,
                    )
                    render_model_correct_count(result_file)

            show_stderr = st.checkbox(
                "show stderr",
                value=False,
                key=f"show_stderr_{benchmark_name}",
            )

            if show_stderr:
                render_stderr_scores(model_names, benchmark_name, selected_files)

            render_question_differences(model_names, benchmark_name, selected_files)


def select_result_file(model_name: str, benchmark_name: str) -> Path | None:
    """Render one model-side run selector and return its result file."""
    selected = st.selectbox(
        "Run",
        get_result_selection_options(model_name, benchmark_name),
        key=f"result_selector_{model_name}_{benchmark_name}",
    )

    return resolve_result_selection(model_name, benchmark_name, selected)


def get_result_selection_options(
    model_name: str,
    benchmark_name: str,
) -> list[str]:
    """Return run selection options for one model/benchmark side."""
    labels = {
        format_result_file_label(result_file)
        for result_file in get_result_files(model_name, benchmark_name)
    }

    return ["latest", "best", *sorted(labels)]


def render_model_benchmark_score(
    model_name: str,
    benchmark_name: str,
    result_file: Path | None,
    *,
    show_stderr: bool,
) -> None:
    """Render the latest saved score for one model and benchmark."""
    if result_file is None:
        st.caption(model_name)
        st.info("No result file found.")
        return

    result = load_json(result_file)
    st.caption(format_result_metadata(model_name, result_file, result))

    scores = result.get("scores", {}).get(benchmark_name, {})
    if not scores:
        st.info("No scores found.")
        return

    for metric_name, value in scores.items():
        if metric_name == "alias":
            continue

        if not show_stderr and "_stderr" in metric_name:
            continue

        st.metric(format_metric_name(metric_name), format_score_value(value))


def render_model_correct_count(result_file: Path | None) -> None:
    """Render total right count for one model/benchmark latest result."""
    if result_file is None:
        return

    result = load_json(result_file)
    examples = result.get("examples", [])
    total = len(examples)
    correct = sum(1 for example in examples if is_example_correct(example))

    st.metric("questions right", f"{correct}/{total}")


def format_result_metadata(
    model_name: str,
    result_file: Path,
    result: dict[str, Any],
) -> str:
    """Return compact metadata for one selected result."""
    parts = [model_name, format_result_file_label(result_file)]
    parameters = result.get("benchmark", {}).get("parameters", {})
    if "limit" in parameters:
        parts.append(f"limit:{parameters['limit']}")

    return " • ".join(parts)


def render_stderr_scores(
    model_names: list[str],
    benchmark_name: str,
    selected_files: dict[str, Path | None],
) -> None:
    """Render stderr metrics below the main scores when requested."""
    st.markdown("**stderr**")
    columns = st.columns(len(model_names))
    for index, model_name in enumerate(model_names):
        with columns[index]:
            result_file = selected_files.get(model_name)
            if result_file is None:
                continue

            result = load_json(result_file)
            scores = result.get("scores", {}).get(benchmark_name, {})
            for metric_name, value in scores.items():
                if "_stderr" in metric_name:
                    st.metric(format_metric_name(metric_name), format_score_value(value))


def format_metric_name(metric_name: str) -> str:
    """Return a readable metric label."""
    label = metric_name.replace("_stderr", " stderr")
    label = label.replace("_", " ")
    label = label.replace(",", " / ")
    return label.strip()


def format_score_value(value: Any) -> Any:
    """Format score-like values for display."""
    if isinstance(value, bool):
        return value

    if isinstance(value, (int, float)) and 0 <= value <= 1:
        return f"{value * 100:.1f}%"

    return value


def render_question_differences(
    model_names: list[str],
    benchmark_name: str,
    selected_files: dict[str, Path | None],
) -> None:
    """Render questions where two models disagree on correctness."""
    if len(model_names) != 2:
        return

    first_model, second_model = model_names
    first_file = selected_files.get(first_model)
    second_file = selected_files.get(second_model)
    if first_file is None or second_file is None:
        return

    first_result = load_json(first_file)
    second_result = load_json(second_file)
    first_examples = examples_by_doc_id(first_result)
    second_examples = examples_by_doc_id(second_result)
    common_doc_ids = sorted(set(first_examples) & set(second_examples))

    first_right = []
    second_right = []
    for doc_id in common_doc_ids:
        first_example = first_examples[doc_id]
        second_example = second_examples[doc_id]
        first_correct = is_example_correct(first_example)
        second_correct = is_example_correct(second_example)

        if first_correct and not second_correct:
            first_right.append((doc_id, first_example, second_example))
        elif second_correct and not first_correct:
            second_right.append((doc_id, first_example, second_example))

    st.divider()
    st.markdown("**Question Differences**")

    left, right = st.columns(2)
    with left:
        render_difference_group(
            first_model,
            first_right,
            first_model,
            second_model,
        )

    with right:
        render_difference_group(
            second_model,
            second_right,
            first_model,
            second_model,
        )


def examples_by_doc_id(result: dict[str, Any]) -> dict[Any, dict[str, Any]]:
    """Index saved examples by doc id."""
    return {
        example.get("doc_id"): example
        for example in result.get("examples", [])
    }


def is_example_correct(example: dict[str, Any]) -> bool:
    """Return whether any per-question metric marks this example as correct."""
    for score in example.get("scores", []):
        for value in score.get("metrics", {}).values():
            if value is True or value == 1:
                return True

    return False


def render_difference_group(
    improved_model: str,
    differences: list[tuple[Any, dict[str, Any], dict[str, Any]]],
    first_model: str,
    second_model: str,
) -> None:
    """Render one side of the correctness-difference comparison."""
    st.markdown(f"**{improved_model}** answered better than the other model: {len(differences)}")
    if not differences:
        st.info("No differences found.")
        return

    for doc_id, first_example, second_example in differences:
        with st.expander(f"doc {doc_id}", expanded=False):
            st.markdown("**Question**")
            st.write(first_example.get("question", ""))

            st.markdown(f"**{first_model} answer**")
            st.write(first_example.get("model_answer", ""))
            render_metric_bullets(first_example)

            st.markdown(f"**{second_model} answer**")
            st.write(second_example.get("model_answer", ""))
            render_metric_bullets(second_example)


def render_metric_bullets(example: dict[str, Any]) -> None:
    """Render per-filter metrics as simple bullet lines."""
    for score in example.get("scores", []):
        filter_name = score.get("filter", "unknown filter")
        for metric_name, value in score.get("metrics", {}).items():
            st.markdown(
                f"- {filter_name}: {format_metric_name(metric_name)} = "
                f"{format_score_value(value)}"
            )


def get_latest_result_file(model_name: str, benchmark_name: str) -> Path | None:
    """Return newest JSON result for one model/benchmark folder."""
    result_files = get_result_files(model_name, benchmark_name)
    if not result_files:
        return None

    return max(result_files, key=lambda path: path.stat().st_mtime)


def get_result_files(model_name: str, benchmark_name: str) -> list[Path]:
    """Return saved JSON files for one model/benchmark folder."""
    benchmark_dir = OUTPUT_DIR / model_name / benchmark_name
    if not benchmark_dir.exists():
        return []

    return sorted(benchmark_dir.glob("*.json"), key=lambda path: path.stat().st_mtime)


def resolve_result_selection(
    model_name: str,
    benchmark_name: str,
    selection: str,
) -> Path | None:
    """Resolve a run selector value to a concrete result file."""
    if selection == "latest":
        return get_latest_result_file(model_name, benchmark_name)

    if selection == "best":
        return get_best_result_file(model_name, benchmark_name)

    for result_file in get_result_files(model_name, benchmark_name):
        if format_result_file_label(result_file) == selection:
            return result_file

    return None


def get_best_result_file(model_name: str, benchmark_name: str) -> Path | None:
    """Return the result file with the highest aggregate non-stderr score."""
    result_files = get_result_files(model_name, benchmark_name)
    if not result_files:
        return None

    return max(result_files, key=lambda path: get_best_score_value(load_json(path)))


def get_best_score_value(result: dict[str, Any]) -> float:
    """Return the highest numeric aggregate score in a result."""
    best = float("-inf")
    for benchmark_scores in result.get("scores", {}).values():
        for metric_name, value in benchmark_scores.items():
            if (
                metric_name == "alias"
                or "_stderr" in metric_name
                or not isinstance(value, (int, float))
            ):
                continue
            best = max(best, float(value))

    return best


def format_result_file_label(path: Path) -> str:
    """Format a timestamp result filename for display."""
    timestamp = path.stem
    parts = timestamp.split("_", maxsplit=1)
    if len(parts) != 2:
        return timestamp

    return f"{parts[0]} {parts[1].replace('-', ':')}"


def load_json(path: Path) -> dict[str, Any]:
    """Load a saved JSON result file."""
    return json.loads(path.read_text(encoding="utf-8"))


def parse_results_dir() -> Path:
    """Return the results directory passed through Streamlit CLI args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--results-dir", default="output")
    args, _unknown = parser.parse_known_args()
    return Path(args.results_dir).expanduser().resolve()


if __name__ == "__main__":
    main()
