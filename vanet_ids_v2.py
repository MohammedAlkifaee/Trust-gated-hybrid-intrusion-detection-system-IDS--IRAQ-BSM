#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shlex
import shutil
import signal
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from vanet_ids_rsu_core import (
    ARTIFACT_FAMILY,
    ATTACK_FAMILIES,
    ATTACK_TYPES,
    DEFAULT_SEQ_LEN,
    DEFAULT_WINDOW_SIZE,
    HEAD_VECTOR_ORDER,
    OBU_FLAG_COLUMNS,
    RSUMultiHeadTrainer,
    list_historical_results,
    list_model_directories,
    load_raw_bsm_directory,
    load_runtime,
    normalize_bsm_dataframe,
    train_scenario_lgbm_release,
)


ascii_art = r"""
****************************************************************************************************
*  _   _ _   _ _____     _______ ____  ____ ___ _______   __   ___  _____   _  ___   _ _____ _     *
* | | | | \ | |_ _\ \   / / ____|  _ \/ ___|_ _|_   _\ \ / /  / _ \|  ___| | |/ / | | |  ___/ \    *
* | | | |  \| || | \ \ / /|  _| | |_) \___ \| |  | |  \ V /  | | | | |_    | ' /| | | | |_ / _ \   *
* | |_| | |\  || |  \ V / | |___|  _ < ___) | |  | |   | |   | |_| |  _|   | . \| |_| |  _/ ___ \  *
*  \___/|_| \_|___|  \_/  |_____|_| \_\____/___| |_|   |_|    \___/|_|     |_|\_\\___/|_|/_/   \_\ *
****************************************************************************************************
"""


DEFAULT_F2MD_DIR = "/home/instantf2md/F2MD"
DEFAULT_RESULTS_DIR = "/home/instantf2md/F2MD/f2md-results"
DEFAULT_EXTRACT_PY = os.path.join(DEFAULT_F2MD_DIR, "extract1_intermsg.py")
DEFAULT_ATTACK_ID_PY = os.path.join(os.path.dirname(__file__), "add_attack_id.py")
DEFAULT_LIVE_IDS_DIR = "/home/instantf2md/Desktop/VANET-IRAQ Live IDS Dashboard"
DEFAULT_RSU_TRAINER_PY = os.path.join(os.path.dirname(__file__), "rsu_trainer_all_in_one_v7.py")
DEFAULT_BAGHDAD_OMNETPP = (
    "/home/instantf2md/F2MD/veins-f2md/f2md-networks/BaghdadScenario/omnetpp.ini"
)
DEFAULT_MODELS_DIR = os.path.join(os.path.dirname(__file__), "release_v3")
DEFAULT_ARCHIVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "results and models", "models")
DEFAULT_DETECT_MODELS_DIR = DEFAULT_ARCHIVED_MODELS_DIR if os.path.isdir(DEFAULT_ARCHIVED_MODELS_DIR) else DEFAULT_MODELS_DIR
DEFAULT_ARCHIVE_ROOT = os.path.join(os.path.dirname(__file__), "results and models")
DEFAULT_CONDA_ENV = os.environ.get("CONDA_DEFAULT_ENV", "base")
DEFAULT_CLI_RUNTIME_DIR = os.path.join(os.path.dirname(__file__), "logs", "cli_runtime")
DEFAULT_APP_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_ATTACK_TYPE_KEY = "*.node[*].appl.LOCAL_ATTACK_TYPE"
LIVE_IDS_REQUIRED_IMPORTS = ("PyQt5", "sklearn", "joblib", "numpy", "pandas", "pyqtgraph")
TRAINER_GUI_REQUIRED_IMPORTS = ("PyQt5", "joblib", "numpy", "pandas", "sklearn", "lightgbm", "tensorflow")
MANAGED_SERVICES = ("daemon", "scenario", "live-ids")
PREDICTION_COLUMNS = ("final_decision", "fused_anom", "rsu_anom", "obu_anom")
RISK_SCORE_COLUMNS = ("p_final", "fused_risk", "rsu_score", "obu_risk")
CH3_IDENTITY_COLUMNS = ("row_id", "row_key", "receiver_pseudo", "sender_pseudo", "t_curr", "dt")
CH3_OBU_COLUMNS = (*OBU_FLAG_COLUMNS, "obu_flag_count", "obu_risk", "obu_anom")
CH3_HEAD_SCORE_COLUMNS = tuple(f"p_{head_name}" for head_name in HEAD_VECTOR_ORDER)
CH3_FUSION_COLUMNS = ("p_final",)
CH3_TRUST_COLUMNS = ("adaptive_threshold", "trust_sender_before", "trust_sender_after", "final_decision")
CH3_OUTPUT_COLUMNS = (
    *CH3_IDENTITY_COLUMNS,
    *CH3_OBU_COLUMNS,
    *CH3_HEAD_SCORE_COLUMNS,
    *CH3_FUSION_COLUMNS,
    *CH3_TRUST_COLUMNS,
)
PERFORMANCE_TARGETS = {
    "precision": 0.95,
    "recall": 0.90,
    "f1": 0.95,
    "fpr": 0.01,
}


def log(message: str) -> None:
    print(message, flush=True)


def ensure_dir(path: str | os.PathLike[str]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def log_cmd(cmd: str) -> None:
    log(f"CMD: {cmd}")


def run_blocking(cmd_list: List[str], cwd: Optional[str] = None) -> None:
    log_cmd(" ".join(shlex.quote(part) for part in cmd_list))
    subprocess.run(cmd_list, cwd=cwd, check=True)


def ensure_cli_runtime_dir() -> str:
    return ensure_dir(DEFAULT_CLI_RUNTIME_DIR)


def _service_state_path(service_name: str) -> Path:
    return Path(ensure_cli_runtime_dir()) / f"{service_name}.json"


def _service_log_path(service_name: str) -> Path:
    return Path(ensure_cli_runtime_dir()) / f"{service_name}.log"


def _read_service_state(service_name: str) -> Dict[str, Any]:
    path = _service_state_path(service_name)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_service_state(service_name: str, state: Dict[str, Any]) -> None:
    path = _service_state_path(service_name)
    path.write_text(json.dumps(state, indent=2), encoding="utf-8")


def _remove_service_state(service_name: str) -> None:
    path = _service_state_path(service_name)
    if path.exists():
        path.unlink()


def _pid_is_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def service_status(service_name: str) -> Dict[str, Any]:
    state = _read_service_state(service_name)
    pid = int(state.get("pid", 0) or 0)
    running = _pid_is_running(pid)
    if state and not running:
        state["running"] = False
    return {
        "service": service_name,
        "running": running,
        "pid": pid if pid > 0 else None,
        "started_at": state.get("started_at"),
        "cwd": state.get("cwd"),
        "log_path": state.get("log_path") or str(_service_log_path(service_name)),
        "cmd": state.get("cmd"),
    }


def _live_ids_runtime_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(base_env or os.environ.copy())
    env["DISPLAY"] = env.get("DISPLAY") or ":0"
    env["QT_QPA_PLATFORM"] = env.get("QT_QPA_PLATFORM") or "xcb"
    env["QT_OPENGL"] = env.get("QT_OPENGL") or "software"
    env["LIBGL_ALWAYS_SOFTWARE"] = env.get("LIBGL_ALWAYS_SOFTWARE") or "1"
    env["QT_XCB_GL_INTEGRATION"] = env.get("QT_XCB_GL_INTEGRATION") or "none"
    return env


def _python_probe_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(base_env or os.environ.copy())
    for key in ("VIRTUAL_ENV", "PYTHONHOME", "PYTHONPATH"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _conda_env_python_path(conda_env: str) -> Optional[str]:
    conda = shutil.which("conda")
    if not conda:
        return None
    result = subprocess.run(
        [conda, "env", "list", "--json"],
        check=False,
        capture_output=True,
        text=True,
        env=_python_probe_env(),
    )
    if result.returncode != 0:
        return None
    try:
        payload = json.loads(result.stdout)
    except Exception:
        return None

    env_name = str(conda_env or "").strip()
    env_details = payload.get("envs_details") or {}
    for env_path, detail in env_details.items():
        name = str((detail or {}).get("name") or "").strip()
        is_base = bool((detail or {}).get("base"))
        if name == env_name or (env_name == "base" and is_base):
            python_path = Path(env_path) / "bin" / "python"
            if python_path.exists():
                return str(python_path)
    return None


def _probe_python_imports(python_bin: str, modules: tuple[str, ...]) -> tuple[bool, List[str]]:
    probe_code = (
        "import importlib.util\n"
        f"modules = {modules!r}\n"
        "missing = [name for name in modules if importlib.util.find_spec(name) is None]\n"
        "print('\\n'.join(missing))\n"
        "raise SystemExit(0 if not missing else 3)\n"
    )
    result = subprocess.run(
        [python_bin, "-c", probe_code],
        check=False,
        capture_output=True,
        text=True,
        env=_python_probe_env(),
    )
    missing = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if result.returncode == 0:
        return True, []
    if result.returncode == 3:
        return False, missing
    stderr_summary = result.stderr.strip().splitlines()
    return False, [stderr_summary[-1] if stderr_summary else f"python exited with code {result.returncode}"]


def _resolve_python_for_modules(modules: tuple[str, ...], preferred_env: str = DEFAULT_CONDA_ENV) -> tuple[str, str]:
    candidates: List[tuple[str, str]] = []
    seen: set[str] = set()

    for env_name in [preferred_env, DEFAULT_CONDA_ENV, "base"]:
        env_name = str(env_name or "").strip()
        if not env_name:
            continue
        python_bin = _conda_env_python_path(env_name)
        if python_bin and python_bin not in seen:
            candidates.append((python_bin, env_name))
            seen.add(python_bin)

    current_python = sys.executable or shutil.which("python3") or shutil.which("python")
    if current_python and current_python not in seen:
        candidates.append((current_python, "current"))
        seen.add(current_python)

    errors: List[str] = []
    for python_bin, label in candidates:
        ok, missing = _probe_python_imports(python_bin, modules)
        if ok:
            return python_bin, label
        errors.append(f"{label} ({python_bin}): missing/failing {', '.join(missing)}")
    raise RuntimeError("No Python runtime is ready. " + " | ".join(errors))


def resolve_live_ids_conda_env(preferred_env: str = DEFAULT_CONDA_ENV) -> str:
    _, label = _resolve_python_for_modules(LIVE_IDS_REQUIRED_IMPORTS, preferred_env)
    return label


def _prepare_live_ids_dashboard_launch(live_dir: str, conda_env: str = DEFAULT_CONDA_ENV) -> tuple[str, str]:
    if not os.path.isdir(live_dir):
        raise FileNotFoundError(f"Live IDS directory not found: {live_dir}")
    main_py = os.path.join(live_dir, "main.py")
    if not os.path.exists(main_py):
        raise FileNotFoundError(f"main.py not found in {live_dir}")
    python_bin, resolved_env = _resolve_python_for_modules(LIVE_IDS_REQUIRED_IMPORTS, conda_env)
    runtime_env = _live_ids_runtime_env()
    env_prefix = " ".join(
        f"{key}={shlex.quote(str(runtime_env[key]))}"
        for key in ("DISPLAY", "QT_QPA_PLATFORM", "QT_OPENGL", "LIBGL_ALWAYS_SOFTWARE", "QT_XCB_GL_INTEGRATION")
    )
    cmd = (
        f"cd {shlex.quote(live_dir)} && "
        f"{env_prefix} "
        f"{shlex.quote(python_bin)} main.py"
    )
    return cmd, resolved_env


def _build_service_command(service_name: str, *, f2md_dir: str, live_ids_dir: str, conda_env: str) -> tuple[str, Optional[str], Dict[str, str]]:
    env = os.environ.copy()
    if service_name == "daemon":
        return cmd_launch_daemon(f2md_dir), f2md_dir, env
    if service_name == "scenario":
        return cmd_run_scenario(f2md_dir), f2md_dir, env
    if service_name == "live-ids":
        env = _live_ids_runtime_env(env)
        cmd, _ = _prepare_live_ids_dashboard_launch(live_ids_dir, conda_env)
        return cmd, live_ids_dir, env
    raise ValueError(f"Unsupported service: {service_name}")


def start_managed_service(
    service_name: str,
    *,
    f2md_dir: str = DEFAULT_F2MD_DIR,
    live_ids_dir: str = DEFAULT_LIVE_IDS_DIR,
    conda_env: str = DEFAULT_CONDA_ENV,
) -> Dict[str, Any]:
    if service_name not in MANAGED_SERVICES:
        raise ValueError(f"Unsupported service: {service_name}")
    current = service_status(service_name)
    if current["running"]:
        raise RuntimeError(f"{service_name} is already running with pid={current['pid']}.")

    cmd, cwd, env = _build_service_command(
        service_name,
        f2md_dir=f2md_dir,
        live_ids_dir=live_ids_dir,
        conda_env=conda_env,
    )
    log_path = _service_log_path(service_name)
    with open(log_path, "ab", buffering=0) as handle:
        process = subprocess.Popen(
            ["bash", "-lc", cmd],
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            close_fds=True,
        )
    state = {
        "service": service_name,
        "pid": process.pid,
        "cmd": cmd,
        "cwd": cwd,
        "log_path": str(log_path),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_service_state(service_name, state)
    return service_status(service_name)


def stop_managed_service(service_name: str, *, force: bool = False) -> Dict[str, Any]:
    if service_name not in MANAGED_SERVICES:
        raise ValueError(f"Unsupported service: {service_name}")
    state = _read_service_state(service_name)
    pid = int(state.get("pid", 0) or 0)
    if pid <= 0 or not _pid_is_running(pid):
        _remove_service_state(service_name)
        return service_status(service_name)

    sig = signal.SIGKILL if force else signal.SIGTERM
    try:
        os.killpg(pid, sig)
    except OSError:
        try:
            os.kill(pid, sig)
        except OSError:
            pass

    for _ in range(20):
        if not _pid_is_running(pid):
            break
        time.sleep(0.1)

    if not _pid_is_running(pid):
        _remove_service_state(service_name)
    return service_status(service_name)


def tail_text_file(path: str, lines: int = 40) -> str:
    file_path = Path(path)
    if not file_path.exists():
        return f"Log file not found: {file_path}"
    content = file_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if not content:
        return f"{file_path}\n<empty>"
    tail = content[-max(int(lines), 1):]
    return f"{file_path}\n" + "\n".join(tail)


def print_service_statuses() -> None:
    log("\nManaged service status:")
    for service_name in MANAGED_SERVICES:
        status = service_status(service_name)
        state = "RUNNING" if status["running"] else "STOPPED"
        pid = status["pid"] if status["pid"] is not None else "-"
        started = status["started_at"] or "-"
        log_path = status["log_path"]
        log(f"  - {service_name}: {state} | pid={pid} | started={started} | log={log_path}")


def _open_new_terminal(cmd: str, *, title: str = "VANET", cwd: Optional[str] = None) -> None:
    devnull = subprocess.DEVNULL
    runtime_dir = Path(ensure_cli_runtime_dir())

    def _write_terminal_script(command: str, workdir: Optional[str]) -> str:
        script_path = runtime_dir / f"terminal_{int(time.time() * 1000)}_{os.getpid()}.sh"
        lines = ["#!/usr/bin/env bash"]
        if workdir:
            lines.append(f"cd {shlex.quote(workdir)} || exit 1")
        lines.append(command)
        lines.append('status=$?')
        lines.append('printf "\\n[Command exited with status %s]\\n" "$status"')
        lines.append("exec bash")
        script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        os.chmod(script_path, os.stat(script_path).st_mode | stat.S_IXUSR)
        return str(script_path)

    qterminal = shutil.which("qterminal")
    if qterminal:
        script_path = _write_terminal_script(cmd, cwd)
        subprocess.Popen(
            [qterminal, "-e", script_path, "-w", cwd or str(Path.home())],
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    if shutil.which("gnome-terminal"):
        subprocess.Popen(
            ["gnome-terminal", "--title", title, "--", "bash", "-lc", cmd],
            cwd=cwd,
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    if shutil.which("x-terminal-emulator"):
        terminal_path = Path(shutil.which("x-terminal-emulator") or "").resolve()
        if terminal_path.name == "qterminal":
            script_path = _write_terminal_script(cmd, cwd)
            subprocess.Popen(
                [str(terminal_path), "-e", script_path, "-w", cwd or str(Path.home())],
                start_new_session=True,
                stdout=devnull,
                stderr=devnull,
            )
            return
        subprocess.Popen(
            ["x-terminal-emulator", "-e", "bash", "-lc", cmd],
            cwd=cwd,
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    subprocess.Popen(
        ["bash", "-lc", cmd],
        cwd=cwd,
        start_new_session=True,
        stdout=devnull,
        stderr=devnull,
    )


def open_in_new_terminal(cmd: str, *, cwd: Optional[str] = None, title: str = "VANET") -> None:
    _open_new_terminal(cmd, title=title, cwd=cwd)


def open_terminal_app(cwd: Optional[str] = None) -> None:
    workdir = cwd or DEFAULT_APP_DIR or str(Path.home())
    devnull = subprocess.DEVNULL
    qterminal = shutil.which("qterminal")
    if qterminal:
        subprocess.Popen(
            [qterminal, "-w", workdir],
            cwd=workdir,
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    terminal_path = shutil.which("x-terminal-emulator")
    if terminal_path and Path(terminal_path).resolve().name == "qterminal":
        subprocess.Popen(
            [terminal_path, "-w", workdir],
            cwd=workdir,
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    if shutil.which("gnome-terminal"):
        log("qterminal not found in PATH. Falling back to gnome-terminal.")
        subprocess.Popen(
            ["gnome-terminal", "--working-directory", workdir],
            cwd=workdir,
            start_new_session=True,
            stdout=devnull,
            stderr=devnull,
        )
        return
    log("qterminal not found in PATH. Falling back to the default terminal launcher.")
    open_in_new_terminal("exec bash", cwd=workdir, title="VANET Terminal")


def cmd_launch_daemon(f2md_dir: str) -> str:
    _ = f2md_dir
    return "./launchSumoTraciDaemon"


def cmd_run_scenario(f2md_dir: str) -> str:
    _ = f2md_dir
    return "./runScenario"


def cmd_live_ids_dashboard(live_dir: str, conda_env: str = DEFAULT_CONDA_ENV) -> str:
    cmd, _ = _prepare_live_ids_dashboard_launch(live_dir, conda_env)
    return cmd


def run_extract_intermsg(
    *,
    results_dir: str,
    root: str,
    out_csv: str,
    version: str = "v2",
    py: Optional[str] = None,
) -> None:
    """Keep the original F2MD extraction helper unchanged."""

    script = os.path.join(results_dir, "extract1_intermsg.py")
    if not os.path.exists(script):
        script = DEFAULT_EXTRACT_PY
    python_bin = py or (sys.executable if sys.executable else "python3")
    run_blocking(
        [python_bin, script, "--root", root, "--out", out_csv, "--version", version],
        cwd=results_dir,
    )


def add_attack_id_to_csv(
    *,
    csv_path: str,
    attack_id: int,
    output_path: Optional[str] = None,
    script_path: str = DEFAULT_ATTACK_ID_PY,
) -> str:
    """Keep add_attack_id.py as the post-extraction labeling step."""

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"add_attack_id.py not found at {script_path}")

    python_bin = sys.executable if sys.executable else "python3"
    cmd = [python_bin, script_path, "--input", csv_path, "--attack-id", str(int(attack_id))]
    if output_path and output_path != csv_path:
        cmd += ["--output", output_path]
    else:
        cmd.append("--inplace")
        output_path = csv_path
    run_blocking(cmd, cwd=os.path.dirname(script_path) or None)
    return str(output_path)


def read_local_attack_type(config_path: str = DEFAULT_BAGHDAD_OMNETPP) -> int:
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Scenario config not found: {config_path}")
    pattern = re.compile(r"^\s*\*\.node\[\*\]\.appl\.LOCAL_ATTACK_TYPE\s*=\s*(\d+)\s*$")
    for raw_line in Path(config_path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        match = pattern.match(line)
        if match:
            return int(match.group(1))
    raise ValueError(f"{LOCAL_ATTACK_TYPE_KEY} was not found in {config_path}")


def run_preprocessing(
    *,
    results_dir: str,
    root: str,
    out_csv: str,
    version: str = "v2",
    config_path: str = DEFAULT_BAGHDAD_OMNETPP,
    py: Optional[str] = None,
) -> str:
    run_extract_intermsg(
        results_dir=results_dir,
        root=root,
        out_csv=out_csv,
        version=version,
        py=py,
    )
    attack_id = read_local_attack_type(config_path)
    labeled_csv = add_attack_id_to_csv(csv_path=out_csv, attack_id=attack_id, output_path=out_csv)
    attack_name = ATTACK_TYPES.get(attack_id, "unknown")
    log(f"Preprocessing completed: {labeled_csv}")
    log(f"Attack ID applied from {config_path}: {LOCAL_ATTACK_TYPE_KEY} = {attack_id} ({attack_name})")
    return labeled_csv


def run_live_ids_dashboard(live_dir: str = DEFAULT_LIVE_IDS_DIR, conda_env: str = "base") -> None:
    cmd, resolved_env = _prepare_live_ids_dashboard_launch(live_dir, conda_env)
    open_in_new_terminal(
        cmd,
        cwd=live_dir,
        title="VANET Live IDS",
    )
    log(f"Live IDS dashboard opened in a new terminal from {live_dir} using Python runtime '{resolved_env}'")


def _prepare_trainer_gui_launch(trainer_path: str, conda_env: str = DEFAULT_CONDA_ENV) -> tuple[str, str]:
    if not os.path.exists(trainer_path):
        raise FileNotFoundError(f"Trainer script not found: {trainer_path}")
    python_bin, resolved_env = _resolve_python_for_modules(TRAINER_GUI_REQUIRED_IMPORTS, conda_env)
    runtime_env = _live_ids_runtime_env()
    env_prefix = " ".join(
        f"{key}={shlex.quote(str(runtime_env[key]))}"
        for key in ("DISPLAY", "QT_QPA_PLATFORM", "QT_OPENGL", "LIBGL_ALWAYS_SOFTWARE", "QT_XCB_GL_INTEGRATION")
    )
    cmd = (
        f"cd {shlex.quote(os.path.dirname(trainer_path) or os.getcwd())} && "
        f"{env_prefix} "
        f"{shlex.quote(python_bin)} {shlex.quote(trainer_path)}"
    )
    return cmd, resolved_env


def launch_rsu_trainer_gui(trainer_path: str = DEFAULT_RSU_TRAINER_PY) -> None:
    cmd, resolved_env = _prepare_trainer_gui_launch(trainer_path, DEFAULT_CONDA_ENV)
    open_in_new_terminal(
        cmd,
        cwd=os.path.dirname(trainer_path) or None,
        title="VANET RSU Trainer",
    )
    log(f"RSU Trainer GUI opened using Python runtime '{resolved_env}'")


def list_models_cli(models_root: str) -> None:
    models = list_model_directories(models_root)
    if not models:
        log(f"No ready model directories found under {os.path.abspath(models_root)}")
        return
    log(f"Available model directories under {os.path.abspath(models_root)}:")
    for item in models:
        feature_count = item["feature_count"] if item["feature_count"] is not None else "-"
        family = item.get("family") or "-"
        created = item.get("created") or "-"
        runtime = item.get("score_runtime") or "-"
        head_name = item.get("head_name") or "-"
        log(
            f"  - {item['path']} | type={item['type']} | family={family} | head={head_name} | "
            f"version={item['version']} | created={created} | features={feature_count} | runtime={runtime}"
        )


def _safe_float(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _format_float(value: Any, digits: int = 4) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number:.{digits}f}"


def _format_int(value: Any) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return str(int(round(number)))


def _format_percent(value: Any, digits: int = 2) -> str:
    number = _safe_float(value)
    if number is None:
        return "-"
    return f"{number * 100:.{digits}f}%"


def _score_bar(score: Optional[float], width: int = 24) -> str:
    if score is None or pd.isna(score):
        return "." * width
    value = float(max(0.0, min(1.0, score)))
    filled = int(round(value * width))
    return "#" * filled + "." * (width - filled)


def _display_date_for(path: Path) -> str:
    if "results and models" in str(path):
        return "2025-12-20"
    return time.strftime("%Y-%m-%d", time.localtime(path.stat().st_mtime))


def _relative_path_for(path: Path, root_path: Path) -> str:
    try:
        return str(path.relative_to(root_path))
    except Exception:
        return str(path)


def _attack_families_for_id(attack_id: Optional[int]) -> List[str]:
    if attack_id is None:
        return ["unknown"]
    if attack_id == 0:
        return ["genuine"]
    families = [family for family, ids in ATTACK_FAMILIES.items() if attack_id in ids]
    return families or ["unknown"]


def _best_by_score(rows: List[Dict[str, Any]], score_keys: tuple[str, ...]) -> Optional[Dict[str, Any]]:
    if not rows:
        return None
    return max(
        rows,
        key=lambda row: tuple(_safe_float(row.get(key)) if _safe_float(row.get(key)) is not None else -1.0 for key in score_keys),
    )


def _collect_family_metric_rows(history_root: str) -> List[Dict[str, Any]]:
    root_path = Path(history_root)
    if not root_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for path in sorted(root_path.rglob("*_family_metrics.csv")):
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception:
            continue
        if "Attack family" not in df.columns:
            continue
        for _, row in df.iterrows():
            family = str(row.get("Attack family", "unknown")).strip() or "unknown"
            rows.append(
                {
                    "path": str(path),
                    "relative_path": _relative_path_for(path, root_path),
                    "display_date": _display_date_for(path),
                    "family": family,
                    "accuracy": _safe_float(row.get("ACC")),
                    "precision": _safe_float(row.get("Precision")),
                    "recall": _safe_float(row.get("Recall")),
                    "f1": _safe_float(row.get("F1")),
                    "roc_auc": _safe_float(row.get("ROC-AUC")),
                    "test_benign": _safe_float(row.get("#Test benign")),
                    "test_attack": _safe_float(row.get("#Test attack")),
                }
            )
    return rows


def _collect_eval_report_rows(history_root: str) -> List[Dict[str, Any]]:
    root_path = Path(history_root)
    if not root_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    for path in sorted(root_path.rglob("eval_report_*.json")):
        try:
            report = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        attack_id_text = path.name.removeprefix("eval_report_").removesuffix(".json")
        attack_id = int(attack_id_text) if attack_id_text.isdigit() else None
        metrics = report.get("model_metrics", {}) or {}
        detect_csv = path.with_name(f"detect_{attack_id}.csv") if attack_id is not None else None
        rows.append(
            {
                "path": str(path),
                "relative_path": _relative_path_for(path, root_path),
                "display_date": _display_date_for(path),
                "detect_csv": str(detect_csv) if detect_csv and detect_csv.exists() else None,
                "attack_id": attack_id,
                "attack_name": ATTACK_TYPES.get(attack_id, "unknown"),
                "families": _attack_families_for_id(attack_id),
                "rows": report.get("rows_eval_dt<=60.0") or report.get("rows_total_labels"),
                "precision": _safe_float(metrics.get("precision")),
                "recall": _safe_float(metrics.get("recall")),
                "f1": _safe_float(metrics.get("f1")),
                "support": _safe_float(metrics.get("support_attacks")),
                "tp": _safe_float(metrics.get("TP")),
                "fp": _safe_float(metrics.get("FP")),
                "fn": _safe_float(metrics.get("FN")),
                "tn": _safe_float(metrics.get("TN")),
            }
        )
    return rows


def _target_status(value: Any, target: float, *, higher_is_better: bool = True) -> str:
    number = _safe_float(value)
    if number is None:
        return "n/a"
    passed = number >= target if higher_is_better else number <= target
    return "ok" if passed else "below-target"


def _target_summary(item: Dict[str, Any]) -> str:
    checks = [
        f"F1 { _target_status(item.get('score'), PERFORMANCE_TARGETS['f1']) }",
        f"precision { _target_status(item.get('precision'), PERFORMANCE_TARGETS['precision']) }",
        f"recall { _target_status(item.get('recall'), PERFORMANCE_TARGETS['recall']) }",
    ]
    fpr = _safe_float(item.get("fpr"))
    if fpr is not None:
        checks.append(f"FPR { _target_status(fpr, PERFORMANCE_TARGETS['fpr'], higher_is_better=False) }")
    return ", ".join(checks)


def _log_performance_targets() -> None:
    log("")
    log("Expected Thesis-Level Target Bands:")
    log(
        "  - actual run should be near: "
        f"F1>={PERFORMANCE_TARGETS['f1']:.2f}, "
        f"Precision>={PERFORMANCE_TARGETS['precision']:.2f}, "
        f"Recall>={PERFORMANCE_TARGETS['recall']:.2f}, "
        f"FPR<={PERFORMANCE_TARGETS['fpr']:.2f}"
    )
    log("  - these are target bands for judging new runs, not stored paper result rows.")


def _missing_columns(columns: set[str], required: tuple[str, ...]) -> List[str]:
    return [column for column in required if column not in columns]


def _short_missing_text(missing: List[str], *, max_items: int = 6) -> str:
    if not missing:
        return "none"
    preview = ", ".join(missing[:max_items])
    if len(missing) > max_items:
        preview += f", ... (+{len(missing) - max_items})"
    return preview


def _chapter3_output_missing_columns(df: pd.DataFrame) -> List[str]:
    return _missing_columns(set(df.columns), CH3_OUTPUT_COLUMNS)


def _chapter3_runtime_manifest(runtime: Any) -> Dict[str, Any]:
    manifest = getattr(runtime, "manifest", None)
    return manifest if isinstance(manifest, dict) else {}


def _chapter3_offline_audit(
    runtime: Any,
    scored: pd.DataFrame,
    scorecard: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    manifest = _chapter3_runtime_manifest(runtime)
    columns = set(scored.columns)
    artifact_family = str(manifest.get("artifact_family") or type(runtime).__name__)
    thesis_grade_complete = bool(manifest.get("thesis_grade_complete", False))
    chapter3_operations_applied = bool(manifest.get("chapter3_operations_applied", False))
    manifest_enabled_heads = [str(head) for head in manifest.get("enabled_heads", []) if str(head)]
    enabled_heads = manifest_enabled_heads or [
        head_name for head_name in HEAD_VECTOR_ORDER if f"p_{head_name}" in columns
    ]
    missing_enabled_heads = [head for head in HEAD_VECTOR_ORDER if head not in enabled_heads]
    is_complete_bundle = (
        artifact_family == ARTIFACT_FAMILY
        and thesis_grade_complete
        and not missing_enabled_heads
    )
    release_status_ok: Optional[bool] = is_complete_bundle
    if chapter3_operations_applied and not is_complete_bundle:
        release_status_ok = None

    checks: List[Dict[str, Any]] = []

    def add(name: str, ok: Optional[bool], detail: str) -> None:
        if ok is None:
            status = "WARN"
        else:
            status = "PASS" if ok else "FAIL"
        checks.append({"name": name, "status": status, "detail": detail})

    add(
        "Release bundle",
        release_status_ok,
        (
            f"artifact_family={artifact_family}; thesis_grade_complete={thesis_grade_complete}; "
            f"chapter3_operations_applied={chapter3_operations_applied}; "
            f"enabled_heads={','.join(enabled_heads) or '-'}"
            f"{' (from output)' if enabled_heads and not manifest_enabled_heads else ''}; "
            f"missing_heads={','.join(missing_enabled_heads) or '-'}"
        ),
    )

    missing_identity = _missing_columns(columns, CH3_IDENTITY_COLUMNS)
    add(
        "Stage 2 normalized identity/timing output",
        not missing_identity,
        f"required={len(CH3_IDENTITY_COLUMNS)}; missing={_short_missing_text(missing_identity)}",
    )

    missing_obu = _missing_columns(columns, CH3_OBU_COLUMNS)
    add(
        "Stage 1 OBU physical/protocol screening",
        not missing_obu,
        f"required={len(CH3_OBU_COLUMNS)}; missing={_short_missing_text(missing_obu)}",
    )

    missing_heads = _missing_columns(columns, CH3_HEAD_SCORE_COLUMNS)
    add(
        "RSU hybrid detector heads",
        not missing_heads,
        f"required={','.join(CH3_HEAD_SCORE_COLUMNS)}; missing={_short_missing_text(missing_heads)}",
    )

    missing_fusion = _missing_columns(columns, CH3_FUSION_COLUMNS)
    add(
        "Stacking meta-classifier fusion",
        not missing_fusion,
        (
            f"p_final_present={'p_final' in columns}; "
            f"strict_bundle_required_for_saved_global_meta={ARTIFACT_FAMILY}; "
            f"missing={_short_missing_text(missing_fusion)}"
        ),
    )

    missing_trust = _missing_columns(columns, CH3_TRUST_COLUMNS)
    add(
        "Adaptive trust-gated decision",
        not missing_trust,
        f"required={','.join(CH3_TRUST_COLUMNS)}; missing={_short_missing_text(missing_trust)}",
    )

    if scorecard and scorecard.get("metric_name"):
        f1 = _safe_float(scorecard.get("score"))
        precision = _safe_float(scorecard.get("precision"))
        recall = _safe_float(scorecard.get("recall"))
        fpr = _safe_float(scorecard.get("fpr"))
        metrics_ok = (
            f1 is not None
            and precision is not None
            and recall is not None
            and fpr is not None
            and f1 >= PERFORMANCE_TARGETS["f1"]
            and precision >= PERFORMANCE_TARGETS["precision"]
            and recall >= PERFORMANCE_TARGETS["recall"]
            and fpr <= PERFORMANCE_TARGETS["fpr"]
        )
        add(
            "Chapter 4-style target-band check",
            metrics_ok,
            (
                f"F1={_format_float(f1)}; Precision={_format_float(precision)}; "
                f"Recall={_format_float(recall)}; FPR={_format_float(fpr, 6)}"
            ),
        )
    else:
        add(
            "Chapter 4-style target-band check",
            None,
            "labels were not available in the detection CSV, so F1/precision/recall/FPR were not checked",
        )

    return checks


def _log_pipeline_audit(checks: List[Dict[str, Any]]) -> None:
    failed = [check for check in checks if check["status"] == "FAIL"]
    structural_failed = [
        check for check in failed if check["name"] != "Chapter 4-style target-band check"
    ]
    metric_failed = [
        check for check in failed if check["name"] == "Chapter 4-style target-band check"
    ]
    warned = [check for check in checks if check["status"] == "WARN"]
    log("")
    log("Offline Detection Audit:")
    for check in checks:
        log(f"  - {check['name']}: {check['status']} | {check['detail']}")
    if structural_failed:
        log(
            "  result: offline pipeline is incomplete. Treat this run as scenario validation only, "
            "not final thesis evidence."
        )
    elif metric_failed:
        log("  result: offline operations complete, but the target band is not fully met.")
    elif warned:
        log("  result: offline pipeline columns are complete, but metric evidence is incomplete.")
    else:
        log("  result: complete offline detection path with checked target-band metrics.")


def _log_pipeline_runtime_steps(runtime: Any) -> None:
    manifest = _chapter3_runtime_manifest(runtime)
    enabled_heads = [str(head) for head in manifest.get("enabled_heads", []) if str(head)]
    family_bundles = manifest.get("family_bundles", {})
    log("")
    log("Offline Pipeline Steps:")
    log("  1. OBU screening: physical, consistency, timestamp, and protocol evidence flags")
    log("  2. Feature extraction: kinematic, timing/rate, rolling-window, and Sybil/network features")
    if enabled_heads:
        log(f"  3. RSU detector heads: {', '.join(enabled_heads)}")
    else:
        log("  3. RSU detector heads: runtime will report available score columns after detection")
    if isinstance(family_bundles, dict) and family_bundles:
        for family, detail in family_bundles.items():
            log(
                f"     - {family}: {detail.get('path')} | "
                f"features={detail.get('feature_count')} | explicit_head={detail.get('has_explicit_head')}"
            )
    log("  4. Fusion: produce p_final from RSU head scores")
    log("  5. Trust gate: sender trust updates adaptive_threshold and final_decision")


def _looks_like_detection_output(path: Path) -> bool:
    name = path.name.lower()
    if path.suffix.lower() != ".csv":
        return False
    if name.endswith(("_family_metrics.csv", "_cleaning_stats.csv", "_class_distribution.csv", "_scaling_stats.csv")):
        return False
    if not any(token in name for token in ("detect", "offline", "out", "audit")):
        return False
    try:
        columns = set(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return False
    return bool(columns.intersection(PREDICTION_COLUMNS) or columns.intersection(RISK_SCORE_COLUMNS))


def _detection_scorecard_from_dataframe(path: Path, root_path: Path, df: pd.DataFrame) -> Dict[str, Any]:
    prediction_col = next((col for col in PREDICTION_COLUMNS if col in df.columns), None)
    score_col = next((col for col in RISK_SCORE_COLUMNS if col in df.columns), None)
    prediction = None
    if prediction_col:
        prediction = pd.to_numeric(df[prediction_col], errors="coerce").fillna(0).astype(int)
    score_series = None
    if score_col:
        score_series = pd.to_numeric(df[score_col], errors="coerce")

    label_series = None
    if "label" in df.columns:
        label_series = pd.to_numeric(df["label"], errors="coerce").fillna(0).astype(int).clip(0, 1)
    elif "attack_id" in df.columns:
        attack_ids = pd.to_numeric(df["attack_id"], errors="coerce").fillna(0).astype(int)
        label_series = (attack_ids != 0).astype(int)

    metric_score = None
    metric_name = None
    accuracy = precision = recall = None
    tn = fp = fn = tp = fpr = tnr = None
    attack_support = int(label_series.sum()) if label_series is not None else None
    if prediction is not None and label_series is not None and len(df) > 0:
        report = classification_report(label_series, prediction, output_dict=True, zero_division=0)
        cm = confusion_matrix(label_series, prediction, labels=[0, 1])
        tn, fp, fn, tp = (int(value) for value in cm.ravel())
        fpr = float(fp / (fp + tn)) if (fp + tn) else None
        tnr = float(tn / (tn + fp)) if (tn + fp) else None
        attack_report = report.get("1", {})
        weighted_report = report.get("weighted avg", {})
        attack_f1 = _safe_float(attack_report.get("f1-score"))
        weighted_f1 = _safe_float(weighted_report.get("f1-score"))
        metric_score = attack_f1 if attack_support and attack_support > 0 else weighted_f1
        metric_name = "attack_f1" if attack_support and attack_support > 0 else "weighted_f1"
        accuracy = _safe_float(report.get("accuracy"))
        precision = _safe_float(attack_report.get("precision"))
        recall = _safe_float(attack_report.get("recall"))

    family_metrics: List[Dict[str, Any]] = []
    families_seen: List[str] = []
    if prediction is not None and "attack_id" in df.columns:
        attack_ids = pd.to_numeric(df["attack_id"], errors="coerce").fillna(0).astype(int)
        seen: set[str] = set()
        for attack_id in sorted(set(attack_ids.tolist())):
            if attack_id == 0:
                continue
            for family in _attack_families_for_id(attack_id):
                if family not in {"genuine", "unknown"}:
                    seen.add(family)
        families_seen = sorted(seen)

        for family, family_ids in ATTACK_FAMILIES.items():
            y_family = attack_ids.isin(family_ids).astype(int)
            support = int(y_family.sum())
            if support <= 0:
                continue
            report = classification_report(y_family, prediction, output_dict=True, zero_division=0)
            attack_report = report.get("1", {})
            family_metrics.append(
                {
                    "family": family,
                    "support": support,
                    "accuracy": _safe_float(report.get("accuracy")),
                    "precision": _safe_float(attack_report.get("precision")),
                    "recall": _safe_float(attack_report.get("recall")),
                    "f1": _safe_float(attack_report.get("f1-score")),
                }
            )

    alerts = int(prediction.sum()) if prediction is not None else None
    alert_rate = float(alerts / len(df)) if alerts is not None and len(df) else None
    runtime_values: List[str] = []
    for runtime_col in ("runtime_mode", "legacy_family"):
        if runtime_col in df.columns:
            runtime_values.extend(str(value) for value in df[runtime_col].dropna().astype(str).unique()[:3])
    chapter3_missing_columns = _chapter3_output_missing_columns(df)

    return {
        "path": str(path),
        "relative_path": _relative_path_for(path, root_path),
        "display_date": _display_date_for(path) if path.exists() else time.strftime("%Y-%m-%d"),
        "rows": int(len(df)),
        "prediction_col": prediction_col,
        "score_col": score_col,
        "score": metric_score,
        "metric_name": metric_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "fpr": fpr,
        "tnr": tnr,
        "attack_support": attack_support,
        "alerts": alerts,
        "alert_rate": alert_rate,
        "mean_risk": _safe_float(score_series.mean()) if score_series is not None else None,
        "p95_risk": _safe_float(score_series.quantile(0.95)) if score_series is not None and len(score_series) else None,
        "max_risk": _safe_float(score_series.max()) if score_series is not None else None,
        "families": families_seen,
        "family_metrics": family_metrics,
        "runtime": ",".join(runtime_values) if runtime_values else "-",
        "chapter3_output_complete": not chapter3_missing_columns,
        "chapter3_missing_columns": chapter3_missing_columns,
    }


def _log_detection_scorecard(item: Dict[str, Any], title: str = "Family Performance") -> None:
    score = _safe_float(item.get("score"))
    log("")
    log(f"{title}:")
    log("  family           f1      accuracy  precision  recall  fpr       tp    fp    fn    tn")
    if item.get("metric_name"):
        families = item.get("family_metrics") or []
        if families:
            for row in sorted(families, key=lambda family_row: str(family_row.get("family") or "")):
                log(
                    f"  {str(row.get('family') or '-')[:15]:15} "
                    f"{_format_float(row.get('f1')):7} "
                    f"{_format_float(row.get('accuracy')):9} "
                    f"{_format_float(row.get('precision')):10} "
                    f"{_format_float(row.get('recall')):7} "
                    f"{_format_float(item.get('fpr'), 6):9} "
                    f"{_format_int(item.get('tp')):5} "
                    f"{_format_int(item.get('fp')):5} "
                    f"{_format_int(item.get('fn')):5} "
                    f"{_format_int(item.get('tn')):5}"
                )
        else:
            log(
                f"  {'detected':15} "
                f"{_format_float(score):7} "
                f"{_format_float(item.get('accuracy')):9} "
                f"{_format_float(item.get('precision')):10} "
                f"{_format_float(item.get('recall')):7} "
                f"{_format_float(item.get('fpr'), 6):9} "
                f"{_format_int(item.get('tp')):5} "
                f"{_format_int(item.get('fp')):5} "
                f"{_format_int(item.get('fn')):5} "
                f"{_format_int(item.get('tn')):5}"
            )
    else:
        log(
            f"  {'unverified':15} "
            f"{_format_float(score):7} "
            f"{_format_float(item.get('accuracy')):9} "
            f"{_format_float(item.get('precision')):10} "
            f"{_format_float(item.get('recall')):7} "
            f"{_format_float(item.get('fpr'), 6):9} "
            f"{_format_int(item.get('tp')):5} "
            f"{_format_int(item.get('fp')):5} "
            f"{_format_int(item.get('fn')):5} "
            f"{_format_int(item.get('tn')):5}"
        )


def _vehicle_decision_rows(scored: pd.DataFrame) -> List[Dict[str, Any]]:
    required = {"sender_pseudo", "p_final", "final_decision"}
    if not required.issubset(scored.columns):
        return []
    ordered = scored.copy()
    sort_cols = [col for col in ("sender_pseudo", "t_curr", "row_id") if col in ordered.columns]
    if sort_cols:
        ordered = ordered.sort_values(sort_cols, kind="mergesort")
    trust_col = "trust_sender_after" if "trust_sender_after" in ordered.columns else "trust_sender_before"
    rows: List[Dict[str, Any]] = []
    for sender, group in ordered.groupby("sender_pseudo", sort=False):
        p_final = pd.to_numeric(group["p_final"], errors="coerce").fillna(0.0)
        decision = pd.to_numeric(group["final_decision"], errors="coerce").fillna(0).astype(int)
        trust_score = None
        if trust_col in group.columns:
            trust_values = pd.to_numeric(group[trust_col], errors="coerce").dropna()
            if not trust_values.empty:
                trust_score = float(trust_values.iloc[-1])
        rows.append(
            {
                "vehicle": str(sender),
                "trust_score": trust_score,
                "p_final": float(p_final.max()) if len(p_final) else None,
                "final_decision": int(decision.max()) if len(decision) else 0,
            }
        )
    return sorted(rows, key=lambda row: (-int(row["final_decision"]), -float(row.get("p_final") or 0.0), row["vehicle"]))


def _log_vehicle_decision_table(scored: pd.DataFrame) -> None:
    rows = _vehicle_decision_rows(scored)
    if not rows:
        return
    log("")
    log("Vehicle Decisions:")
    log("  vehicle                  trust_score  p_final  final_decision")
    for row in rows:
        decision_text = "ATTACK" if int(row["final_decision"]) == 1 else "BENIGN"
        log(
            f"  {row['vehicle'][:24]:24} "
            f"{_format_float(row.get('trust_score')):11} "
            f"{_format_float(row.get('p_final')):7} "
            f"{decision_text}"
        )


def _collect_offline_detection_scorecards(history_root: str) -> List[Dict[str, Any]]:
    root_path = Path(history_root)
    if not root_path.exists():
        return []

    rows: List[Dict[str, Any]] = []
    required_columns = {
        "label",
        "attack_id",
        "sender_pseudo",
        "receiver_pseudo",
        "runtime_mode",
        "legacy_family",
        *PREDICTION_COLUMNS,
        *RISK_SCORE_COLUMNS,
        *CH3_OUTPUT_COLUMNS,
    }
    for path in sorted(root_path.rglob("*.csv")):
        if not _looks_like_detection_output(path):
            continue
        try:
            header = list(pd.read_csv(path, nrows=0).columns)
            usecols = [col for col in header if col in required_columns]
            df = pd.read_csv(path, usecols=usecols, low_memory=False)
        except Exception:
            continue
        rows.append(_detection_scorecard_from_dataframe(path, root_path, df))
    return sorted(
        rows,
        key=lambda row: (
            _safe_float(row.get("score")) if _safe_float(row.get("score")) is not None else -1.0,
            _safe_float(row.get("mean_risk")) if _safe_float(row.get("mean_risk")) is not None else -1.0,
            row.get("display_date") or "",
        ),
        reverse=True,
    )


def _discussion_lines(
    scored: List[Dict[str, Any]],
    family_rows: List[Dict[str, Any]],
    eval_rows: List[Dict[str, Any]],
    detection_rows: List[Dict[str, Any]],
) -> List[str]:
    lines: List[str] = []
    scored_detection_rows = [row for row in detection_rows if _safe_float(row.get("score")) is not None]
    best_detection = _best_by_score(scored_detection_rows, ("score", "recall", "precision"))
    if best_detection:
        lines.append(
            "Current best checked detection CSV: "
            f"{best_detection.get('relative_path')} has {_format_float(best_detection.get('score'))} "
            f"{best_detection.get('metric_name')}; target check: {_target_summary(best_detection)}."
        )
        if _safe_float(best_detection.get("score")) == 0.0:
            lines.append(
                "This run is not discussion-ready because the detector produced no useful attack recall. "
                "Use a compatible thesis-grade multi-head bundle or retrain on the current Baghdad/F2MD schema."
            )
        if not best_detection.get("chapter3_output_complete"):
            lines.append(
                "This score is not full offline-pipeline evidence because the detection CSV is missing required "
                f"offline-pipeline columns: {_short_missing_text(best_detection.get('chapter3_missing_columns') or [])}."
            )
    else:
        lines.append("No labeled offline detection scorecard was found. Run detect-offline, then verify to generate one.")

    if family_rows:
        lines.append(
            "Archived *_family_metrics.csv files are diagnostics only; do not use isolated perfect rows as current run performance."
        )

    thesis_candidate_scores = [item for item in scored if item.get("type") != "family_metrics_csv"]
    best_scored = _best_by_score(thesis_candidate_scores, ("score",))
    if best_scored:
        lines.append(
            "Highest archived diagnostic artifact: "
            f"{_format_float(best_scored.get('score'))} {best_scored.get('metric_name') or 'score'} "
            f"from {best_scored.get('relative_path') or best_scored.get('path')}."
        )

    eval_scored = [row for row in eval_rows if _safe_float(row.get("f1")) is not None]
    best_eval = _best_by_score(eval_scored, ("f1", "recall", "precision"))
    if best_eval:
        lines.append(
            "Highest legacy per-attack offline eval diagnostic: "
            f"attack_id={best_eval.get('attack_id')} {best_eval.get('attack_name')} "
            f"({'+'.join(best_eval.get('families', []))}) with F1={_format_float(best_eval.get('f1'))}, "
            f"Precision={_format_float(best_eval.get('precision'))}, Recall={_format_float(best_eval.get('recall'))}."
        )

    if not scored_detection_rows and detection_rows:
        lowest_mean = min(
            detection_rows,
            key=lambda row: _safe_float(row.get("mean_risk")) if _safe_float(row.get("mean_risk")) is not None else 999.0,
        )
        lines.append(
            "Detection CSVs were found without labels for verified F1; "
            f"the lowest mean-risk run is {lowest_mean.get('relative_path')} "
            f"with mean_risk={_format_float(lowest_mean.get('mean_risk'))}."
        )

    if not lines:
        lines.append("No scored artifacts were found; run train-rsu, detect-offline, and verify to create performance evidence.")
    return lines


def performance_score_cli(history_root: str, limit: int, extra_roots: Optional[List[str]] = None) -> None:
    roots: List[str] = []
    for root in [history_root, *(extra_roots or [])]:
        if not root:
            continue
        abs_root = os.path.abspath(root)
        if abs_root not in roots:
            roots.append(abs_root)

    history: List[Dict[str, Any]] = []
    family_rows: List[Dict[str, Any]] = []
    eval_rows: List[Dict[str, Any]] = []
    detection_rows: List[Dict[str, Any]] = []
    for root in roots:
        history.extend(list_historical_results(root))
        family_rows.extend(_collect_family_metric_rows(root))
        eval_rows.extend(_collect_eval_report_rows(root))
        detection_rows.extend(_collect_offline_detection_scorecards(root))

    if not history and not family_rows and not eval_rows and not detection_rows:
        log(f"No performance result files found under: {', '.join(roots)}")
        return
    detection_rows = sorted(
        detection_rows,
        key=lambda row: (
            _safe_float(row.get("score")) if _safe_float(row.get("score")) is not None else -1.0,
            _safe_float(row.get("mean_risk")) if _safe_float(row.get("mean_risk")) is not None else -1.0,
            row.get("display_date") or "",
        ),
        reverse=True,
    )
    scored = [item for item in history if _safe_float(item.get("score")) is not None]
    profile_only = [item for item in history if item not in scored]
    type_counts = pd.Series([item["type"] for item in history]).value_counts().to_dict()

    log("Performance Score Dashboard:")
    for root in roots:
        log(f"  - scan_root: {root}")
    log("")
    log("Overview:")
    log(f"  - historical_artifacts: {len(history)}")
    log(f"  - scored_artifacts: {len(scored)}")
    log(f"  - offline_detection_csvs_checked: {len(detection_rows)}")
    log(f"  - offline_eval_reports_checked: {len(eval_rows)}")
    log(f"  - family_metric_rows_checked: {len(family_rows)}")
    log(f"  - supporting_profile_artifacts: {len(profile_only)}")
    archive_dates = sorted({item.get('display_date') for item in history if item.get('display_date')})
    if archive_dates:
        log(f"  - display_dates: {', '.join(archive_dates[:3])}{' ...' if len(archive_dates) > 3 else ''}")

    log("")
    log("Artifact Mix:")
    for artifact_type, count in sorted(type_counts.items(), key=lambda item: (-item[1], item[0])):
        log(f"  - {artifact_type}: {count}")
    if detection_rows:
        log(f"  - offline_detection_csv: {len(detection_rows)}")

    _log_performance_targets()

    if eval_rows:
        log("")
        log("Legacy Offline Eval Diagnostics By Group:")
        for family in ("pos_speed", "replay_stale", "dos", "sybil"):
            candidates = [row for row in eval_rows if family in row.get("families", []) and _safe_float(row.get("f1")) is not None]
            best = _best_by_score(candidates, ("f1", "recall", "precision"))
            if not best:
                continue
            log(
                f"  - {family}: [{_score_bar(best.get('f1'))}] F1={_format_float(best.get('f1'))} | "
                f"Precision={_format_float(best.get('precision'))} | Recall={_format_float(best.get('recall'))} | "
                f"attack_id={best.get('attack_id')} {best.get('attack_name')} | support={_format_int(best.get('support'))}"
            )
            log(f"    {best.get('relative_path')}")

    if detection_rows:
        log("")
        log("Offline Detection Scorecards:")
        for item in detection_rows[:limit]:
            score = _safe_float(item.get("score"))
            metric_name = item.get("metric_name") or "unverified"
            family_text = ",".join(item.get("families") or []) or "-"
            log(
                f"  - [{_score_bar(score)}] {_format_float(score)} {metric_name} | "
                f"date={item.get('display_date', '-')} | rows={item.get('rows')} | "
                f"alerts={_format_int(item.get('alerts'))} | alert_rate={_format_float(item.get('alert_rate'))} | "
                f"mean_risk={_format_float(item.get('mean_risk'))} | p95_risk={_format_float(item.get('p95_risk'))} | "
                f"max_risk={_format_float(item.get('max_risk'))} | families={family_text}"
            )
            if item.get("metric_name"):
                log(
                    f"    confusion: TN={_format_int(item.get('tn'))} | FP={_format_int(item.get('fp'))} | "
                    f"FN={_format_int(item.get('fn'))} | TP={_format_int(item.get('tp'))} | "
                    f"FPR={_format_float(item.get('fpr'), 6)} | target={_target_summary(item)}"
                )
            log(f"    {item.get('relative_path')}")
            if item.get("family_metrics"):
                family_details = sorted(
                    item["family_metrics"],
                    key=lambda row: _safe_float(row.get("f1")) if _safe_float(row.get("f1")) is not None else -1.0,
                    reverse=True,
                )
                detail = "; ".join(
                    f"{row['family']} F1={_format_float(row.get('f1'))} P={_format_float(row.get('precision'))} "
                    f"R={_format_float(row.get('recall'))} support={row.get('support')}"
                    for row in family_details[:4]
                )
                log(f"    family detail: {detail}")
            else:
                log(
                    f"    detail: pred={item.get('prediction_col') or '-'} | "
                    f"score={item.get('score_col') or '-'} | runtime={item.get('runtime') or '-'}"
                )

    thesis_candidate_scores = [item for item in scored if item.get("type") != "family_metrics_csv"]
    family_metric_scores = [item for item in scored if item.get("type") == "family_metrics_csv"]
    if thesis_candidate_scores:
        log("")
        log("Historical Evaluation Diagnostics:")
        ranked = sorted(thesis_candidate_scores, key=lambda item: _safe_float(item.get("score")) or -1.0, reverse=True)[:limit]
        for item in ranked:
            score = _safe_float(item["score"])
            metric_name = item.get("metric_name") or "score"
            rows = item["rows"] if item["rows"] is not None else "-"
            threshold = item["threshold"] if item["threshold"] is not None else "-"
            note = item["note"] or "-"
            rel_path = item.get("relative_path") or item["path"]
            log(
                f"  - [{_score_bar(score)}] {_format_float(score)} {metric_name} | "
                f"date={item.get('display_date', '-')} | rows={rows} | threshold={threshold}"
            )
            log(f"    {rel_path}")
            log(f"    detail: {note}")
    if family_metric_scores:
        log("")
        log(
            "Family metric CSV diagnostics skipped from ranking: "
            f"{len(family_metric_scores)} files. Use current offline scorecards for run performance."
        )

    if profile_only:
        log("")
        log("Supporting Artifacts:")
        support_limit = min(limit, 5)
        for item in profile_only[:support_limit]:
            rel_path = item.get("relative_path") or item["path"]
            note = item["note"] or "-"
            log(f"  - {item.get('display_date', '-')} | {item['type']} | {rel_path}")
            log(f"    detail: {note}")
        if len(profile_only) > support_limit:
            log(f"  - ... {len(profile_only) - support_limit} more supporting artifacts hidden")

    log("")
    log("Discussion Comment:")
    for line in _discussion_lines(scored, family_rows, eval_rows, detection_rows):
        log(f"  - {line}")

    remaining_history = len(history) - min(len(history), limit)
    remaining_detections = len(detection_rows) - min(len(detection_rows), limit)
    if remaining_history > 0 or remaining_detections > 0:
        log("")
        log(
            "More files are available: "
            f"historical={max(remaining_history, 0)}, detections={max(remaining_detections, 0)}. "
            "Increase --limit to view more."
        )


def list_history_cli(history_root: str, limit: int) -> None:
    performance_score_cli(history_root, limit)


def show_log_cli(service_name: str, lines: int) -> None:
    if service_name not in MANAGED_SERVICES:
        raise ValueError(f"Unsupported service: {service_name}")
    status = service_status(service_name)
    log(tail_text_file(status["log_path"], lines=lines))


def print_cli_home(
    *,
    f2md_dir: str,
    results_dir: str,
    models_dir: str,
    history_root: str,
) -> None:
    log("\nCLI Control Center:")
    log("  Terminal-based tools open in new terminal windows.")
    log("\nPaths:")
    log(f"  - f2md_dir: {f2md_dir}")
    log(f"  - results_dir: {results_dir}")
    log(f"  - models_dir: {models_dir}")
    log(f"  - history_root: {history_root}")
    log("\nOne-terminal commands:")
    log("   1. open-terminal")
    log("   2. list-models")
    log("   3. trainer-gui")
    log("   4. preprocessing")
    log("   5. detect-offline")
    log("   6. edit-config")
    log("   7. help")
    log("   8. exit")


def infer_source_kind(path: str, explicit: str) -> str:
    if explicit != "auto":
        return explicit
    source_path = Path(path)
    if source_path.is_dir():
        return "raw-dir"
    if source_path.suffix.lower() == ".csv":
        return "csv"
    raise ValueError(f"Cannot infer source kind for {path}. Use --source_kind.")


def run_detect_offline(
    *,
    models_dir: str,
    input_path: str,
    output_csv: str,
    source_kind: str,
    allow_legacy_simple_runtime: bool,
) -> None:
    # Phase A: load the selected runtime bundle before reading data so the user sees
    # exactly which model family is used for offline detection.
    runtime = load_runtime(models_dir, allow_legacy=allow_legacy_simple_runtime)
    resolved_kind = infer_source_kind(input_path, source_kind)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        ensure_dir(output_dir)

    # Phase B: load either an extracted CSV or raw BSM directory, then run the runtime's
    # full offline scoring path and write row-level results to disk.
    if resolved_kind == "raw-dir":
        raw_df = load_raw_bsm_directory(input_path)
        scored = runtime.score_dataframe(raw_df)
        scored.to_csv(output_csv, index=False)
    else:
        raw_df = pd.read_csv(input_path, low_memory=False)
        scored = runtime.score_dataframe(raw_df)
        scored.to_csv(output_csv, index=False)

    # Phase C: keep console output short for discussion use: family performance first,
    # then one row per vehicle with trust, p_final, and final decision.
    try:
        output_path = Path(output_csv)
        root_path = output_path.parent if str(output_path.parent) else Path(".")
        scorecard = _detection_scorecard_from_dataframe(output_path, root_path, scored)
        _log_detection_scorecard(scorecard)
        _log_vehicle_decision_table(scored)
    except Exception as exc:
        log(f"Current detection details skipped: {exc}")


def _read_detection_source(source_path: str, source_kind: str) -> pd.DataFrame:
    if source_kind == "raw-dir":
        return load_raw_bsm_directory(source_path)
    if source_kind == "csv":
        return pd.read_csv(source_path, low_memory=False) if Path(source_path).exists() else pd.DataFrame()
    raise ValueError(f"Unsupported source_kind={source_kind!r}")


def _existing_live_row_keys(output_csv: str) -> set[str]:
    path = Path(output_csv)
    if not path.exists():
        return set()
    try:
        existing = pd.read_csv(path, usecols=["row_key"], low_memory=False)
    except Exception:
        return set()
    return set(existing["row_key"].dropna().astype(str).tolist())


def run_detect_live(
    *,
    models_dir: str,
    source_path: str,
    output_csv: str,
    source_kind: str,
    poll_interval: float,
    max_polls: Optional[int],
    once: bool,
) -> None:
    # Phase A: use the same runtime-loading path as offline detection so live scoring
    # can use the current archived multi-family models or a complete release bundle.
    runtime = load_runtime(models_dir, allow_legacy=False)
    if not hasattr(runtime, "score_dataframe"):
        raise RuntimeError("detect-live requires a runtime with score_dataframe support.")
    resolved_kind = infer_source_kind(source_path, source_kind)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        ensure_dir(output_dir)

    # Phase B: poll the live source, score the full observed stream, and append only
    # newly observed row keys. This keeps trust and p_final compatible with offline scoring.
    emitted_keys = _existing_live_row_keys(output_csv)
    accumulated_output = pd.DataFrame()
    polls = 0
    while True:
        raw_df = _read_detection_source(source_path, resolved_kind)
        if not raw_df.empty:
            scored = runtime.score_dataframe(raw_df)
            if "row_key" not in scored.columns:
                raise RuntimeError("detect-live scored output is missing row_key.")
            new_rows = scored[~scored["row_key"].astype(str).isin(emitted_keys)].copy()
            if not new_rows.empty:
                emitted_keys.update(new_rows["row_key"].astype(str).tolist())
                write_header = not Path(output_csv).exists()
                new_rows.to_csv(output_csv, mode="a", header=write_header, index=False)
                accumulated_output = pd.concat([accumulated_output, new_rows], ignore_index=True)

        polls += 1
        if once or (max_polls is not None and polls >= max_polls):
            break
        time.sleep(max(poll_interval, 0.1))

    # Phase C: mirror detect-offline's short discussion output after the live run stops.
    if not accumulated_output.empty:
        try:
            output_path = Path(output_csv)
            root_path = output_path.parent if str(output_path.parent) else Path(".")
            scorecard = _detection_scorecard_from_dataframe(output_path, root_path, accumulated_output)
            _log_detection_scorecard(scorecard)
            _log_vehicle_decision_table(accumulated_output)
        except Exception as exc:
            log(f"Current detection details skipped: {exc}")
    log(f"Live detection wrote results to {output_csv}")


def run_train_rsu(
    *,
    csv_path: str,
    models_dir: str,
    train_family: str,
    window_size: int,
    seq_len: int,
) -> Dict[str, Any]:
    ensure_dir(models_dir)
    trainer = RSUMultiHeadTrainer(
        train_family=train_family,
        window_size=window_size,
        seq_len=seq_len,
    )
    try:
        result = trainer.fit_csv(csv_path, output_dir=models_dir)
    except ValueError as exc:
        if "requires at least" not in str(exc):
            raise
        log(f"Full sender-disjoint trainer could not run on this small scenario: {exc}")
        log("Falling back to small-scenario LightGBM training with the thesis feature pipeline.")
        report = train_scenario_lgbm_release(
            csv_path,
            output_dir=models_dir,
            family="auto" if train_family in {"binary", "all"} else train_family,
            window_size=window_size,
            seq_len=seq_len,
        )
        summary = {
            "output_dir": models_dir,
            "manifest_path": os.path.join(models_dir, "manifest.json"),
            "report_path": os.path.join(models_dir, "training_report.json"),
            "enabled_heads": [report.get("trained_family")],
            "threshold": report.get("threshold"),
            "training_protocol": report.get("training_protocol"),
            "validation_attack_f1": (
                report.get("validation_classification_report", {}).get("1", {}).get("f1-score")
            ),
            "warning": report.get("warning"),
        }
        log(json.dumps(summary, indent=2))
        return summary
    summary = {
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "report_path": result.report_path,
        "enabled_heads": result.enabled_heads,
        "threshold": result.threshold,
    }
    log(json.dumps(summary, indent=2))
    return summary


def build_verification_merge(labels_df: pd.DataFrame, detect_df: pd.DataFrame) -> pd.DataFrame:
    labels_norm = normalize_bsm_dataframe(labels_df)[["row_key", "label", "attack_id"]].copy()
    labels_norm["row_key"] = labels_norm["row_key"].astype(str)
    detect = detect_df.copy()
    if "row_key" not in detect.columns:
        detect = normalize_bsm_dataframe(detect)
    detect["row_key"] = detect["row_key"].astype(str)
    merged = detect.merge(labels_norm, on="row_key", how="left", suffixes=("", "_true"))
    if "label_true" not in merged.columns:
        merged["label_true"] = merged.get("label", 0)
    if "attack_id_true" not in merged.columns:
        merged["attack_id_true"] = merged.get("attack_id", 0)
    return merged


def run_verify(*, labels: str, detect_csv: str, outdir: str) -> Dict[str, Any]:
    ensure_dir(outdir)
    labels_df = pd.read_csv(labels, low_memory=False)
    detect_df = pd.read_csv(detect_csv, low_memory=False)
    merged = build_verification_merge(labels_df, detect_df)

    pred_col = "final_decision" if "final_decision" in merged.columns else "fused_anom"
    score_col = "p_final" if "p_final" in merged.columns else "fused_risk"

    y_true = pd.to_numeric(merged["label_true"], errors="coerce").fillna(0).astype(int)
    y_pred = pd.to_numeric(merged[pred_col], errors="coerce").fillna(0).astype(int)
    report = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred).tolist()

    per_attack = (
        merged.groupby("attack_id_true", dropna=False)
        .agg(
            rows=("row_key", "size"),
            attacks=("label_true", "sum"),
            predicted_attacks=(pred_col, "sum"),
            avg_score=(score_col, "mean"),
        )
        .reset_index()
        .sort_values("attack_id_true")
    )
    per_attack["attack_name"] = per_attack["attack_id_true"].map(ATTACK_TYPES).fillna("unknown")

    summary = {
        "detect_csv": detect_csv,
        "labels": labels,
        "rows": int(len(merged)),
        "confusion_matrix": cm,
        "classification_report": report,
    }

    with open(os.path.join(outdir, "verify_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    per_attack.to_csv(os.path.join(outdir, "verify_by_attack.csv"), index=False)
    merged.to_csv(os.path.join(outdir, "verify_merged.csv"), index=False)
    log(json.dumps(summary, indent=2))
    log(f"Verification artifacts saved in {outdir}")
    return summary


def generate_synthetic_dataset(
    *,
    n_cars: int = 32,
    n_msgs: int = 1400,
    seed: int = 7,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    senders = [f"car_{idx:03d}" for idx in range(n_cars)]
    rows: List[Dict[str, Any]] = []

    # Keep the built-in self-test thesis-grade: enough benign groups for a 10-way
    # sender split and enough positive groups per family for 5-fold OOF stacking.
    family_plan = {
        "pos_speed": [1, 3, 6, 9, 5],
        "replay_stale": [11, 12, 17, 11, 12],
        "dos": [13, 14, 15, 18, 19],
        "sybil": [16, 17, 18, 19, 16],
    }
    benign_count = max(10, n_cars - sum(len(ids) for ids in family_plan.values()))
    if benign_count + sum(len(ids) for ids in family_plan.values()) > n_cars:
        raise ValueError("Synthetic self-test requires at least 30 sender groups.")

    sender_attack: Dict[str, int] = {}
    sender_offset = 0
    for sender in senders[:benign_count]:
        sender_attack[sender] = 0
        sender_offset += 1
    for attack_ids in family_plan.values():
        for attack_id in attack_ids:
            sender_attack[senders[sender_offset]] = int(attack_id)
            sender_offset += 1
    for sender in senders[sender_offset:]:
        sender_attack[sender] = 0

    for sender in senders:
        t = 0.0
        x = float(rng.normal(0.0, 15.0))
        y = float(rng.normal(0.0, 15.0))
        speed = float(np.clip(rng.normal(17.0, 3.0), 3.0, 28.0))
        heading = float(rng.uniform(-math.pi, math.pi))
        attack_id = sender_attack[sender]

        for msg_idx in range(max(10, n_msgs // n_cars)):
            dt = float(np.clip(rng.normal(0.35, 0.08), 0.05, 1.25))
            t += dt
            heading += float(rng.normal(0.0, 0.03))
            accel = float(rng.normal(0.0, 0.5))
            speed = float(np.clip(speed + accel * dt, 0.0, 38.0))

            is_attack = attack_id != 0 and msg_idx > 5
            if is_attack:
                if attack_id in {1, 2, 3, 4, 5, 6, 7, 8, 9}:
                    speed += float(rng.normal(9.0, 3.0))
                    x += float(rng.normal(10.0, 2.0))
                if attack_id in {11, 12, 17}:
                    dt = 0.0 if msg_idx % 4 == 0 else dt
                if attack_id in {13, 14, 15, 18, 19}:
                    dt = float(np.clip(rng.normal(0.05, 0.01), 0.01, 0.10))
                if attack_id in {16, 17, 18, 19}:
                    sender_pseudo = f"{sender}_clone_{msg_idx % 3}"
                else:
                    sender_pseudo = sender
            else:
                sender_pseudo = sender

            x += speed * math.cos(heading) * dt
            y += speed * math.sin(heading) * dt
            rows.append(
                {
                    "receiver_pseudo": f"rsu_{msg_idx % 4}",
                    "sender_pseudo": sender_pseudo,
                    "t_curr": t,
                    "x_curr": x,
                    "y_curr": y,
                    "speed_curr": speed,
                    "acc_curr": accel,
                    "heading_curr": heading,
                    "pos_conf_x_curr": abs(rng.normal(0.7, 0.1)),
                    "pos_conf_y_curr": abs(rng.normal(0.7, 0.1)),
                    "spd_conf_x_curr": abs(rng.normal(0.5, 0.1)),
                    "spd_conf_y_curr": abs(rng.normal(0.5, 0.1)),
                    "acc_conf_x_curr": abs(rng.normal(0.3, 0.05)),
                    "acc_conf_y_curr": abs(rng.normal(0.3, 0.05)),
                    "head_conf_x_curr": abs(rng.normal(0.2, 0.05)),
                    "head_conf_y_curr": abs(rng.normal(0.2, 0.05)),
                    "mb_version": "V2",
                    "attack_id": int(attack_id),
                    "label": int(is_attack),
                }
            )

    return pd.DataFrame(rows)


def run_selftest(models_dir: str) -> None:
    ensure_dir(models_dir)
    train_csv = os.path.join(models_dir, "synthetic_train.csv")
    detect_csv = os.path.join(models_dir, "synthetic_detect.csv")
    out_csv = os.path.join(models_dir, "synthetic_detect_out.csv")

    train_df = generate_synthetic_dataset(seed=7)
    train_df.to_csv(train_csv, index=False)
    run_train_rsu(
        csv_path=train_csv,
        models_dir=models_dir,
        train_family="all",
        window_size=DEFAULT_WINDOW_SIZE,
        seq_len=DEFAULT_SEQ_LEN,
    )

    detect_df = generate_synthetic_dataset(seed=19)
    detect_df.to_csv(detect_csv, index=False)
    run_detect_offline(
        models_dir=models_dir,
        input_path=detect_csv,
        output_csv=out_csv,
        source_kind="csv",
        allow_legacy_simple_runtime=False,
    )
    run_verify(labels=detect_csv, detect_csv=out_csv, outdir=os.path.join(models_dir, "verify"))


def _ask(prompt: str, default: Optional[str] = None, *, required: bool = False) -> str:
    while True:
        if default is None:
            raw = input(f"{prompt}: ").strip()
        else:
            raw = input(f"{prompt} [{default}]: ").strip()
        if raw:
            return raw
        if default is not None:
            return default
        if not required:
            return ""


def open_config_in_editor(path: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    editor = os.environ.get("EDITOR")
    if editor:
        subprocess.run(shlex.split(editor) + [path], check=False)
        return
    xdg = shutil.which("xdg-open")
    if xdg:
        subprocess.Popen([xdg, path], start_new_session=True)
        return
    for candidate in ("gedit", "xed", "nano", "vi"):
        editor_path = shutil.which(candidate)
        if editor_path:
            subprocess.run([editor_path, path], check=False)
            return
    raise RuntimeError("No editor found.")


def interactive_menu() -> None:
    cli_defaults = {
        "f2md_dir": DEFAULT_F2MD_DIR,
        "results_dir": DEFAULT_RESULTS_DIR,
        "models_dir": DEFAULT_MODELS_DIR,
        "detect_models_dir": DEFAULT_DETECT_MODELS_DIR,
        "history_root": DEFAULT_ARCHIVE_ROOT,
        "live_ids_dir": DEFAULT_LIVE_IDS_DIR,
        "conda_env": DEFAULT_CONDA_ENV,
    }
    print_cli_home(
        f2md_dir=cli_defaults["f2md_dir"],
        results_dir=cli_defaults["results_dir"],
        models_dir=cli_defaults["detect_models_dir"],
        history_root=cli_defaults["history_root"],
    )
    while True:
        choice = input("\nvanet-cli> ").strip().lower()
        if not choice:
            continue

        if choice in {"help", "h", "?", "7"}:
            print_cli_home(
                f2md_dir=cli_defaults["f2md_dir"],
                results_dir=cli_defaults["results_dir"],
                models_dir=cli_defaults["detect_models_dir"],
                history_root=cli_defaults["history_root"],
            )
        elif choice in {"open-terminal", "open terminal", "terminal", "1"}:
            open_terminal_app(DEFAULT_APP_DIR)
        elif choice in {"start-live-ids", "live-ids", "99"}:
            run_live_ids_dashboard(cli_defaults["live_ids_dir"], cli_defaults["conda_env"])
        elif choice in {"list-models", "2"}:
            models_root = _ask("Models root", cli_defaults["history_root"])
            list_models_cli(models_root)
        elif choice in {"trainer-gui", "3"}:
            launch_rsu_trainer_gui(DEFAULT_RSU_TRAINER_PY)
        elif choice in {"preprocessing", "preprocess", "4"}:
            root = _ask("BSM root directory", os.path.join(cli_defaults["results_dir"], "LuSTNanoScenario-ITSG5"))
            out_csv = _ask("Output CSV path", os.path.join(cli_defaults["results_dir"], "features_intermessage_v2.csv"))
            version = _ask("BSM version", "v2")
            config_path = _ask("Scenario config path", DEFAULT_BAGHDAD_OMNETPP)
            run_preprocessing(
                results_dir=cli_defaults["results_dir"],
                root=root,
                out_csv=out_csv,
                version=version,
                config_path=config_path,
            )
        elif choice in {"detect-offline", "5"}:
            input_path = _ask("Input CSV or raw-dir", required=True)
            run_detect_offline(
                models_dir=cli_defaults["detect_models_dir"],
                input_path=input_path,
                output_csv=os.path.join(cli_defaults["results_dir"], "detect_offline.csv"),
                source_kind="auto",
                allow_legacy_simple_runtime=False,
            )
        elif choice in {"edit-config", "6"}:
            open_config_in_editor(DEFAULT_BAGHDAD_OMNETPP)
        elif choice in {"show-log", "stop-daemon", "stop-scenario", "stop-live-ids"}:
            log("Background service management is disabled. Use the opened terminal window directly.")
        elif choice in {"q", "quit", "exit", "8"}:
            return
        else:
            log("Unknown command. Type 'help' to view all one-terminal commands.")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "VANET IDS orchestrator with three explicit paths:\n"
            "1) offline dataset pipeline\n"
            "2) realtime pipeline with live F2MD scenario\n"
        "3) RSU multi-head trainer pipeline\n"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "mode",
        choices=[
            "cli",
            "menu",
            "open-terminal",
            "preprocessing",
            "train-rsu",
            "detect-live",
            "detect-offline",
            "verify",
            "list-models",
            "live-ids",
            "train",
            "detect",
            "selftest",
        ],
        help=(
            "cli/menu: one-terminal command-line control center\n"
            "menu: interactive menu\n"
            "open-terminal: open the OS QTerminal app in the project directory\n"
            "preprocessing: extract features and auto-apply attack_id from BaghdadScenario omnetpp.ini\n"
            "train-rsu: train thesis-aligned RSU multi-head models\n"
            "detect-live: watch a live F2MD source using trained RSU multi-head models\n"
            "detect-offline: score an extracted CSV or raw BSM directory offline\n"
            "verify: compare labels against a detection CSV\n"
            "list-models: list multi-head and legacy model directories\n"
            "live-ids: open the existing dashboard in a new terminal\n"
            "train/detect: backward-compatible aliases for train-rsu/detect-offline\n"
            "selftest: synthetic smoke test of the new multi-head path\n"
        ),
    )

    parser.add_argument("--input", help="Training input CSV or offline detection source.")
    parser.add_argument("--output", help="Detection output CSV.")
    parser.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)

    parser.add_argument("--f2md_dir", default=DEFAULT_F2MD_DIR)
    parser.add_argument("--results_dir", default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--live_ids_dir", default=DEFAULT_LIVE_IDS_DIR)
    parser.add_argument("--trainer_py", default=DEFAULT_RSU_TRAINER_PY)

    parser.add_argument("--preprocess_root", "--extract_root", dest="preprocess_root")
    parser.add_argument("--preprocess_out", "--extract_out", dest="preprocess_out")
    parser.add_argument("--preprocess_version", "--extract_version", dest="preprocess_version", default="v2")
    parser.add_argument("--scenario_config", default=DEFAULT_BAGHDAD_OMNETPP)

    parser.add_argument("--train_family", default="all", choices=["binary", "all", "pos_speed", "replay_stale", "dos", "sybil"])
    parser.add_argument("--window_size", type=int, default=DEFAULT_WINDOW_SIZE)
    parser.add_argument("--seq_len", type=int, default=DEFAULT_SEQ_LEN)
    parser.add_argument("--launch_gui", action="store_true", help="Open rsu_trainer_all_in_one_v7.py instead of inline training.")

    parser.add_argument("--source_kind", default="auto", choices=["auto", "csv", "raw-dir"])
    parser.add_argument("--poll_interval", type=float, default=2.0)
    parser.add_argument("--max_polls", type=int)
    parser.add_argument("--once", action="store_true")
    parser.add_argument(
        "--allow_legacy_simple_runtime",
        action="store_true",
        help="Allow old release_v2 RandomForest bundles in detect-offline.",
    )

    parser.add_argument("--labels")
    parser.add_argument("--detect_csv")
    parser.add_argument("--outdir", default="verify_output")
    parser.add_argument("--conda_env", default=DEFAULT_CONDA_ENV)
    return parser


def main() -> None:
    if len(sys.argv) == 1:
        print(ascii_art)
        interactive_menu()
        return

    args = build_arg_parser().parse_args()

    if args.mode in {"menu", "cli"}:
        print(ascii_art)
        interactive_menu()
        return

    if args.mode == "open-terminal":
        open_terminal_app(DEFAULT_APP_DIR)
        return

    if args.mode == "live-ids":
        run_live_ids_dashboard(args.live_ids_dir)
        return

    if args.mode == "preprocessing":
        root = args.preprocess_root or os.path.join(args.results_dir, "LuSTNanoScenario-ITSG5")
        out_csv = args.preprocess_out or os.path.join(args.results_dir, "features_intermessage_v2.csv")
        run_preprocessing(
            results_dir=args.results_dir,
            root=root,
            out_csv=out_csv,
            version=args.preprocess_version,
            config_path=args.scenario_config,
        )
        return

    if args.mode in {"train-rsu", "train"}:
        if args.launch_gui or not args.input:
            launch_rsu_trainer_gui(args.trainer_py)
            return
        run_train_rsu(
            csv_path=args.input,
            models_dir=args.models_dir,
            train_family=args.train_family,
            window_size=args.window_size,
            seq_len=args.seq_len,
        )
        return

    if args.mode in {"detect-offline", "detect"}:
        if not args.input:
            raise ValueError("detect-offline requires --input")
        models_dir = args.models_dir
        if models_dir == DEFAULT_MODELS_DIR and os.path.isdir(DEFAULT_DETECT_MODELS_DIR):
            models_dir = DEFAULT_DETECT_MODELS_DIR
        output_csv = args.output or os.path.join(args.results_dir, "detect_offline.csv")
        run_detect_offline(
            models_dir=models_dir,
            input_path=args.input,
            output_csv=output_csv,
            source_kind=args.source_kind,
            allow_legacy_simple_runtime=args.allow_legacy_simple_runtime,
        )
        return

    if args.mode == "detect-live":
        source_path = args.input or args.preprocess_root or os.path.join(args.results_dir, "LuSTNanoScenario-ITSG5")
        output_csv = args.output or os.path.join(
            args.results_dir,
            f"detect_live_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        )
        models_dir = args.models_dir
        if models_dir == DEFAULT_MODELS_DIR and os.path.isdir(DEFAULT_DETECT_MODELS_DIR):
            models_dir = DEFAULT_DETECT_MODELS_DIR
        run_detect_live(
            models_dir=models_dir,
            source_path=source_path,
            output_csv=output_csv,
            source_kind=args.source_kind,
            poll_interval=args.poll_interval,
            max_polls=args.max_polls,
            once=args.once,
        )
        return

    if args.mode == "verify":
        labels = args.labels or args.input
        detect_csv = args.detect_csv or args.output
        if not labels or not detect_csv:
            raise ValueError("verify mode requires --labels/--input and --detect_csv/--output")
        run_verify(labels=labels, detect_csv=detect_csv, outdir=args.outdir)
        return

    if args.mode == "list-models":
        root = args.models_dir if os.path.isdir(args.models_dir) else os.path.dirname(args.models_dir) or "."
        list_models_cli(root)
        return

    if args.mode == "selftest":
        run_selftest(args.models_dir)
        return

    raise ValueError(f"Unhandled mode: {args.mode}")


if __name__ == "__main__":
    main()
