"""The PLGrid scripts, run under bash against a fake cluster: no network, no real ssh.

A stub `ssh` runs the "remote" command in a temporary home directory, a stub `scp` copies files in and out of
it, and stubs for sbatch, squeue, sacct and python record how they were called. This checks the shell logic
(file transfer, quoting of arguments, job-array options, waiting for a job, fetching results) that cannot be
tried without a cluster. It does not show that the cluster itself behaves as the stubs assume.
"""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PLGRID = REPO / "scripts" / "plgrid"


def find_bash():
    candidates = [
        os.environ.get("MPN_AL_BASH"),
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
        shutil.which("bash"),
    ]
    for candidate in candidates:
        # On Windows `bash` on PATH is often the WSL launcher, which cannot run these tests.
        if candidate and Path(candidate).exists() and "windowsapps" not in candidate.lower() and "system32" not in candidate.lower():
            return candidate
    return None


BASH = find_bash()
pytestmark = pytest.mark.skipif(BASH is None, reason="needs a POSIX bash (Git Bash on Windows)")


def posix(path) -> str:
    """Path in the notation bash expects (Git Bash: /c/Users/...)."""
    text = Path(path).resolve().as_posix()
    if os.name == "nt" and len(text) > 1 and text[1] == ":":
        text = "/" + text[0].lower() + text[2:]
    return text


STUBS = {
    "ssh": r"""#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in -o|-i) shift 2 ;; -*) shift ;; *) break ;; esac
done
shift
export HOME="$FAKE_HOME"
printenv WANDB_API_KEY >> "$FAKE_HOME/env_leak.txt" 2>/dev/null || true
cd "$HOME"
if [ $# -eq 0 ]; then exec bash -l; fi
exec bash -c "$*"
""",
    "scp": r"""#!/bin/bash
while [ $# -gt 0 ]; do
  case "$1" in -o|-i) shift 2 ;; -*) shift ;; *) break ;; esac
done
remote_path() { local p="${1#*:}"; case "$p" in /*) echo "$p" ;; *) echo "$FAKE_HOME/$p" ;; esac; }
case "$1" in
  *@*:*) cp "$(remote_path "$1")" "$2" ;;
  *) cp "$1" "$(remote_path "$2")" ;;
esac
""",
    "sbatch": r"""#!/bin/bash
printf '%s\n' "$@" > "$FAKE_HOME/sbatch_args.txt"
case " $* " in
  *smoke_gpu.sbatch*)
    run="$SCRATCH/mpn_al_runs/smoke-4242"
    mkdir -p "$run"
    printf '# fake smoke report\n' > "$run/smoke_report.md"
    printf '{"ok": true}\n' > "$run/smoke_report.json"
    printf '3 passed\n' > "$run/pytest.txt"
    ;;
esac
echo 4242
""",
    "squeue": r"""#!/bin/bash
n=$(cat "$FAKE_HOME/squeue_calls" 2>/dev/null || echo 0)
echo $((n + 1)) > "$FAKE_HOME/squeue_calls"
if [ "$n" -lt 1 ]; then echo RUNNING; fi
""",
    "sacct": "#!/bin/bash\necho '4242|COMPLETED|0:0|00:00:42'\n",
    "python": r"""#!/bin/bash
echo "$*" >> "$FAKE_HOME/python_calls.txt"
echo "offline=${HF_HUB_OFFLINE:-} omp=${OMP_NUM_THREADS:-}" >> "$FAKE_HOME/python_env.txt"
""",
}


class Cluster:
    def __init__(self, root: Path):
        self.home = root / "remote_home"
        self.bin = root / "fakebin"
        self.local_out = root / "local_out"
        self.env_file = root / "plg.env"
        activate = self.home / "scratch" / "mpn_al-venv" / "bin" / "activate"
        activate.parent.mkdir(parents=True)
        activate.write_text("", encoding="utf-8")
        self.bin.mkdir()
        for name, body in STUBS.items():
            (self.bin / name).write_bytes(body.encode())
        (self.home / ".bash_profile").write_text(
            'export PATH="$FAKE_BIN:$PATH"\nexport SCRATCH="$FAKE_HOME/scratch"\n', encoding="utf-8"
        )
        self.env_file.write_text(
            "# test settings\nPLG_HOST=cluster.invalid\nPLG_LOGIN=tester\nPLG_GRANT=grant-gpu\n"
            'PLG_PARTITION="part-gpu"\nWANDB_API_KEY=must-never-be-read\n',
            encoding="utf-8",
        )
        self.env = {
            "PLG_ENV_FILE": posix(self.env_file),
            "PLG_SSH": posix(self.bin / "ssh"),
            "PLG_SCP": posix(self.bin / "scp"),
            "PLG_POLL_SECONDS": "0",
            "MPN_AL_LOCAL_OUT": posix(self.local_out),
            "FAKE_HOME": posix(self.home),
            "FAKE_BIN": posix(self.bin),
        }
        for name in ("PLG_HOST", "PLG_LOGIN", "PLG_GRANT", "PLG_PARTITION", "PLG_IDENTITY_FILE", "WANDB_API_KEY"):
            os.environ.pop(name, None)

    def bash(self, *command, cwd=REPO, env=None):
        environment = {**os.environ, **self.env, **(env or {})}
        return subprocess.run(
            [BASH, *map(str, command)], cwd=cwd, env=environment, capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=180,
        )

    def run(self, *args, script=None, env=None):
        return self.bash(script or "scripts/plgrid/remote.sh", *args, env=env)

    def python_calls(self):
        path = self.home / "python_calls.txt"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []


@pytest.fixture()
def cluster(tmp_path):
    return Cluster(tmp_path)


SCRIPTS = sorted(PLGRID.glob("*.sh")) + sorted(PLGRID.glob("*.sbatch"))


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda p: p.name)
def test_shell_scripts_parse_and_have_unix_line_endings(path):
    assert b"\r" not in path.read_bytes(), "CRLF would break the script on the cluster"
    result = subprocess.run([BASH, "-n", f"scripts/plgrid/{path.name}"], cwd=REPO, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr


def test_help_lists_every_command_and_unknown_ones_fail(cluster):
    result = cluster.run("help")
    assert result.returncode == 0
    for command in ("sync", "setup", "download", "smoke", "submit", "run", "status", "logs", "put", "fetch", "shell"):
        assert f"remote.sh {command}" in result.stdout
    unknown = cluster.run("frobnicate")
    assert unknown.returncode != 0 and "unknown command 'frobnicate'" in unknown.stderr


def test_a_missing_settings_file_is_explained(cluster):
    result = cluster.run("sync", env={"PLG_ENV_FILE": "/nonexistent/plg.env"})
    assert result.returncode != 0 and "env.example" in result.stderr


def test_sync_replaces_the_code_directories_and_keeps_everything_else(cluster):
    remote = cluster.home / "mpn_al"
    (remote / "src").mkdir(parents=True)
    (remote / "src" / "stale_module.py").write_text("old")
    (remote / "logs").mkdir()
    (remote / "logs" / "keep.out").write_text("a job log")
    result = cluster.run("sync")
    assert result.returncode == 0, result.stdout + result.stderr
    assert (remote / "src" / "embeddings" / "registry.py").exists()
    assert (remote / "scripts" / "plgrid" / "remote.sh").exists() and (remote / "pyproject.toml").exists()
    assert not (remote / "src" / "stale_module.py").exists(), "stale code must not survive a sync"
    assert (remote / "logs" / "keep.out").exists(), "logs are not code"
    assert not list(remote.rglob("__pycache__")) and not (remote / ".sync.tgz").exists()
    assert (remote / ".plgrid.env").read_text() == "PLG_GRANT=grant-gpu\nPLG_PARTITION=part-gpu\n"  # quotes in .env removed
    assert (cluster.home / "env_leak.txt").read_text() == "", "keys other than PLG_* and MPN_AL_* must not be read"


def test_sync_refuses_scripts_with_windows_line_endings(cluster, tmp_path):
    fake_repo = tmp_path / "repo"
    shutil.copytree(PLGRID, fake_repo / "scripts" / "plgrid", ignore=shutil.ignore_patterns("__pycache__"))
    broken = fake_repo / "scripts" / "plgrid" / "setup_env.sh"
    broken.write_bytes(broken.read_bytes().replace(b"\n", b"\r\n"))
    result = cluster.run("sync", script=posix(fake_repo / "scripts" / "plgrid" / "remote.sh"))
    assert result.returncode != 0 and "Windows line endings" in result.stderr and "setup_env.sh" in result.stderr
    assert not (cluster.home / "mpn_al").exists(), "nothing may be copied when the scripts are broken"


def test_submit_passes_options_and_keeps_arguments_with_spaces_intact(cluster):
    (cluster.home / "mpn_al").mkdir()
    result = cluster.run(
        "submit", "--array=0-3", "--time=00:30:00", "scripts/segment_cells.py", "--head", "dir with space/head.pt", "--out", "x"
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Submitted job 4242" in result.stdout
    assert (cluster.home / "sbatch_args.txt").read_text().splitlines() == [
        "--parsable", "-A", "grant-gpu", "-p", "part-gpu", "--output=logs/%x-%A_%a.out", "--array=0-3", "--time=00:30:00",
        "scripts/plgrid/run_gpu.sbatch", "scripts/segment_cells.py", "--head", "dir with space/head.pt", "--out", "x",
    ]
    assert cluster.run("submit", "scripts/extract_embeddings.py", "--base-model", "phikon-v2").returncode == 0
    arguments = (cluster.home / "sbatch_args.txt").read_text().splitlines()
    assert "--output=logs/%x-%j.out" in arguments and "--array=0-3" not in arguments
    assert cluster.run("submit", "scripts/no_such_script.py").returncode != 0
    assert cluster.run("submit", "--array=0-3").returncode != 0  # a script is required


def test_submit_expands_variables_on_the_cluster_and_never_runs_commands(cluster):
    (cluster.home / "mpn_al").mkdir()
    marker = cluster.home / "pwned"
    result = cluster.run(
        "submit", "scripts/segment_cells.py", "--out", "$SCRATCH/runs/x", "--note", f"$(touch {posix(marker)})",
        "--quote", 'say "hi" `id` \\ done',
    )
    assert result.returncode == 0, result.stdout + result.stderr
    arguments = (cluster.home / "sbatch_args.txt").read_text().splitlines()
    assert arguments[-5] == posix(cluster.home / "scratch") + "/runs/x"  # $SCRATCH was the cluster's, not ours
    assert arguments[-3] == f"$(touch {posix(marker)})" and not marker.exists()
    assert arguments[-1] == 'say "hi" `id` \\ done'


def test_run_executes_a_light_script_with_cluster_variables(cluster):
    assert cluster.run("sync").returncode == 0  # the script sources the environment file that sync deploys
    result = cluster.run("run", "scripts/merge_shards.py", "embeddings", "$MPN_AL_RUNS/round_01/embeddings.parquet")
    assert result.returncode == 0, result.stdout + result.stderr
    expected = posix(cluster.home / "scratch") + "/mpn_al_runs/round_01/embeddings.parquet"
    assert cluster.python_calls()[-1] == f"scripts/merge_shards.py embeddings {expected}"
    assert cluster.run("run").returncode != 0  # a script is required


def test_download_forwards_the_model_names(cluster):
    assert cluster.run("sync").returncode == 0
    result = cluster.run("download", "phikon-v2", "lunit-dino-s16", "--verify")
    assert result.returncode == 0, result.stdout + result.stderr
    assert cluster.python_calls()[-1] == "scripts/download_base_models.py phikon-v2 lunit-dino-s16 --verify"
    assert cluster.run("download").returncode != 0  # at least one model is needed


def test_setup_installs_the_package_and_honours_the_torch_settings(cluster):
    assert cluster.run("sync").returncode == 0
    result = cluster.run(
        "setup", env={"MPN_AL_TORCH_INDEX": "https://example.invalid/whl/cu128", "MPN_AL_TORCH_SPEC": "torch==9.9.9"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    calls = cluster.python_calls()
    assert "-m pip install --index-url https://example.invalid/whl/cu128 torch==9.9.9 torchvision" in calls
    assert "-m pip install -e .[hf,wsi,dev]" in calls
    assert "Environment ready" in result.stdout


def test_smoke_runs_the_whole_flow_and_fetches_the_report(cluster):
    result = cluster.run("smoke", "--models", "lunit-dino-s16")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "job 4242: RUNNING" in result.stdout and "4242|COMPLETED|0:0|00:00:42" in result.stdout
    assert "SMOKE TEST PASSED" in result.stdout
    fetched = cluster.local_out / "smoke-4242"
    assert (fetched / "smoke_report.md").read_text().startswith("# fake smoke report")
    assert (fetched / "smoke_report.json").exists() and (fetched / "pytest.txt").read_text() == "3 passed\n"
    arguments = (cluster.home / "sbatch_args.txt").read_text().splitlines()
    assert "--job-name=mpn-smoke" in arguments and "scripts/plgrid/smoke_gpu.sbatch" in arguments
    assert arguments[-2:] == ["--models", "lunit-dino-s16"]
    assert not list((cluster.home / "scratch").glob(".mpn_al_fetch_*")), "the temporary archive must be removed"


def test_smoke_without_the_environment_says_what_to_do_first(cluster):
    (cluster.home / "scratch" / "mpn_al-venv" / "bin" / "activate").unlink()
    result = cluster.run("smoke")
    assert result.returncode != 0 and "remote.sh setup" in result.stderr
    assert not (cluster.home / "sbatch_args.txt").exists(), "nothing may be submitted without an environment"


def test_a_failed_smoke_test_fails_the_command(cluster):
    stub = cluster.bin / "sbatch"
    stub.write_bytes(stub.read_bytes().replace(b'{"ok": true}', b'{"ok": false}'))
    result = cluster.run("smoke")
    assert result.returncode != 0 and "SMOKE TEST FAILED" in result.stderr


def test_fetch_logs_and_status(cluster):
    run = cluster.home / "scratch" / "runs" / "x"
    (run / "sub").mkdir(parents=True)
    (run / "a.txt").write_text("A")
    (run / "sub" / "b.txt").write_text("B")
    (cluster.home / "mpn_al" / "logs").mkdir(parents=True)
    (cluster.home / "mpn_al" / "logs" / "mpn-al-77.out").write_text("hello from the job\n")

    fetched = cluster.run("fetch", "$SCRATCH/runs/x")  # $SCRATCH is expanded on the cluster
    assert fetched.returncode == 0, fetched.stdout + fetched.stderr
    assert (cluster.local_out / "x" / "a.txt").read_text() == "A" and (cluster.local_out / "x" / "sub" / "b.txt").read_text() == "B"
    only = cluster.run("fetch", "$SCRATCH/runs/x", posix(cluster.local_out / "y"), "sub/b.txt", "missing.txt")
    assert only.returncode == 0 and (cluster.local_out / "y" / "sub" / "b.txt").exists() and not (cluster.local_out / "y" / "a.txt").exists()

    assert "hello from the job" in cluster.run("logs", "77").stdout
    assert "hello from the job" in cluster.run("logs").stdout  # no job id: the newest log
    status = cluster.run("status")
    assert status.returncode == 0 and "recent jobs:" in status.stdout


def test_put_copies_a_directory_or_a_file_to_the_cluster(cluster, tmp_path):
    data = tmp_path / "data"
    (data / "sub").mkdir(parents=True)
    (data / "a.txt").write_text("A")
    (data / "sub" / "b.txt").write_text("B")
    result = cluster.run("put", posix(data), "$SCRATCH/uploads/demo")  # needs no synced code on the cluster
    assert result.returncode == 0, result.stdout + result.stderr
    target = cluster.home / "scratch" / "uploads" / "demo"
    assert (target / "a.txt").read_text() == "A" and (target / "sub" / "b.txt").read_text() == "B"

    single = tmp_path / "one.txt"
    single.write_text("1")
    assert cluster.run("put", posix(single), "$SCRATCH/uploads/files").returncode == 0
    assert (cluster.home / "scratch" / "uploads" / "files" / "one.txt").read_text() == "1"
    assert cluster.run("put", posix(tmp_path / "missing"), "$SCRATCH/uploads/x").returncode != 0
    assert cluster.run("put", posix(single)).returncode != 0  # the destination is required
    assert not list((cluster.home / "scratch").glob(".mpn_al_put_*")), "the temporary archive must be removed"


def wrapper(cluster, script, *args, env=None):
    (cluster.home / "python_calls.txt").unlink(missing_ok=True)
    return cluster.bash(
        "scripts/plgrid/run_gpu.sbatch", script, *args,
        env={
            "SCRATCH": posix(cluster.home / "scratch"), "SLURM_SUBMIT_DIR": posix(REPO), "SLURM_CPUS_PER_TASK": "8",
            "PATH": str(cluster.bin) + os.pathsep + os.environ["PATH"], **(env or {}),
        },
    )


def test_gpu_job_wrapper_adds_options_only_where_the_script_knows_them(cluster):
    array = {"SLURM_ARRAY_TASK_ID": "2", "SLURM_ARRAY_TASK_COUNT": "4"}
    result = wrapper(cluster, "scripts/list_base_models.py", "--check", "x")
    assert result.returncode == 0, result.stdout + result.stderr
    assert cluster.python_calls() == ["scripts/list_base_models.py --check x"]  # knows no --num-workers
    assert "offline=1 omp=8" in (cluster.home / "python_env.txt").read_text()  # compute nodes have no internet

    wrapper(cluster, "scripts/segment_cells.py", "--head", "h", env=array)
    assert cluster.python_calls() == ["scripts/segment_cells.py --head h --num-workers 6 --shard-index 2 --num-shards 4"]
    wrapper(cluster, "scripts/extract_embeddings.py", "--base-model", "m")
    assert cluster.python_calls() == ["scripts/extract_embeddings.py --base-model m --num-workers 6"]  # not an array: no shards
    wrapper(cluster, "scripts/train_segmentation_head.py", "--out", "o", env=array)
    assert cluster.python_calls() == ["scripts/train_segmentation_head.py --out o --num-workers 6"]  # one job, never sharded

    broken = wrapper(cluster, "scripts/segment_cells.py", "--head", "h", env={"SLURM_ARRAY_TASK_ID": "2"})
    assert broken.returncode != 0 and "numbered 0..N-1" in broken.stderr
