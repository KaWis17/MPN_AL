#!/bin/bash
# Control script for running MPN_AL on PLGrid (Athena) from your own machine.
#
#   bash scripts/plgrid/remote.sh sync                          copy the code to ~/mpn_al on the cluster
#   bash scripts/plgrid/remote.sh setup                         create the Python environment (login node, needs internet)
#   bash scripts/plgrid/remote.sh download MODEL... [--verify]  fetch base-model weights (login node)
#   bash scripts/plgrid/remote.sh smoke [--no-wait] [OPTIONS]   sync, run the GPU smoke test, wait, fetch the report
#   bash scripts/plgrid/remote.sh submit [SBATCH OPTIONS] SCRIPT [ARGS]
#                                                               submit one of the scripts in scripts/ as a GPU job
#   bash scripts/plgrid/remote.sh run SCRIPT [ARGS]             run a LIGHT script on the login node (e.g. scripts/merge_shards.py)
#   bash scripts/plgrid/remote.sh status                        your jobs
#   bash scripts/plgrid/remote.sh logs [JOBID]                  tail of a job log
#   bash scripts/plgrid/remote.sh put LOCALPATH REMOTEDIR       copy a file or directory to the cluster (REMOTEDIR may use $SCRATCH)
#   bash scripts/plgrid/remote.sh fetch DIR [LOCALDIR [FILE...]]  copy results back
#   bash scripts/plgrid/remote.sh shell                         an interactive login
#
# Connection settings come from a .env file (default: .env in the repository root, or PLG_ENV_FILE):
# PLG_HOST, PLG_LOGIN, PLG_GRANT, PLG_PARTITION and, unless an SSH agent holds the key, PLG_IDENTITY_FILE
# (see scripts/plgrid/env.example). Only PLG_* and MPN_AL_* keys are read, the file is never executed, and
# values already in the environment win. MPN_AL_* variables are passed on to the cluster, for example
# MPN_AL_TORCH_INDEX for `setup`.
#
# On Windows run it from Git Bash: it calls the native OpenSSH client, which sees the Windows SSH agent
# (Git Bash's own ssh does not; see AL_System/docs/plgrid_connection_setup_windows.md). Connections are
# non-interactive (BatchMode), so a key with a passphrase has to be loaded into the agent first (ssh-add).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_FILE="${PLG_ENV_FILE:-$REPO_ROOT/.env}"
REMOTE_DIR="${MPN_AL_REMOTE_DIR:-mpn_al}"
LOCAL_OUT="${MPN_AL_LOCAL_OUT:-$REPO_ROOT/outputs/plgrid}"
SSH_EXE=""
SCP_EXE=""

die() {
    echo "remote.sh: $*" >&2
    exit 1
}

usage() {
    sed -n '3,/^set -euo/p' "${BASH_SOURCE[0]}" | sed '$d' | sed 's/^# \{0,1\}//'
}

load_env() {
    [ -f "$ENV_FILE" ] || die "$ENV_FILE not found: copy scripts/plgrid/env.example there and fill it in (or set PLG_ENV_FILE)"
    local line key value
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        case "$line" in
            PLG_*=*|MPN_AL_*=*)
                key="${line%%=*}"
                value="${line#*=}"
                case "$key" in *[!A-Za-z0-9_]*) continue ;; esac
                case "$value" in
                    \"*\") value="${value#\"}"; value="${value%\"}" ;;
                    \'*\') value="${value#\'}"; value="${value%\'}" ;;
                esac
                if [ -z "${!key:-}" ]; then
                    export "$key=$value"
                fi
                ;;
        esac
    done < "$ENV_FILE"
}

need() {
    local name
    for name in "$@"; do
        [ -n "${!name:-}" ] || die "$name is not set (put it in $ENV_FILE)"
    done
}

pick_clients() {
    if [ -n "${PLG_SSH:-}" ]; then
        SSH_EXE="$PLG_SSH"
        SCP_EXE="${PLG_SCP:-scp}"
    elif [ -x /c/Windows/System32/OpenSSH/ssh.exe ]; then
        SSH_EXE=/c/Windows/System32/OpenSSH/ssh.exe
        SCP_EXE=/c/Windows/System32/OpenSSH/scp.exe
    else
        SSH_EXE="$(command -v ssh || true)"
        SCP_EXE="$(command -v scp || true)"
    fi
    [ -n "$SSH_EXE" ] && [ -n "$SCP_EXE" ] || die "no ssh/scp client found"
}

validate_remote_dir() {
    case "$REMOTE_DIR" in
        ""|/*|*..*|*[!A-Za-z0-9._/-]*) die "MPN_AL_REMOTE_DIR must be a plain directory name relative to the home directory, got '$REMOTE_DIR'" ;;
    esac
}

ssh_run() {
    local options=(-o BatchMode=yes -o ConnectTimeout=30 -o ServerAliveInterval=30)
    if [ -n "${PLG_IDENTITY_FILE:-}" ]; then
        options+=(-i "$PLG_IDENTITY_FILE")
    fi
    "$SSH_EXE" "${options[@]}" "$PLG_LOGIN@$PLG_HOST" "$@"
}

scp_run() {
    local options=(-o BatchMode=yes -o ConnectTimeout=30)
    if [ -n "${PLG_IDENTITY_FILE:-}" ]; then
        options+=(-i "$PLG_IDENTITY_FILE")
    fi
    "$SCP_EXE" "${options[@]}" "$@"
}

# Runs a command string in a login shell on the cluster (so that `module` works).
remote_login() {
    ssh_run "bash -l -c $(printf '%q' "$1")"
}

# The same, from ~/$REMOTE_DIR (where `sync` puts the code).
remote_bash() {
    remote_login "cd ~/$REMOTE_DIR && $1"
}

# Double-quotes arguments for the remote shell: $VARIABLES in them are expanded there (so '$SCRATCH/x', in single
# quotes on your side, means the cluster's scratch), but command substitution never runs.
quote_remote() {
    local result="" arg
    for arg in "$@"; do
        arg="${arg//\\/\\\\}"
        arg="${arg//\"/\\\"}"
        arg="${arg//\`/\\\`}"
        arg="${arg//\$(/\\\$(}"
        result="$result \"$arg\""
    done
    printf '%s' "${result# }"
}

# `export NAME=value; ` for every MPN_AL_* variable, to prefix remote commands with.
forward_exports() {
    local name
    for name in $(compgen -v MPN_AL_); do
        printf 'export %s=%q; ' "$name" "${!name}"
    done
}

check_connection() {
    ssh_run true || die "cannot log in to $PLG_LOGIN@$PLG_HOST. Connections are non-interactive: load the key into the SSH agent (ssh-add), or set PLG_IDENTITY_FILE to a key without a passphrase."
}

cmd_sync() {
    need PLG_HOST PLG_LOGIN
    validate_remote_dir
    # Windows line endings in a shell script make it fail on the cluster with baffling errors. (grep cannot
    # be used to look for them: in Git Bash it drops the carriage return from its pattern and matches everything.)
    local crlf="" script
    for script in "$REPO_ROOT"/scripts/plgrid/*.sh "$REPO_ROOT"/scripts/plgrid/*.sbatch; do
        [ -f "$script" ] || continue
        if ! tr -d '\r' < "$script" | cmp -s - "$script"; then
            crlf="$crlf ${script#"$REPO_ROOT"/}"
        fi
    done
    [ -z "$crlf" ] || die "these scripts have Windows line endings and would fail on the cluster:$crlf. Fix with: sed -i 's/\\r\$//' FILE (.gitattributes keeps *.sh and *.sbatch on LF in a fresh checkout)"

    local items=() item
    for item in src scripts configs schemas tests docs pyproject.toml README.md .gitattributes; do
        if [ -e "$REPO_ROOT/$item" ]; then
            items+=("$item")
        fi
    done
    local archive
    archive="$(mktemp "${TMPDIR:-/tmp}/mpn_al_sync.XXXXXX")"
    tar czf "$archive" -C "$REPO_ROOT" --exclude='__pycache__' --exclude='*.pyc' --exclude='.pytest_cache' \
        --exclude='*.egg-info' "${items[@]}"
    echo "Copying $(du -h "$archive" | cut -f1) of code to ~/$REMOTE_DIR on $PLG_HOST ..."
    # Only the code directories this script itself deploys are replaced; logs/, results and the environment stay.
    ssh_run "mkdir -p ~/$REMOTE_DIR/logs && cd ~/$REMOTE_DIR && rm -rf src scripts configs schemas tests docs"
    scp_run "$archive" "$PLG_LOGIN@$PLG_HOST:$REMOTE_DIR/.sync.tgz"
    ssh_run "cd ~/$REMOTE_DIR && tar xzf .sync.tgz && rm -f .sync.tgz"
    rm -f "$archive"

    if [ -n "${PLG_GRANT:-}" ] && [ -n "${PLG_PARTITION:-}" ]; then
        local settings
        settings="$(mktemp "${TMPDIR:-/tmp}/mpn_al_env.XXXXXX")"
        printf 'PLG_GRANT=%s\nPLG_PARTITION=%s\n' "$PLG_GRANT" "$PLG_PARTITION" > "$settings"
        scp_run "$settings" "$PLG_LOGIN@$PLG_HOST:$REMOTE_DIR/.plgrid.env"
        rm -f "$settings"
    fi
    echo "Code synced to ~/$REMOTE_DIR."
}

cmd_setup() {
    remote_bash "$(forward_exports) bash scripts/plgrid/remote_setup.sh"
}

cmd_download() {
    [ $# -gt 0 ] || die "usage: download MODEL... [--verify]   (names from: python scripts/list_base_models.py)"
    remote_bash "$(forward_exports) source scripts/plgrid/setup_env.sh && python scripts/download_base_models.py $(printf '%q ' "$@")"
}

# submit_job "<sbatch options>" FILE [ARGS...]: submits scripts/plgrid/FILE from ~/$REMOTE_DIR, prints the job id.
submit_job() {
    need PLG_GRANT PLG_PARTITION
    local options="$1" file="$2"
    shift 2
    local output='logs/%x-%j.out'
    case " $options " in *" --array"*) output='logs/%x-%A_%a.out' ;; esac
    local id
    id="$(remote_bash "mkdir -p logs && sbatch --parsable -A $(printf '%q' "$PLG_GRANT") -p $(printf '%q' "$PLG_PARTITION") --output=$output $options $file $(quote_remote "$@")" | tail -n 1)" || die "sbatch failed"
    id="${id%%;*}"
    [ -n "$id" ] || die "sbatch printed no job id"
    echo "$id"
}

wait_for_job() {
    local job="$1" state last="" failures=0
    while :; do
        if state="$(remote_bash "squeue -h -j $job -o %T 2>/dev/null | head -n 1" 2>/dev/null)"; then
            failures=0
            [ -n "$state" ] || break
            if [ "$state" != "$last" ]; then
                echo "job $job: $state"
                last="$state"
            fi
        else
            failures=$((failures + 1))
            [ "$failures" -lt 10 ] || die "lost the connection to the cluster while waiting for job $job (it keeps running: remote.sh status)"
        fi
        sleep "${PLG_POLL_SECONDS:-30}"
    done
    remote_bash "sacct -j $job -X -n -P -o JobID,State,ExitCode,Elapsed 2>/dev/null | head -n 3" || true
}

# Everything after the script name goes to the script unchanged, except that $VARIABLES are expanded on the
# cluster: put them in single quotes, e.g. --out '$SCRATCH/mpn_al_runs/round_01'.
cmd_submit() {
    local options=()
    while [ $# -gt 0 ] && [ "${1#-}" != "$1" ]; do
        options+=("$1")
        shift
    done
    [ $# -gt 0 ] || die "usage: submit [SBATCH OPTIONS] SCRIPT [ARGS]   e.g. submit --array=0-7 scripts/segment_cells.py --head ..."
    local script="$1"
    shift
    [ -f "$REPO_ROOT/$script" ] || die "$script is not a file of this repository (give the path relative to the repository root)"
    local id
    id="$(submit_job "${options[*]:-}" scripts/plgrid/run_gpu.sbatch "$script" "$@")"
    echo "Submitted job $id (code as last synced: run 'sync' first if you changed it)."
    echo "Follow it with: bash scripts/plgrid/remote.sh status   |   logs $id"
}

# Light work only (merging shards, listing models): the login node is shared. Heavy work goes through `submit`.
# $VARIABLES in the arguments are expanded on the cluster, as in `submit`.
cmd_run() {
    [ $# -gt 0 ] || die "usage: run SCRIPT [ARGS]   e.g. run scripts/merge_shards.py embeddings '\$MPN_AL_RUNS/round_01/embeddings.parquet'"
    remote_bash "$(forward_exports) source scripts/plgrid/setup_env.sh >/dev/null && python $(quote_remote "$@")"
}

cmd_status() {
    remote_bash 'squeue --me -o "%.10i %.18j %.9T %.10M %.12P %R"; echo; echo "recent jobs:"; sacct -S now-2days -X -n -o JobID,JobName%18,State,Elapsed,ExitCode | tail -n 12'
}

cmd_logs() {
    local job="${1:-}"
    if [ -n "$job" ]; then
        remote_bash "tail -n 80 logs/*-$job.out logs/*-${job}_*.out 2>/dev/null"
    else
        remote_bash 'f="$(ls -t logs/*.out 2>/dev/null | head -n 1)"; if [ -n "$f" ]; then echo "== $f"; tail -n 80 "$f"; else echo "no logs yet"; fi'
    fi
}

# fetch REMOTE_DIR [LOCAL_DIR [FILE...]]: REMOTE_DIR may use $SCRATCH (quote it); FILEs are relative to it.
cmd_fetch() {
    [ $# -ge 1 ] || die "usage: fetch REMOTE_DIR [LOCAL_DIR [FILE...]]"
    local remote_dir="$1" local_dir
    local_dir="${2:-$LOCAL_OUT/$(basename "$remote_dir")}"
    shift
    [ $# -eq 0 ] || shift
    local files=("$@")
    [ ${#files[@]} -gt 0 ] || files=(.)
    mkdir -p "$local_dir"
    local scratch archive local_archive
    scratch="$(remote_login 'echo "${SCRATCH:-$HOME}"' | tail -n 1)"
    archive="$scratch/.mpn_al_fetch_$$.tgz"
    remote_login "cd \"$remote_dir\" && tar czf \"$archive\" --ignore-failed-read $(printf '%q ' "${files[@]}")" \
        || die "could not pack $remote_dir on the cluster"
    local_archive="$(mktemp "${TMPDIR:-/tmp}/mpn_al_fetch.XXXXXX")"
    scp_run "$PLG_LOGIN@$PLG_HOST:$archive" "$local_archive"
    tar xzf "$local_archive" -C "$local_dir"
    rm -f "$local_archive"
    remote_login "rm -f \"$archive\"" || true
    echo "Fetched $remote_dir into $local_dir"
}

# put LOCAL_PATH REMOTE_DIR: a directory is copied into REMOTE_DIR (its content), a file becomes REMOTE_DIR/<name>.
# Existing files of the same name are overwritten, nothing else is touched. For many gigabytes (whole slide
# images) use group storage and scp directly; this is for tiles, annotations and other modest amounts.
cmd_put() {
    [ $# -eq 2 ] || die "usage: put LOCAL_PATH REMOTE_DIR   (REMOTE_DIR may use \$SCRATCH; quote it so that your shell leaves it alone)"
    local source="$1" remote_dir="$2"
    [ -e "$source" ] || die "$source does not exist"
    local archive scratch remote_archive
    archive="$(mktemp "${TMPDIR:-/tmp}/mpn_al_put.XXXXXX")"
    if [ -d "$source" ]; then
        tar czf "$archive" -C "$source" .
    else
        tar czf "$archive" -C "$(dirname "$source")" "$(basename "$source")"
    fi
    echo "Copying $(du -h "$archive" | cut -f1) to $remote_dir on $PLG_HOST ..."
    scratch="$(remote_login 'echo "${SCRATCH:-$HOME}"' | tail -n 1)"
    remote_archive="$scratch/.mpn_al_put_$$.tgz"
    scp_run "$archive" "$PLG_LOGIN@$PLG_HOST:$remote_archive"
    rm -f "$archive"
    remote_login "mkdir -p \"$remote_dir\" && tar xzf \"$remote_archive\" -C \"$remote_dir\"; status=\$?; rm -f \"$remote_archive\"; exit \$status" \
        || die "could not unpack the files into $remote_dir"
    echo "Done: $remote_dir"
}

cmd_smoke() {
    local wait=1
    if [ "${1:-}" = "--no-wait" ]; then
        wait=0
        shift
    fi
    need PLG_GRANT PLG_PARTITION
    cmd_sync
    remote_bash 'source scripts/plgrid/setup_env.sh >/dev/null 2>&1; test -f "$MPN_AL_VENV/bin/activate"' \
        || die "no virtual environment on the cluster yet: run 'bash scripts/plgrid/remote.sh setup' first"
    local job
    job="$(submit_job "--job-name=mpn-smoke --time=${MPN_AL_SMOKE_TIME:-01:00:00}" scripts/plgrid/smoke_gpu.sbatch "$@")"
    echo "Submitted the smoke test as job $job."
    if [ "$wait" -eq 0 ]; then
        echo "Not waiting. Later: bash scripts/plgrid/remote.sh logs $job; the report will be in \$MPN_AL_RUNS/smoke-$job on the cluster."
        return 0
    fi
    wait_for_job "$job"
    echo "--- end of the job log"
    remote_bash "tail -n 25 logs/mpn-smoke-$job.out 2>/dev/null" || true
    local runs
    runs="$(remote_bash 'source scripts/plgrid/setup_env.sh >/dev/null 2>&1; echo "$MPN_AL_RUNS"' | tail -n 1)"
    cmd_fetch "$runs/smoke-$job" "$LOCAL_OUT/smoke-$job" smoke_report.json smoke_report.md pytest.txt
    echo
    if [ -f "$LOCAL_OUT/smoke-$job/smoke_report.md" ]; then
        sed -n '1,40p' "$LOCAL_OUT/smoke-$job/smoke_report.md"
    fi
    if grep -q '"ok": true' "$LOCAL_OUT/smoke-$job/smoke_report.json" 2>/dev/null; then
        echo "SMOKE TEST PASSED (full report: $LOCAL_OUT/smoke-$job/smoke_report.md)"
    else
        die "SMOKE TEST FAILED or produced no report: see $LOCAL_OUT/smoke-$job and the job log above"
    fi
}

cmd_shell() {
    local options=(-o ServerAliveInterval=30)
    if [ -n "${PLG_IDENTITY_FILE:-}" ]; then
        options+=(-i "$PLG_IDENTITY_FILE")
    fi
    exec "$SSH_EXE" "${options[@]}" "$PLG_LOGIN@$PLG_HOST"
}

main() {
    local command="${1:-help}"
    if [ $# -gt 0 ]; then
        shift
    fi
    case "$command" in
        help|-h|--help) usage; return 0 ;;
        sync|setup|download|smoke|submit|run|status|logs|put|fetch|shell) ;;
        *) usage >&2; die "unknown command '$command'" ;;
    esac
    load_env
    need PLG_HOST PLG_LOGIN
    validate_remote_dir
    pick_clients
    if [ "$command" != "shell" ]; then
        check_connection
    fi
    "cmd_$command" "$@"
}

main "$@"
