#!/usr/bin/env bash
set -euo pipefail

usage() {
    printf 'usage: %s [--profile paper|smoke] [--plan] [--config CLUSTER_JSON] [--output NEW_DIR] [--figure FIGURE]\n' "$0"
    printf '       %s --resume RUN_DIR [--plan]\n' "$0"
}
config= output= profile= figure=all resume=
execute=(--execute)
while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --config|--output|--profile|--figure|--resume)
            if (($# < 2)) || [[ -z $2 || $2 == --* ]]; then usage >&2; exit 2; fi
            case "$1" in
                --config) config=$2 ;;
                --output) output=$2 ;;
                --profile) profile=$2 ;;
                --figure) figure=$2 ;;
                --resume) resume=$2 ;;
            esac
            shift 2 ;;
        --execute) execute=(--execute); shift ;;
        --plan) execute=(); shift ;;
        *) usage >&2; exit 2 ;;
    esac
done
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ -n $resume ]]; then
    if [[ -n $config$output$profile || -L $resume ]]; then usage >&2; exit 2; fi
    if [[ ! -f $resume/run-options.txt || ! -f $resume/configs.sha256 ]]; then
        usage >&2
        printf 'Resume metadata is missing; contact the authors for an older or incomplete run.\n' >&2
        exit 2
    fi
    mapfile -t saved < "$resume/run-options.txt"
    if [[ ${#saved[@]} != 2 ]] || [[ $figure != all && $figure != "${saved[1]}" ]]; then
        usage >&2; exit 2
    fi
    output=$resume profile=${saved[0]} figure=${saved[1]}
else
    config=${config:-eldr/inputs/cluster.json}
    profile=${profile:-paper}
    output=${output:-eldr/artifacts/runs/$figure-$profile-$(date -u +%Y%m%dT%H%M%S)-$BASHPID}
    if [[ ! -f $config || -e $output || -L $output ]]; then
        usage >&2
        printf 'Provide eldr/inputs/cluster.json (or --config) and a new output directory.\n' >&2
        exit 2
    fi
fi
case "$figure" in
    all|fig10_main_task|fig11_main_language|fig13_signature|fig14_cluster_balance|fig15_locality_band|fig16_prefix_cache) ;;
    *) usage >&2; exit 2 ;;
esac
if [[ $profile != paper && $profile != smoke ]]; then usage >&2; exit 2; fi
if ((${#execute[@]})); then
    exec 9>>/tmp/eldr-ae.lock
    if ! flock -n 9; then
        printf 'Another ELDR experiment holds this controller. Try after it finishes.\n' >&2
        exit 1
    fi
fi
export UV_CACHE_DIR=${UV_CACHE_DIR:-$PWD/eldr/artifacts/cache/uv}
export MPLCONFIGDIR=${MPLCONFIGDIR:-$PWD/eldr/artifacts/cache/matplotlib}
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
if [[ ! -x .venv/bin/python ]]; then uv venv --python 3.12 .venv; fi
uv pip install --python .venv/bin/python -r eldr/requirements.txt
if [[ -z $resume ]]; then
    .venv/bin/python -m eldr.experiments.runner configure \
        --inputs eldr/inputs --config "$config" --output "$output/configs"
fi
output=$(cd -- "$output" && pwd)
if [[ -z $resume ]]; then
    printf '%s\n' "$profile" "$figure" > "$output/run-options.txt"
    sha256sum "$output"/configs/*.json > "$output/configs.sha256"
else
    sha256sum --check --status "$output/configs.sha256"
fi
printf 'Run directory: %s\n' "$output"
trap 'status=$?; if ((status)); then printf "After resolving the error, resume with:\n  bash eldr/scripts/run_all.sh --resume %q\n" "$output" >&2; fi' EXIT
phases=(run)
pending=1
if [[ -n $resume ]]; then
    phases=(check)
    if ((${#execute[@]})); then phases+=(run); fi
    pending=0
fi
for phase in "${phases[@]}"; do
    if [[ $phase == run && $pending == 1 ]] && ((${#execute[@]})); then
        .venv/bin/python -m eldr.experiments.runner bootstrap --config "$output/configs/qwen-task.json"
    fi
    for entry in \
        fig10_main_task:task fig11_main_language:language \
        fig13_signature:task fig13_signature:language \
        fig14_cluster_balance:task fig14_cluster_balance:language \
        fig15_locality_band:task fig15_locality_band:language \
        fig16_prefix_cache:task-prefix; do
        current=${entry%%:*}
        if [[ $figure != all && $figure != "$current" ]]; then continue; fi
        workload=${entry#*:}
        destination="$output/$current"
        case "$current" in
            fig13_*|fig14_*|fig15_*) destination+="/$workload" ;;
        esac
        if [[ $phase == run && -n $resume && -f $destination/complete.json ]]; then
            printf 'SKIP (validated): %s / %s\n' "$current" "$workload"
            continue
        fi
        configs=()
        models=(qwen gptoss gemma)
        if [[ $current == fig16_prefix_cache ]]; then models=(gptoss); fi
        for model in "${models[@]}"; do
            configs+=(--config "$output/configs/$model-$workload.json")
        done
        module=${current#fig??_}
        if [[ $current == fig13_signature ]]; then module=signature_ablation; fi
        log="$output/$current-$workload.log"
        options=("${execute[@]}")
        if [[ -n $resume ]]; then options+=(--resume); fi
        if [[ $phase == check ]]; then
            options=(--resume)
            if [[ ! -f $destination/complete.json ]]; then pending=1; fi
        fi
        printf '%s / %s -> %s\n' "$current" "$workload" "$log"
        if .venv/bin/python -m "eldr.experiments.$module" "${configs[@]}" --profile "$profile" \
            --output "$destination" "${options[@]}" >> "$log" 2>&1; then
            printf 'OK (%s): %s / %s\n' "$phase" "$current" "$workload"
        else
            status=$?
            printf 'Stopped; see %s\n' "$log" >&2
            tail -n 30 -- "$log" >&2
            exit "$status"
        fi
    done
done
if ((${#execute[@]})); then
    printf 'Selected experiments completed. Raw results, summaries and plots: %s\n' "$output"
else
    printf 'Selected plans validated; no GPU experiments started. Configs and plans: %s\n' "$output"
fi
