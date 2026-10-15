#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
name="${1:-tasks234}"
root="${WORK:?WORK is not defined}/Agentic-Environment/campaigns/${name}"
waves="${2:-10}"
lines_per_run=6
runs_per_wave=44
if [[ "${name}" != "tasks234" ]]; then
    runs_per_wave=4
fi
now="$(date +%s)"

fmt() {
    local s="$1"
    if (( s < 0 )); then s=0; fi
    printf '%dh%02dm' $(( s / 3600 )) $(( s % 3600 / 60 ))
}

echo "$(date '+%d/%m %H:%M')  campaign ${name}  (${waves} waves per model, ${runs_per_wave} runs per wave)"
echo
for dir in "${root}"/*/; do
    [[ -d "${dir}" ]] || continue
    model="$(basename "${dir}")"
    state="not in queue"
    wave_job=""
    if [[ -s "${dir}/jobs.txt" ]]; then
        read -r _ wave_job _ <"${dir}/jobs.txt" || true
        ids="$(tr -s ' ' ',' <"${dir}/jobs.txt" | sed 's/,$//')"
        state="$(squeue -h -j "${ids}" -o '%j %T %M' 2>/dev/null | sed 's/^aq-//' | tr '\n' ' ')"
        [[ -n "${state}" ]] || state="finished"
    fi
    done_waves=0
    total=0
    current=""
    for wave in "${dir}"wave_*/; do
        [[ -f "${wave}gpus.txt" ]] || continue
        start="$(stat -c %Y "${wave}gpus.txt")"
        if [[ -f "${wave}.complete.json" ]]; then
            done_waves=$(( done_waves + 1 ))
            total=$(( total + $(stat -c %Y "${wave}.complete.json") - start ))
        else
            current="${wave}"
            current_start="${start}"
        fi
    done
    line="${model}: ${done_waves}/${waves} waves done"
    avg=0
    if (( done_waves > 0 )); then
        avg=$(( total / done_waves ))
        line+=", $(fmt "${avg}") per wave"
    fi
    elapsed=0
    if [[ -n "${current}" ]]; then
        results=0
        if [[ -f "${current}results.jsonl" ]]; then
            results="$(wc -l <"${current}results.jsonl")"
        fi
        runs=$(( results / lines_per_run ))
        elapsed=$(( now - current_start ))
        line+=", $(basename "${current}") at ${runs}/${runs_per_wave} runs after $(fmt "${elapsed}")"
        if (( avg == 0 && runs > 0 )); then
            avg=$(( elapsed * runs_per_wave / runs ))
        fi
    fi
    left=$(( waves - done_waves ))
    if [[ "${state}" == "finished" && left -gt 0 ]]; then
        echo "${line}, STOPPED"
        if [[ -n "${wave_job}" ]]; then
            echo "    slurm: $(sacct -X -n -j "${wave_job}" --format=State%14,Elapsed%12 2>/dev/null | tr -s ' ' | tr '\n' ' ')"
            log="${repo_dir}/leonardo/logs/wave-${wave_job}_1.err"
            if [[ -s "${log}" ]]; then
                tail -n 4 "${log}" | cut -c 1-160 | sed 's/^/    log: /'
            fi
        fi
        continue
    fi
    if (( left > 0 && avg > 0 )); then
        line+=", about $(fmt $(( left * avg - elapsed ))) left"
    fi
    echo "${line}"
    echo "    jobs: ${state}"
done
echo
echo "Results so far: ./leonardo/digest.sh ${name}"
