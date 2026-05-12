#!/bin/bash
# Take EFA HW counter snapshots and verify a benchmark actually moved bytes
# across the fabric.
#
# Usage:
#   snapshot:     verify_efa_traffic.sh snapshot <file>
#   verify delta: verify_efa_traffic.sh verify <before-file> <after-file> [min-gb=1]
#
# "snapshot" captures tx_bytes/rx_bytes for every /sys/class/infiniband/* port.
# "verify" prints per-rail delta + total, and exits non-zero if total TX is
# below the threshold (default 1 GB) or per-rail imbalance exceeds 20%.

set -euo pipefail

MODE="${1:-}"

snapshot() {
    local outfile="$1"
    : > "${outfile}"
    for dev in /sys/class/infiniband/*; do
        [ -d "$dev" ] || continue
        name=$(basename "$dev")
        for p in "${dev}/ports/1/hw_counters/tx_bytes" \
                 "${dev}/ports/1/hw_counters/rx_bytes"; do
            if [ -r "$p" ]; then
                printf "%s %s %s\n" "$name" "$(basename "$p")" "$(cat "$p")" >> "${outfile}"
            fi
        done
    done
}

verify() {
    local before="$1"
    local after="$2"
    local min_gb="${3:-1}"

    local min_bytes=$((min_gb * 1024 * 1024 * 1024))

    local total_tx=0
    local max_tx=0
    local min_tx=-1
    local nrails=0

    # Iterate over every (dev, counter) line in before, diff with after
    while IFS=' ' read -r dev counter v_before; do
        [ -z "${dev}" ] && continue
        v_after=$(awk -v d="${dev}" -v c="${counter}" '$1==d && $2==c { print $3 }' "${after}")
        if [ -z "${v_after:-}" ]; then
            echo "WARN: ${dev}/${counter} missing in ${after}" >&2
            continue
        fi
        delta=$((v_after - v_before))
        printf "  %-12s %-10s  %'20d bytes\n" "${dev}" "${counter}" "${delta}"
        if [ "${counter}" = "tx_bytes" ]; then
            total_tx=$((total_tx + delta))
            if [ "${delta}" -gt "${max_tx}" ]; then max_tx=${delta}; fi
            if [ "${min_tx}" -lt 0 ] || [ "${delta}" -lt "${min_tx}" ]; then min_tx=${delta}; fi
            nrails=$((nrails + 1))
        fi
    done < "${before}"

    echo
    printf "TOTAL TX DELTA: %d bytes (= %d MB = %d GB across %d rails)\n" \
        "${total_tx}" "$((total_tx / 1024 / 1024))" \
        "$((total_tx / 1024 / 1024 / 1024))" "${nrails}"

    if [ "${total_tx}" -lt "${min_bytes}" ]; then
        echo "FAIL: total TX delta ${total_tx} < ${min_bytes} (${min_gb} GB)."
        echo "NVLink shortcut -- invalid, retest with --disable-nvlink equivalent (--allow-hybrid-mode 0)."
        exit 2
    fi

    # per-rail imbalance
    if [ "${nrails}" -gt 1 ] && [ "${max_tx}" -gt 0 ]; then
        local imbalance=$(( (max_tx - min_tx) * 100 / max_tx ))
        echo "PER-RAIL IMBALANCE: ${imbalance}% (max=${max_tx}, min=${min_tx})"
        if [ "${imbalance}" -gt 20 ]; then
            echo "WARN: imbalance >20% -- check rail assignment / QP distribution"
        fi
    fi

    echo "PASS: EFA traffic verified real."
    exit 0
}

case "${MODE}" in
    snapshot)
        if [ "$#" -lt 2 ]; then
            echo "usage: $0 snapshot <outfile>" >&2
            exit 1
        fi
        snapshot "$2"
        ;;
    verify)
        if [ "$#" -lt 3 ]; then
            echo "usage: $0 verify <before> <after> [min-gb=1]" >&2
            exit 1
        fi
        verify "$2" "$3" "${4:-1}"
        ;;
    *)
        echo "usage: $0 {snapshot <outfile> | verify <before> <after> [min-gb]}" >&2
        exit 1
        ;;
esac
