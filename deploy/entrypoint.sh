#!/bin/sh
# Two jobs, then get out of the way.
#
# 1. Size the backend's thread pool to the CPUs this container was actually
#    given. llama.cpp defaults to the number of cores it can *see*, which on a
#    shared host is the host's core count, not the quota: it then oversubscribes
#    the quota and every forward pass pays for the contention. On a 4-CPU
#    machine that is the difference between a stable step time and a bimodal
#    one, and a bimodal step time is a bimodal ITL.
#
# 2. `exec`, so the server is PID 1 and SIGTERM reaches it directly. Without
#    this the shell is PID 1, swallows the signal, and the graceful drain in
#    cadence.api.app never runs -- the platform waits out its kill timeout and
#    then SIGKILLs a process holding a batch of half-finished streams.
set -eu

if [ -z "${CADENCE_N_THREADS:-}" ]; then
    quota=""
    # cgroup v2, then v1.
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        read -r q p < /sys/fs/cgroup/cpu.max || true
        [ "${q:-max}" != "max" ] && quota=$(( (q + p - 1) / p ))
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]; then
        q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        p=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        [ "$q" -gt 0 ] && quota=$(( (q + p - 1) / p ))
    fi
    [ -z "$quota" ] && quota=$(nproc 2>/dev/null || echo 2)
    [ "$quota" -lt 1 ] && quota=1
    export CADENCE_N_THREADS="$quota"
fi

echo "cadence: starting with CADENCE_N_THREADS=${CADENCE_N_THREADS}" >&2
exec "$@"
