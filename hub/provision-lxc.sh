#!/bin/sh
# provision-lxc.sh — one-time logging hygiene for systemd LXC inference hosts
#
# WHY
# Debian/Ubuntu LXC images ship BOTH systemd-journald and rsyslog. The
# rsyslogd AppArmor profile predates systemd's current socket layout: it
# permits the legacy /dev/log but NOT /run/systemd/journal/dev-log (where
# /dev/log now points). Every sendmsg() rsyslog sends to the journal
# socket is therefore DENIED and generates an AppArmor audit record — a
# flood (tens/minute) that:
#   * spams the host's kernel log (audit records land in the host's
#     journal, namespace-labelled for the container), and
#   * continuously flushes the host's dmesg ring buffer, so real events
#     (USB attach, OOM, NVMe errors) never persist in it.
# Aggravator: AppArmor profiles are inherited across fork/exec. A process
# spawned from rsyslogd that execs a binary with no profile of its own
# (observed: systemd-journald) keeps the rsyslogd profile and keeps
# spewing audit records even after rsyslog is stopped. Only a full
# container stop/start clears that.
#
# FIX
# journald becomes the single logger. /dev/log already points at the
# journal socket, so every syslog(3) caller still reaches journald. We
# purge rsyslog (which removes the offending profile) and persist the
# journal to disk with a cap, so logs survive a reboot without filling
# a small rootfs.
#
# USAGE
#   pct exec <vmid> -- <path-to-this-script>     # run inside the LXC
#   pct stop <vmid> && pct start <vmid>          # so the kernel drops
#                                                 # the loaded rsyslogd profile
#
# Idempotent — safe to re-run.

set -eu

if dpkg -s rsyslog >/dev/null 2>&1; then
    echo "purging rsyslog (removes the stale AppArmor profile)..."
    apt-get purge -y rsyslog
fi

# Belt and braces: even if a package drag pulls rsyslog back in, the unit
# stays off.
systemctl disable rsyslog 2>/dev/null || true
systemctl mask rsyslog 2>/dev/null || true

# Persistent journal with a cap (32 GB rootfs here; 100 M of logs is ample
# and can't eat the disk).
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\nSystemMaxUse=100M\n' \
    > /etc/systemd/journald.conf.d/llmlab-persistent.conf

echo "OK: rsyslog purged; journald persistent (100M cap) in" \
    "/etc/systemd/journald.conf.d/llmlab-persistent.conf"
echo "Next: stop+start the container so the kernel drops the loaded" \
    "rsyslogd profile (aa-status inside the CT should no longer list it)."
