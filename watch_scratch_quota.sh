#!/bin/bash
# Watch the shared gl-scratch fileset quota and email on trouble.
#
# WHY: the jjparkcv_root fileset is shared across ~12 users and swings by TBs without
# warning (observed 62 GiB -> 927 GiB -> 4.4 TiB inside four hours). A full fileset kills
# training mid-save. User cron is blocked on Great Lakes ("not allowed to use this
# program"), so this is a plain poll loop meant to run in tmux on a login node. It sleeps
# between polls and shells out to mmlsquota once per cycle -- far below the login-node
# 2-CPU / 4-GB cap, and lighter than a compile.
#
# SETUP: copy this file into your OWN directory, then pass YOUR address as the first
# argument -- a bare uniqname gets @umich.edu added. There is no default recipient: the
# script refuses to start without one, so nobody silently mails the previous owner.
# (The MAILTO env var works too, but an argument wins over it.)
#
# USAGE (login node):
#   ./watch_scratch_quota.sh uniqname --test   # one test email + exit; check inbox AND spam
#   tmux new -s diskwatch                      # --test exits non-zero if mail failed
#   ./watch_scratch_quota.sh uniqname
#   # detach: Ctrl-b then d      reattach: tmux attach -t diskwatch
#
#   ./watch_scratch_quota.sh --once            # print one reading + exit (no mail, no address)
#   ./watch_scratch_quota.sh --help
#
# NOTE: tmux is tied to the login node you started it on (gl-login1/2/3) -- reattach from
# that same host. Nothing is shared between users; each runs their own copy and gets their
# own mail. The thresholds below describe the SHARED fileset, so the defaults suit everyone.
# Only one instance per user may run at a time (lockfile); a second one exits immediately.
#
# Knobs are env vars, e.g.:  WARN_GIB=800 CHECK_EVERY=300 ./watch_scratch_quota.sh uniqname
# All of them must be plain integers -- anything else is rejected at startup rather than
# silently breaking the watcher hours later. RECOVER_FACTOR is a PERCENT (115, not 1.15).
#
# EMAIL BEHAVIOUR
#   - one STARTED mail when you launch it (proves delivery works right away).
#   - after that: silence, except threshold CROSSINGS and one daily heartbeat.
#   - the heartbeat is the point: if it stops arriving, this watcher died (login node
#     rebooted, ARC killed it, tmux gone) and you are no longer covered. Treat missing
#     heartbeats as the alarm.

set -uo pipefail

FILESET="${FILESET:-jjparkcv_root}"
DEVICE="${DEVICE:-gl-scratch}"
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

die() { echo "ERROR: $*" >&2; exit 1; }

MODE=watch
recipient=""
for arg in "$@"; do
    case "$arg" in
        --once) MODE=once ;;
        --test) MODE=test ;;
        -h|--help) sed -n '2,/^$/p' "$SELF" | sed 's/^# \?//'; exit 0 ;;
        -*) die "unknown option '$arg' (try --help)" ;;
        # A dropped dash would otherwise silently become a recipient like test@umich.edu.
        once|test|help) die "did you mean --$arg ?" ;;
        "") die "empty recipient argument (would have mailed '@umich.edu')" ;;
        *@*) recipient="$arg" ;;                 # full address
        *)   recipient="${arg}@umich.edu" ;;     # bare uniqname
    esac
    [[ -n "$recipient" && -n "${MAILTO_ARG:-}" && "$recipient" != "$MAILTO_ARG" ]] \
        && die "more than one recipient given ('$MAILTO_ARG' and '$recipient'); pass only one"
    [[ -n "$recipient" ]] && MAILTO_ARG="$recipient"
done
[[ -n "$recipient" ]] && MAILTO="$recipient"

# --once only prints a reading, so it needs no recipient. Everything else mails, and we
# refuse rather than default to some previous owner's address.
if [[ "$MODE" != "once" ]]; then
    [[ -n "${MAILTO:-}" ]] || die "no recipient -- this watcher would have nowhere to send alerts.
  $SELF uniqname@umich.edu     (or just: $SELF uniqname)
  or: export MAILTO=uniqname@umich.edu"
    [[ "$MAILTO" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]] \
        || die "'$MAILTO' does not look like an email address"
fi

WARN_GIB=${WARN_GIB:-500}        # headroom on the SHARED fileset, not your personal usage
CRIT_GIB=${CRIT_GIB:-200}
CHECK_EVERY=${CHECK_EVERY:-600}  # 10 min
RENAG_SECS=${RENAG_SECS:-21600}  # re-nag at most every 6h while still bad
HEARTBEAT_HOUR=${HEARTBEAT_HOUR:-8}
RECOVER_FACTOR=${RECOVER_FACTOR:-115}  # PERCENT: need free > WARN*1.15 to clear (hysteresis)

# Validate every knob up front. Without this, a bareword (CRIT_GIB=abc) kills the loop via
# set -u seconds after the "you are covered" mail, a float (RECOVER_FACTOR=1.15) wedges the
# state machine so it never recovers, and a zero-padded hour (HEARTBEAT_HOUR=08) silently
# disables the heartbeat forever. All three are invisible in a detached tmux pane.
for knob in WARN_GIB CRIT_GIB CHECK_EVERY RENAG_SECS HEARTBEAT_HOUR RECOVER_FACTOR; do
    [[ "${!knob}" =~ ^[0-9]+$ ]] || die "$knob must be a whole number, got '${!knob}'$(
        [[ "$knob" == RECOVER_FACTOR ]] && echo " (it is a PERCENT: use 115, not 1.15)")"
done
HEARTBEAT_HOUR=$((10#$HEARTBEAT_HOUR))   # so 08 is 8, not an octal error
(( HEARTBEAT_HOUR <= 23 ))    || die "HEARTBEAT_HOUR must be 0-23, got $HEARTBEAT_HOUR"
(( CHECK_EVERY >= 1 ))        || die "CHECK_EVERY must be >= 1"
(( CRIT_GIB < WARN_GIB ))     || die "CRIT_GIB ($CRIT_GIB) must be below WARN_GIB ($WARN_GIB)"
(( RECOVER_FACTOR >= 100 ))   || die "RECOVER_FACTOR is a percent >= 100, got $RECOVER_FACTOR"

LOG="${LOG:-$HOME/.diskwatch.log}"
mkdir -p "$(dirname "$LOG")" || die "cannot create log directory $(dirname "$LOG")"
touch "$LOG" 2>/dev/null     || die "cannot write log file $LOG"

log() { echo "$(date '+%F %T') $*" >> "$LOG"; }

free_gib() {
    # -> "used limit free"; empty on failure so the loop can survive a transient blip.
    # timeout: GPFS can block indefinitely during an outage, which is exactly when we care.
    timeout 60 mmlsquota -j "$FILESET" --block-size 1G "$DEVICE" 2>/dev/null \
        | awk -v OFS=' ' '/^'"$DEVICE"'/{print $3, $5, $5-$3-$6; found=1; exit} END{exit !found}' \
        | grep -E '^-?[0-9]+ -?[0-9]+ -?[0-9]+$'   # reject partial/garbled lines
}

send() {  # send <subject> <body>  -> exit status reflects whether mail accepted it
    if ! command -v mail >/dev/null 2>&1; then
        log "MAIL-UNAVAILABLE: $1"; return 1
    fi
    if printf '%s\n' "$2" | mail -s "$1" "$MAILTO"; then
        log "MAIL-QUEUED: $1"; return 0    # queued != delivered; bounces go to local mail
    else
        log "MAIL-FAILED: $1"; return 1
    fi
}

fmt() {
    [[ "$1" =~ ^-?[0-9]+$ ]] || { printf '%s' "$1"; return; }
    awk -v g="$1" 'BEGIN{ if (g>=1024) printf "%.2f TiB", g/1024; else printf "%d GiB", g }'
}

read -r _u _l cur < <(free_gib) \
    || die "mmlsquota failed or gave unparsable output for fileset '$FILESET' on '$DEVICE'.
  Check the fileset name, that you are in its group, and that mmlsquota is on PATH:
    mmlsquota -j $FILESET --block-size 1G $DEVICE"
(( _l > 0 )) || die "fileset '$FILESET' reports no block limit (limit=0); free space is meaningless here"

if [[ "$MODE" == "once" ]]; then
    echo "used=$_u limit=$_l free=$(fmt "$cur")  [WARN<$WARN_GIB CRIT<$CRIT_GIB]"; exit 0
fi
if [[ "$MODE" == "test" ]]; then
    if send "[diskwatch] TEST - $DEVICE free $(fmt "$cur")" \
"This is a one-off test from $SELF on $(hostname).

If you are reading this, mail delivery works and the watcher can reach you.

  fileset : $FILESET on $DEVICE
  used    : $(fmt "$_u") / $(fmt "$_l")
  free    : $(fmt "$cur")
  WARN    : < $(fmt "$WARN_GIB")
  CRITICAL: < $(fmt "$CRIT_GIB")"; then
        echo "test mail queued -> $MAILTO (see $LOG); check your inbox AND spam"; exit 0
    else
        die "could not send test mail to $MAILTO -- see $LOG. Do NOT rely on this watcher until this works."
    fi
fi

# One watcher per user. Two instances (easy to do: tmux is per-login-node) would double
# every alert and, worse, make a dead one undetectable because the other keeps heartbeating.
LOCK="${LOCK:-$HOME/.diskwatch.lock}"
exec 9>"$LOCK" || die "cannot open lockfile $LOCK"
flock -n 9 || die "another watcher is already running (lock: $LOCK). Find it with: pgrep -u $USER -f $(basename "$SELF")"

state=OK; last_alert=0; day_min=$cur; hb_day=""; fails=0; broken_notified=0
log "START pid=$$ host=$(hostname) free=$(fmt "$cur") warn=$WARN_GIB crit=$CRIT_GIB every=${CHECK_EVERY}s"
echo "watching $DEVICE:$FILESET every ${CHECK_EVERY}s - free now $(fmt "$cur"); log: $LOG"
echo "detach with Ctrl-b d ; alerts + daily ${HEARTBEAT_HOUR}:00 heartbeat -> $MAILTO"

# Startup mail: proves the watcher is live and that mail still reaches you, without
# waiting until the next heartbeat hour. Also records the baseline you started from.
send "[diskwatch] STARTED - $DEVICE free $(fmt "$cur")" \
"Watcher started on $(hostname), pid $$, at $(date '+%F %T').

  free now : $(fmt "$cur")
  used     : $(fmt "$_u") / $(fmt "$_l")
  poll     : every ${CHECK_EVERY}s
  WARN     : < $(fmt "$WARN_GIB")
  CRITICAL : < $(fmt "$CRIT_GIB")
  heartbeat: daily at ${HEARTBEAT_HOUR}:00
  jobs     : $(squeue -u "$USER" -h -t R -o '%j' 2>/dev/null | tr '\n' ' ')

From here on this mailbox stays quiet unless a threshold is crossed. You will get
one heartbeat a day at ${HEARTBEAT_HOUR}:00 - if that ever stops arriving, the watcher
died and you are no longer covered.

  reattach: tmux attach -t diskwatch      (must be on $(hostname))
  log     : $LOG"

# Today's heartbeat slot has already passed (or is now), and the startup mail just covered
# it -- don't fire a near-duplicate on the very next poll.
(( $(date +%-H) >= HEARTBEAT_HOUR )) && hb_day=$(date +%F)

while true; do
    if read -r used limit cur < <(free_gib); then
        ok=1; fails=0; broken_notified=0
    else
        ok=0; fails=$((fails + 1))
        log "QUOTA-QUERY-FAILED ($fails in a row)"
    fi
    now=$(date +%s); hour=$(date +%-H); today=$(date +%F)

    if (( ok )); then
        (( cur < day_min )) && day_min=$cur

        # Hysteresis on BOTH edges: without it, free space oscillating around a threshold
        # mails on every single poll (measured: 7 mails in 8 polls around CRIT).
        if   (( cur < CRIT_GIB )); then new=CRITICAL
        elif (( cur < WARN_GIB )); then
            if [[ "$state" == CRITICAL ]] && (( cur <= CRIT_GIB * RECOVER_FACTOR / 100 ))
            then new=CRITICAL          # still inside the CRIT recovery band -> hold
            else new=WARN; fi
        elif (( cur > WARN_GIB * RECOVER_FACTOR / 100 )); then new=OK
        else new=$state; fi   # inside the WARN hysteresis band -> hold

        log "free=${cur}GiB used=${used} state=$new"

        if [[ "$new" != "$state" ]]; then
            case "$new" in
              CRITICAL|WARN)
                # Direction matters: CRITICAL->WARN is an improvement, and mailing it with
                # the same "crossed the threshold" wording reads like a fresh problem.
                if [[ "$state" == CRITICAL ]]; then dir="easing: was CRITICAL, now $new"
                else                                dir="worsening: crossed the $new threshold"; fi
                send "[$DEVICE $new] free $(fmt "$cur") ($dir)" \
"$DEVICE free space -- $dir.

  free now : $(fmt "$cur")
  used     : $(fmt "$used") / $(fmt "$limit")
  min since last heartbeat : $(fmt "$day_min")
  host     : $(hostname)   time: $(date '+%F %T')

This fileset is shared across the lab, so the space may not be yours -- but a full
fileset will kill anyone's job mid-checkpoint. Remember each in-flight save needs
room for a transient .tmp beside the checkpoint, on top of the checkpoint itself.

Check:  squeue -u \$USER
        mmlsquota -j $FILESET --block-size auto $DEVICE
        du -sh /scratch/$FILESET/*/\$USER/* 2>/dev/null | sort -h | tail"
                last_alert=$now ;;
              OK)
                send "[$DEVICE RECOVERED] free $(fmt "$cur")" \
"$DEVICE recovered: free is back to $(fmt "$cur") (was $state; min since last heartbeat $(fmt "$day_min"))." ;;
            esac
            state=$new
        elif [[ "$state" != "OK" ]] && (( now - last_alert > RENAG_SECS )); then
            send "[$DEVICE still $state] free $(fmt "$cur")" \
"Still $state. free=$(fmt "$cur")  used=$(fmt "$used")/$(fmt "$limit")  min since last heartbeat=$(fmt "$day_min")"
            last_alert=$now
        fi
    elif (( fails >= 3 && broken_notified == 0 )); then
        # A permanent failure (fileset renamed, group revoked, mmlsquota gone) would
        # otherwise retry forever in silence while the process still looks healthy.
        send "[diskwatch] QUOTA QUERY BROKEN on $(hostname)" \
"Cannot read the quota for $FILESET on $DEVICE -- $fails consecutive failures.

The watcher is still running but is NOT able to warn you about disk space.

  host : $(hostname)   pid: $$   time: $(date '+%F %T')
  try  : mmlsquota -j $FILESET --block-size 1G $DEVICE
  log  : $LOG"
        broken_notified=1
    fi

    # Fires on the first poll at/after the heartbeat hour, so a coarse CHECK_EVERY or a
    # skipped poll delays it rather than losing the day entirely. Runs on the failure
    # path too -- a broken watcher must still prove it is alive.
    if (( hour >= HEARTBEAT_HOUR )) && [[ "$today" != "$hb_day" ]]; then
        if (( ok )); then status_line="  free now : $(fmt "$cur")
  used     : $(fmt "$used") / $(fmt "$limit")
  min since last heartbeat : $(fmt "$day_min")
  state    : $state   (WARN<$(fmt "$WARN_GIB")  CRIT<$(fmt "$CRIT_GIB"))"
        else status_line="  QUOTA QUERY IS FAILING ($fails in a row) - not currently able to warn you."; fi
        send "[diskwatch] daily - $DEVICE $( (( ok )) && echo "free $(fmt "$cur") ($state)" || echo "QUERY FAILING")" \
"Watcher alive on $(hostname), pid $$.

$status_line
  jobs     : $(squeue -u "$USER" -h -t R -o '%j' 2>/dev/null | tr '\n' ' ')

If this daily mail ever STOPS, the watcher died (login-node reboot, ARC killed
it, or tmux is gone) and you are no longer being warned. Restart with:
  tmux new -s diskwatch ; $SELF $MAILTO"
        hb_day=$today; (( ok )) && day_min=$cur   # reset the window
    fi

    sleep "$CHECK_EVERY"
done
