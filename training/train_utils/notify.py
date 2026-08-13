"""Email notifications for long-running jobs.

Great Lakes compute nodes run a local postfix listener on 127.0.0.1:25, so a job can hand
a message to it with no credentials and no outbound network config. Verified working from
`gl1723` on 2026-08-04.

    from train_utils.notify import send_email
    send_email("training collapsed", "grad_aggregator < 1e-6 for 500 steps")

Design rules, all load-bearing:
  * **Never raises.** A notification failure must not kill a 14-day training run. Every
    call is wrapped and returns a bool instead.
  * **Never blocks for long.** Hard socket timeout; a hung MTA costs seconds, not a job.
  * **Deduplicated by key.** `once_key` makes a given alarm fire exactly once per process,
    so a per-step tripwire cannot emit 100k emails.
"""

import logging
import os
import socket
import smtplib
import time
from email.message import EmailMessage

DEFAULT_TO = os.environ.get("VGGT_NOTIFY_EMAIL", "minsukc@umich.edu")
SMTP_HOST = os.environ.get("VGGT_SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("VGGT_SMTP_PORT", "25"))
TIMEOUT_S = 20

_SENT_KEYS = set()


def _context():
    """Job identity, so an alarm in the inbox is actionable without digging."""
    bits = [f"host={socket.getfqdn()}"]
    for var in ("SLURM_JOB_ID", "SLURM_ARRAY_TASK_ID", "SLURM_JOB_NAME"):
        if os.environ.get(var):
            bits.append(f"{var.lower()}={os.environ[var]}")
    bits.append(f"cwd={os.getcwd()}")
    bits.append(f"time={time.strftime('%Y-%m-%d %H:%M:%S')}")
    return "\n".join(bits)


def send_email(subject, body, to=None, once_key=None, prefix="[vggt]"):
    """Send a notification email. Returns True if the MTA accepted it.

    Args:
        subject:  short subject line (``prefix`` is prepended).
        body:     message body; job context (host, SLURM id, cwd, time) is appended.
        to:       recipient; defaults to $VGGT_NOTIFY_EMAIL or minsukc@umich.edu.
        once_key: if given, only the FIRST call with this key in this process sends.
                  Use for per-step tripwires.
        prefix:   subject prefix.
    """
    if once_key is not None:
        if once_key in _SENT_KEYS:
            return False
        _SENT_KEYS.add(once_key)

    to = to or DEFAULT_TO
    try:
        msg = EmailMessage()
        msg["From"] = f"vggt-train@{socket.getfqdn()}"
        msg["To"] = to
        msg["Subject"] = f"{prefix} {subject}"
        msg.set_content(f"{body}\n\n--\n{_context()}\n")
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=TIMEOUT_S) as s:
            refused = s.send_message(msg)
        if refused:
            logging.warning(f"[notify] some recipients refused: {refused}")
            return False
        logging.info(f"[notify] emailed {to}: {subject}")
        return True
    except Exception as e:
        # Deliberately broad: notification is best-effort, training is not.
        logging.warning(f"[notify] email failed (ignored): {type(e).__name__}: {e}")
        return False


class GradientCollapseAlarm:
    """Fires once when a gradient norm stays at/below `threshold` for `patience` steps.

    Motivated by docs/64: both pooled1337 runs had `grad_aggregator` sitting below 1e-6
    for ~70 epochs (a dead ReLU in the DPT head severed the gradient path) while
    `grad_point` looked healthy, so nothing in the standard logging flagged it and two
    GPUs burned two days producing a frozen model.

    Threshold calibration (measured, docs/64): a HEALTHY run's `grad_aggregator` median is
    1e-2..8e-2; even a badly degraded 3e-4 arm still reads ~6e-5. Sitting under 1e-6 for
    hundreds of consecutive steps does not happen in a live run, so false positives are
    not a practical concern.
    """

    def __init__(self, threshold=1e-6, patience=200, name="aggregator", enabled=True):
        self.threshold = float(threshold)
        self.patience = int(patience)
        self.name = name
        self.enabled = bool(enabled)
        self.run = 0
        self.fired = False

    def update(self, value, step, epoch=None, extra=""):
        """Feed one observation. Returns True on the step the alarm fires."""
        if not self.enabled or self.fired:
            return False
        if value is None or value > self.threshold:
            self.run = 0
            return False
        self.run += 1
        if self.run < self.patience:
            return False
        self.fired = True
        msg = (f"grad_{self.name} has been <= {self.threshold:g} for {self.run} consecutive "
               f"steps (step {step}, epoch {epoch}).\n\n"
               f"This is the docs/64 failure signature: a dead ReLU in the DPT head "
               f"(point_head.scratch.output_conv2[1]) makes the head emit only its bias, so "
               f"the predicted DVF is constant and NO gradient reaches the aggregator. It is "
               f"NOT recoverable -- the run is wasting GPU time from here on.\n\n"
               f"Check: PYTHONPATH=training:. python tools/probe_aggft_collapse.py "
               f"--ckpt <log_dir>/ckpts/checkpoint_last.pt\n{extra}")
        logging.error(f"[ALARM] {msg}")
        send_email(f"GRADIENT COLLAPSE: grad_{self.name} dead", msg,
                   once_key=f"gradcollapse:{self.name}")
        return True
