import subprocess
from datetime import datetime, timezone
from threading import Lock


class MediaCommandError(RuntimeError):
    def __init__(
        self,
        message,
        command_label=None,
        returncode=None,
        stdout=None,
        stderr=None,
        timed_out=False,
    ):
        super().__init__(message)
        self.command_label = command_label
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.timed_out = timed_out

    def as_dict(self):
        return {
            "message": str(self),
            "command_label": self.command_label,
            "returncode": self.returncode,
            "timed_out": self.timed_out,
            "stdout_tail": MediaCommandRunner.compact_output(self.stdout),
            "stderr_tail": MediaCommandRunner.compact_output(self.stderr),
        }


class MediaCommandRunner:
    def __init__(self, timeout_seconds, media_commands=None, proot_distro=None):
        self.timeout_seconds = max(1, int(timeout_seconds or 1))
        self.media_commands = set(media_commands or ())
        self.proot_distro = proot_distro
        self._lock = Lock()
        self._started = 0
        self._succeeded = 0
        self._failed = 0
        self._timed_out = 0
        self._last_error = None

    @staticmethod
    def compact_output(text, limit=800):
        output = str(text or "").strip()
        if len(output) <= limit:
            return output
        return output[-limit:]

    def prepare(self, args):
        if args and args[0] in self.media_commands and self.proot_distro:
            return [self.proot_distro, "login", "ubuntu", "--", *args]
        return args

    def label(self, args):
        if not args:
            return "command"
        if args[0] == "proot-distro" and "--" in args:
            passthrough = args[args.index("--") + 1:]
            if passthrough:
                return " ".join(str(part) for part in passthrough[:3])
        return " ".join(str(part) for part in args[:3])

    def _record_start(self):
        with self._lock:
            self._started += 1

    def _record_success(self):
        with self._lock:
            self._succeeded += 1

    def _record_error(self, error):
        with self._lock:
            self._failed += 1
            if getattr(error, "timed_out", False):
                self._timed_out += 1
            self._last_error = {
                **error.as_dict(),
                "at": datetime.now(timezone.utc).isoformat(),
            }

    def run(self, args, timeout=None, check=True):
        prepared = self.prepare(args)
        limit = self.timeout_seconds if timeout is None else timeout
        label = self.label(prepared)
        self._record_start()
        try:
            result = subprocess.run(
                prepared,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=limit,
            )
        except subprocess.TimeoutExpired as exc:
            output = self.compact_output((exc.stderr or "") or (exc.stdout or ""))
            suffix = f": {output}" if output else ""
            error = MediaCommandError(
                f"{label} timed out after {limit}s{suffix}",
                command_label=label,
                stdout=exc.stdout,
                stderr=exc.stderr,
                timed_out=True,
            )
            self._record_error(error)
            raise error from exc

        if check and result.returncode != 0:
            output = self.compact_output(result.stderr or result.stdout)
            error = MediaCommandError(
                f"{label} failed: {output or 'command failed'}",
                command_label=label,
                returncode=result.returncode,
                stdout=result.stdout,
                stderr=result.stderr,
            )
            self._record_error(error)
            raise error

        self._record_success()
        return result

    def stats(self):
        with self._lock:
            return {
                "timeout_seconds": self.timeout_seconds,
                "media_commands": sorted(self.media_commands),
                "proot_enabled": bool(self.proot_distro),
                "started": self._started,
                "succeeded": self._succeeded,
                "failed": self._failed,
                "timed_out": self._timed_out,
                "last_error": self._last_error,
            }
