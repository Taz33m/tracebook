"""Reserve and atomically publish new L3 report and trace files."""

from __future__ import annotations

import os
import uuid
from contextlib import ExitStack, contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Optional

from .model import BookReplayError


def _identity(stat: os.stat_result) -> tuple[int, int]:
    return stat.st_dev, stat.st_ino


def _write_payload(descriptor: int, payload: bytes) -> None:
    with os.fdopen(os.dup(descriptor), "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


class OutputReservation:
    """Own a sidecar reservation while the destination itself stays absent."""

    def __init__(self, destination: str | Path) -> None:
        path = Path(destination).expanduser()
        self._requested_target = path.absolute()
        try:
            self.target = path.parent.resolve() / path.name
        except RuntimeError as exc:
            raise BookReplayError(f"cannot resolve book-replay path: {exc}") from exc
        self._lock_name = f".{self.target.name}.tracebook-in-progress"
        self._stage_directory_name = f".tracebook-stage-{uuid.uuid4().hex}"
        self._stage_name = "payload"
        self._parent_fd: Optional[int] = None
        self._stage_directory_fd: Optional[int] = None
        self._stage_directory_identity: Optional[tuple[int, int]] = None
        self._lock_fd: Optional[int] = None
        self._stage_fd: Optional[int] = None
        self._published = False

    def __enter__(self) -> "OutputReservation":
        if not all(operation in os.supports_dir_fd for operation in (os.open, os.stat, os.link)):
            raise BookReplayError(
                "L3 artifact publication requires descriptor-relative file operations"
            )
        try:
            self.target.parent.mkdir(parents=True, exist_ok=True)
            self._parent_fd = os.open(
                self.target.parent,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
            )
            self._require_absent()
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            self._lock_fd = os.open(self._lock_name, flags, 0o600, dir_fd=self._parent_fd)
            os.mkdir(self._stage_directory_name, mode=0o700, dir_fd=self._parent_fd)
            # Retain the identity even if opening the newly created directory
            # fails. Cleanup must not remove a replacement at the same name.
            self._stage_directory_identity = _identity(
                os.stat(self._stage_directory_name, dir_fd=self._parent_fd, follow_symlinks=False)
            )
            stage_directory_fd = os.open(
                self._stage_directory_name,
                os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=self._parent_fd,
            )
            try:
                if _identity(os.fstat(stage_directory_fd)) != self._stage_directory_identity:
                    raise BookReplayError(f"book-replay output reservation changed: {self.target}")
            except BaseException:
                os.close(stage_directory_fd)
                raise
            self._stage_directory_fd = stage_directory_fd
            self._stage_fd = os.open(
                self._stage_name, flags, 0o400, dir_fd=self._stage_directory_fd
            )
            # Keep the source entry in a private, non-writable directory. The
            # write descriptor remains usable, but another normal file writer
            # cannot replace the source entry in the validate/link window.
            # Linking relative to this directory's descriptor also survives a
            # competing rename/replacement of its entry in the output parent.
            os.fchmod(self._stage_directory_fd, 0o500)
            self.validate()
        except BaseException:
            try:
                self.close()
            except OSError:
                pass
            raise
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            self.close()
        except OSError:
            if exc_type is None:
                raise

    def _require_absent(self) -> None:
        try:
            os.stat(self.target.name, dir_fd=self._parent_fd, follow_symlinks=False)
        except FileNotFoundError:
            return
        raise BookReplayError(f"book-replay output already exists: {self.target}")

    def _owns_entry(
        self, name: str, descriptor: Optional[int], *, directory: Optional[int] = None
    ) -> bool:
        directory = self._parent_fd if directory is None else directory
        if directory is None or descriptor is None:
            return False
        try:
            return _identity(os.stat(name, dir_fd=directory, follow_symlinks=False)) == _identity(
                os.fstat(descriptor)
            )
        except OSError:
            return False

    def validate(self) -> None:
        if self._parent_fd is None:
            raise BookReplayError("book-replay output reservation is closed")
        if (
            not self._same_parent()
            or not self._owns_entry(self._lock_name, self._lock_fd)
            or not self._owns_entry(self._stage_directory_name, self._stage_directory_fd)
            or not self._owns_entry(
                self._stage_name, self._stage_fd, directory=self._stage_directory_fd
            )
        ):
            raise BookReplayError(f"book-replay output reservation changed: {self.target}")
        self._require_absent()

    def _owns_stage_directory(self) -> bool:
        if self._parent_fd is None or self._stage_directory_identity is None:
            return False
        try:
            return (
                _identity(
                    os.stat(
                        self._stage_directory_name, dir_fd=self._parent_fd, follow_symlinks=False
                    )
                )
                == self._stage_directory_identity
            )
        except OSError:
            return False

    def _same_parent(self) -> bool:
        if self._parent_fd is None:
            return False
        try:
            expected = _identity(os.fstat(self._parent_fd))
            return (
                _identity(self.target.parent.stat(follow_symlinks=False)) == expected
                and _identity(self._requested_target.parent.stat()) == expected
            )
        except OSError:
            return False

    def stage(self, payload: bytes) -> None:
        self.validate()
        if self._stage_fd is None:
            raise BookReplayError("book-replay output reservation is closed")
        _write_payload(self._stage_fd, payload)

    def publish(self) -> None:
        self.validate()
        # A hard link publishes the complete staged inode atomically and fails
        # if anything appeared at the destination after validation.
        # Mark the attempt first: interruption immediately after a successful
        # link still needs rollback, which checks the destination's inode.
        self._published = True
        os.link(
            self._stage_name,
            self.target.name,
            src_dir_fd=self._stage_directory_fd,
            dst_dir_fd=self._parent_fd,
            follow_symlinks=False,
        )
        self.validate_published()

    def validate_published(self) -> None:
        if (
            not self._published
            or not self._same_parent()
            or not self._owns_entry(self.target.name, self._stage_fd)
        ):
            raise BookReplayError(f"book-replay output changed during publication: {self.target}")

    def rollback(self) -> None:
        if self._published and self._owns_entry(self.target.name, self._stage_fd):
            os.unlink(self.target.name, dir_fd=self._parent_fd)
        self._published = False

    def close(self) -> None:
        error: Optional[OSError] = None
        if self._stage_directory_fd is not None:
            try:
                os.fchmod(self._stage_directory_fd, 0o700)
            except OSError as exc:
                error = exc
        for name, descriptor, directory in (
            (self._stage_name, self._stage_fd, self._stage_directory_fd),
            (self._lock_name, self._lock_fd, self._parent_fd),
        ):
            if descriptor is not None:
                try:
                    if self._owns_entry(name, descriptor, directory=directory):
                        os.unlink(name, dir_fd=directory)
                except OSError as exc:
                    error = error or exc
                finally:
                    try:
                        os.close(descriptor)
                    except OSError as exc:
                        error = error or exc
        self._stage_fd = self._lock_fd = None
        try:
            if self._owns_stage_directory():
                os.rmdir(self._stage_directory_name, dir_fd=self._parent_fd)
        except OSError as exc:
            error = error or exc
        self._stage_directory_identity = None
        if self._stage_directory_fd is not None:
            try:
                os.close(self._stage_directory_fd)
            except OSError as exc:
                error = error or exc
            self._stage_directory_fd = None
        if self._parent_fd is not None:
            try:
                os.close(self._parent_fd)
            except OSError as exc:
                error = error or exc
            self._parent_fd = None
        if error is not None:
            raise error


@contextmanager
def reserve_outputs(*destinations: Optional[str]) -> Iterator[dict[str, OutputReservation]]:
    """Reserve every output before candidate work, including conditional traces."""
    with ExitStack() as stack:
        outputs = {
            destination: stack.enter_context(OutputReservation(destination))
            for destination in destinations
            if destination is not None
        }
        # Also catch a destination colliding with another output's sidecar.
        for output in outputs.values():
            output.validate()
        yield outputs


def publish_outputs(payloads: Iterable[tuple[OutputReservation, bytes]]) -> None:
    """Stage all bytes, then publish in order; roll back this attempt on failure."""
    prepared = list(payloads)
    try:
        for output, payload in prepared:
            output.stage(payload)
        for output, _payload in prepared:
            output.publish()
        for output, _payload in prepared:
            output.validate_published()
    except BaseException:
        for output, _payload in reversed(prepared):
            try:
                output.rollback()
            except OSError:
                pass
        raise
