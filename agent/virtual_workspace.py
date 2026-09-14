"""In-memory workspace for dataset generation.

Generation composes ~3,000 traces a second, each with its own files. Doing
that on disk is neither fast nor safe. This backend keeps the files in a dict
and inherits every tool behaviour and observation string from
``WorkspaceBase``, so a trace generated here contains exactly what the real
tool would have returned - the observation is produced by the same code, not
imitated.
"""

from agent.workspace_tools import WorkspaceBase


class VirtualWorkspace(WorkspaceBase):
    def __init__(self, files: dict = None):
        self.files = dict(files or {})

    def _get(self, rel):
        return self.files.get(rel)

    def _put(self, rel, text):
        self.files[rel] = text

    def _names(self):
        return sorted(self.files)

    def snapshot(self) -> dict:
        return dict(self.files)
