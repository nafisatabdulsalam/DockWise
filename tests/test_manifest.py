from __future__ import annotations

from dockwise.runner import CompletionManifest


def test_completion_manifest_resumes_without_repeating(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    first = CompletionManifest(path, {"protocol": "test"})
    first.complete("run-a")
    second = CompletionManifest(path, {"protocol": "ignored-on-resume"})
    assert second.contains("run-a")
    second.error("run-b", "deliberate")
    third = CompletionManifest(path, {})
    assert third.data["errors"]["run-b"] == "deliberate"

