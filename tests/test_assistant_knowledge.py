"""The library: authorized folders, incremental, cited, and never a way out."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from isharded.assistant import knowledge
from isharded.assistant.app import Assistant
from isharded.assistant.cli import add_arguments, run
from isharded.assistant.contracts import Reach, Trust, is_tainted, lowest_reach
from isharded.assistant.knowledge import LibraryRefused, chunk_blocks, chunk_markdown, fuse, is_secret


@pytest.fixture
def assistant(tmp_path):
    with Assistant(tmp_path / "home") as instance:
        yield instance


@pytest.fixture
def folder(tmp_path):
    root = tmp_path / "docs"
    root.mkdir()
    return root


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def found(items) -> set[str]:
    return {Path(item.provenance.source_id.rsplit(":", 1)[0]).name for item in items}


def run_cli(home: Path, *arguments: str) -> int:
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    return run(parser.parse_args(["--home", str(home), *arguments]))


def tiny_pdf(pages: list[str]) -> bytes:
    """A real PDF with text on each page, built by hand so no writer is needed."""
    objects = ["<< /Type /Catalog /Pages 2 0 R >>", ""]
    kids = []
    for index, text in enumerate(pages):
        page, content = 4 + index * 2, 5 + index * 2
        kids.append(f"{page} 0 R")
        stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET"
        objects += [
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents {content} 0 R /Resources << /Font << /F1 3 0 R >> >> >>",
            f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
        ]
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>"
    objects.insert(2, "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    body, offsets = "%PDF-1.4\n", []
    for number, item in enumerate(objects, 1):
        offsets.append(len(body))
        body += f"{number} 0 obj\n{item}\nendobj\n"
    table = len(body)
    body += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n"
    body += "".join(f"{offset:010d} 00000 n \n" for offset in offsets)
    body += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{table}\n%%EOF\n"
    return body.encode("latin-1")


# Chunking


def test_markdown_is_chunked_by_heading_with_line_numbers():
    text = "intro line\n\n# First\nalpha\n\n## Second\nbeta\n\n```\n# not a heading\n```\n\n# Third\ngamma\n"
    chunks = chunk_markdown(text)
    assert [chunk.heading for chunk in chunks] == ["", "First", "Second", "Third"]
    assert [chunk.line for chunk in chunks] == [1, 3, 6, 13]
    assert "# not a heading" in chunks[2].text


def test_code_is_chunked_by_blank_line_blocks_with_line_numbers():
    blocks = [f"def function_{number}():\n    return {number}" for number in range(6)]
    chunks = chunk_blocks("\n\n".join(blocks) + "\n", limit=70)
    assert len(chunks) == 3
    lines = ("\n\n".join(blocks)).split("\n")
    for chunk in chunks:
        assert lines[chunk.line - 1] == chunk.text.split("\n")[0]
        assert lines[chunk.end_line - 1] == chunk.text.split("\n")[-1]


def test_a_block_longer_than_the_limit_is_cut_and_nothing_is_lost():
    text = "\n".join(f"line {number}" for number in range(400))
    chunks = chunk_blocks(text, limit=200)
    assert len(chunks) > 5 and all(len(chunk.text) <= 200 for chunk in chunks)
    assert "\n".join(chunk.text for chunk in chunks) == text


def test_one_enormous_line_is_cut_too():
    text = "x" * 5000
    chunks = chunk_blocks(text, limit=1000)
    assert "".join(chunk.text for chunk in chunks) == text and all(chunk.line == 1 for chunk in chunks)


def test_an_empty_file_has_no_chunks():
    assert chunk_blocks("") == [] and chunk_blocks("\n\n \n") == [] and chunk_markdown("") == []


def test_windows_line_endings_keep_line_numbers_right():
    chunks = chunk_blocks("one\r\n\r\ntwo\r\n", limit=5)
    assert [chunk.line for chunk in chunks] == [1, 3]


def test_a_citation_skips_blank_lines_at_the_start_of_a_cut_section():
    lines = ["# Title", "a" * 30, "", "", "b" * 30]
    chunks = chunk_markdown("\n".join(lines), limit=40)
    for chunk in chunks:
        assert lines[chunk.line - 1] == chunk.text.split("\n")[0]


# Searching and citations


def test_search_returns_external_items_with_a_citation_that_checks_out(assistant, folder):
    source = write(folder / "notes.md", "# Trip\n\nBooked the ferry.\n\n# Budget\n\nSpent 40 on a lighthouse tour.\n")
    write(folder / "code.py", "import os\n\n\ndef lighthouse_beam():\n    return 1\n")
    assistant.library.add("alice", folder, Reach.MACHINE)
    assistant.library.index("alice")

    items = assistant.library.search("alice", "lighthouse")
    assert found(items) == {"notes.md", "code.py"}
    assert is_tainted(items) and lowest_reach(items) is Reach.MACHINE
    for item in items:
        assert item.provenance.trust is Trust.EXTERNAL and item.provenance.source == "library"
        path, line = item.provenance.source_id.rsplit(":", 1)
        checked = Path(path).read_text(encoding="utf-8").split("\n")[int(line) - 1]
        assert checked == item.text.split("\n")[0]
    assert str(source) in {item.provenance.source_id.rsplit(":", 1)[0] for item in items}


def test_a_pdf_is_chunked_by_page_and_cited_by_page(assistant, folder):
    (folder / "manual.pdf").write_bytes(tiny_pdf(["The first page is about turbines", "The second page is about gearboxes"]))
    assistant.library.add("alice", folder)
    report = assistant.library.index("alice")
    assert report.added == 1 and report.problems == []
    item = assistant.library.search("alice", "gearboxes")[0]
    assert item.provenance.source_id == f"{folder / 'manual.pdf'}#page=2"
    assert "second page" in item.text


def test_without_pypdf_a_pdf_is_skipped_with_a_plain_message(assistant, folder, monkeypatch):
    (folder / "manual.pdf").write_bytes(tiny_pdf(["About turbines"]))
    write(folder / "a.txt", "plain text survives")
    monkeypatch.setitem(sys.modules, "pypdf", None)
    assistant.library.add("alice", folder)
    report = assistant.library.index("alice")
    assert report.added == 1 and report.skipped == 1
    assert report.problems == ["PDF files were skipped. To index them: pip install pypdf"]


def test_a_damaged_pdf_does_not_stop_the_folder(assistant, folder):
    (folder / "broken.pdf").write_bytes(b"%PDF-1.4 this is not really a pdf")
    write(folder / "a.txt", "plain text survives")
    assistant.library.add("alice", folder)
    report = assistant.library.index("alice")
    assert report.added == 1 and "broken.pdf" in report.problems[0]


def test_search_survives_any_text(assistant, folder):
    write(folder / "a.txt", "alpha")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    for query in ('"', "*", "NEAR(", "a:b", "", "   ", "AND"):
        assert isinstance(assistant.library.search("alice", query), list)


# Incremental indexing


def test_only_what_changed_is_read(assistant, folder, monkeypatch):
    write(folder / "same.txt", "stays the same")
    changing = write(folder / "change.txt", "the old harbour")
    assistant.library.add("alice", folder)
    first = assistant.library.index("alice")
    assert (first.added, first.updated, first.unchanged) == (2, 0, 0)

    opened = []
    real_open = open

    def watching(path, *arguments, **options):
        opened.append(os.path.basename(str(path)))
        return real_open(path, *arguments, **options)

    monkeypatch.setattr("builtins.open", watching)
    second = assistant.library.index("alice")
    assert (second.added, second.updated, second.unchanged) == (0, 0, 2) and opened == []

    write(changing, "the new airport!")
    os.utime(changing, ns=(10**18, 10**18))
    third = assistant.library.index("alice")
    monkeypatch.undo()
    assert (third.updated, third.unchanged) == (1, 1) and opened == ["change.txt"]
    assert found(assistant.library.search("alice", "airport")) == {"change.txt"}
    assert assistant.library.search("alice", "harbour") == []


def test_a_touched_file_with_the_same_bytes_is_not_chunked_again(assistant, folder):
    source = write(folder / "a.txt", "some words")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    before = assistant.store.one("SELECT id FROM library_chunks")["id"]
    os.utime(source, ns=(10**18, 10**18))
    report = assistant.library.index("alice")
    assert (report.updated, report.unchanged) == (0, 1)
    assert assistant.store.one("SELECT id FROM library_chunks")["id"] == before
    assert assistant.library.index("alice").unchanged == 1


def test_what_was_deleted_is_removed(assistant, folder):
    gone = write(folder / "deleted.txt", "the vanishing word")
    write(folder / "kept.txt", "the staying word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    gone.unlink()
    assert assistant.library.index("alice").removed == 1
    assert assistant.library.search("alice", "vanishing") == []
    assert found(assistant.library.search("alice", "staying")) == {"kept.txt"}
    assert assistant.store.one("SELECT count(*) AS total FROM library_chunks")["total"] == 1


def test_a_folder_that_cannot_be_reached_keeps_its_index(assistant, folder):
    write(folder / "a.txt", "the unplugged word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    hidden = folder.with_name("moved")
    folder.rename(hidden)
    report = assistant.library.index("alice")
    assert report.removed == 0 and "cannot be reached" in report.problems[0]
    assert assistant.library.search("alice", "unplugged")


def test_a_folder_that_cannot_be_listed_keeps_what_was_indexed(assistant, folder, monkeypatch):
    write(folder / "a.txt", "a root word")
    write(folder / "sub" / "b.txt", "a nested word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    real = os.scandir

    def refusing(path):
        if os.path.basename(str(path)) == "sub":
            raise PermissionError("denied")
        return real(path)

    monkeypatch.setattr(os, "scandir", refusing)
    report = assistant.library.index("alice")
    assert report.removed == 0 and any("could not be read" in problem for problem in report.problems)
    assert assistant.library.search("alice", "nested")


def test_a_root_that_cannot_be_listed_keeps_everything(assistant, folder, monkeypatch):
    write(folder / "a.txt", "a root word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")

    def refusing(path):
        raise PermissionError("denied")

    monkeypatch.setattr(os, "scandir", refusing)
    assert assistant.library.index("alice").removed == 0
    monkeypatch.undo()
    assert assistant.library.search("alice", "root")


def test_a_locked_file_is_not_treated_as_deleted(assistant, folder, monkeypatch):
    source = write(folder / "a.txt", "the locked word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    write(source, "the locked word, changed")
    os.utime(source, ns=(10**18, 10**18))
    real_open = open

    def refusing(path, *arguments, **options):
        if str(path).endswith("a.txt"):
            raise PermissionError("in use")
        return real_open(path, *arguments, **options)

    monkeypatch.setattr("builtins.open", refusing)
    report = assistant.library.index("alice")
    monkeypatch.undo()
    assert report.removed == 0 and report.skipped == 1
    assert assistant.library.search("alice", "locked")


def test_removing_a_root_removes_its_index_and_not_the_files(assistant, folder):
    source = write(folder / "a.txt", "a removable word")
    root = assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assistant.library.remove("alice", root)
    assert assistant.library.search("alice", "removable") == []
    assert assistant.store.one("SELECT count(*) AS total FROM library_chunks")["total"] == 0
    assert assistant.store.one("SELECT count(*) AS total FROM library_fts WHERE library_fts MATCH 'removable'")["total"] == 0
    assert source.exists()


# Secrets and walls


@pytest.mark.parametrize(
    "name",
    [".env", ".env.local", ".ENV", "server.key", "SERVER.KEY", "cert.pem", "id_rsa", "id_rsa.pub", "ID_RSA", "credentials", "credentials.json", "Credentials.txt"],
)
def test_secret_names_are_recognised_in_any_case(name):
    assert is_secret(name)


@pytest.mark.parametrize("name", ["notes.md", "environment.txt", "keys.txt", "pem.txt", "my_credentials_doc.md"])
def test_ordinary_names_are_not_secrets(name):
    assert not is_secret(name)


def test_secrets_git_and_binary_files_are_never_indexed(assistant, folder):
    needle = "zanzibarneedle"
    for name in (".env", ".env.production", "server.key", "cert.pem", "id_rsa", "credentials.json", "CREDENTIALS"):
        write(folder / name, needle)
    write(folder / ".git" / "config", needle)
    write(folder / "sub" / ".git" / "HEAD", needle)
    write(folder / "credentials.d" / "inner.txt", needle)
    write(folder / ".env.d" / "inner.txt", needle)
    (folder / "image.bin").write_bytes(b"\x00\x01\x02" + needle.encode())
    write(folder / "ok.txt", "ordinary words")
    assistant.library.add("alice", folder)
    report = assistant.library.index("alice")
    assert report.added == 1
    assert assistant.library.search("alice", needle) == []
    assert needle.encode() not in assistant.store.path.read_bytes()


def test_a_file_that_becomes_binary_leaves_the_index(assistant, folder):
    source = write(folder / "a.dat", "readable word")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    source.write_bytes(b"\x00\x00binary")
    os.utime(source, ns=(10**18, 10**18))
    assert assistant.library.index("alice").removed == 1
    assert assistant.library.search("alice", "readable") == []


def link_to(link: Path, target: Path) -> None:
    """A real directory link: a symlink, or a junction where those need privileges."""
    try:
        link.symlink_to(target, target_is_directory=target.is_dir())
    except OSError:
        if os.name != "nt" or not target.is_dir():
            pytest.skip("this machine cannot make links")
        subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def junction_to(link: Path, target: Path) -> None:
    subprocess.run(["cmd", "/c", "mklink", "/J", str(link), str(target)], check=True, capture_output=True)


def test_a_folder_link_is_never_followed_out_of_the_root(assistant, folder, tmp_path):
    outside = tmp_path / "private"
    write(outside / "diary.txt", "the outsideword secret")
    link_to(folder / "escape", outside)
    write(folder / "ok.txt", "ordinary")
    assistant.library.add("alice", folder)
    assert assistant.library.index("alice").added == 1
    assert assistant.library.search("alice", "outsideword") == []


@pytest.mark.skipif(os.name != "nt", reason="junctions are a Windows feature")
def test_a_junction_is_never_followed_out_of_the_root(assistant, folder, tmp_path):
    outside = tmp_path / "private"
    write(outside / "diary.txt", "the junctionword secret")
    junction_to(folder / "escape", outside)
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assert assistant.library.search("alice", "junctionword") == []


def test_a_file_link_is_never_followed_out_of_the_root(assistant, folder, tmp_path):
    outside = write(tmp_path / "secret.txt", "the filelinkword secret")
    try:
        (folder / "escape.txt").symlink_to(outside)
    except OSError:
        pytest.skip("this machine cannot make file links")
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assert assistant.library.search("alice", "filelinkword") == []


def test_a_root_given_as_a_link_is_resolved_when_authorized(assistant, folder, tmp_path):
    write(folder / "a.txt", "a resolved word")
    link = tmp_path / "shortcut"
    link_to(link, folder)
    root = assistant.library.add("alice", link)
    assert assistant.library.roots("alice")[0]["path"] == str(folder.resolve()) and root


# Users are separated


def test_one_users_folder_is_invisible_to_another(assistant, folder):
    write(folder / "a.txt", "alice private walrus")
    root = assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assert assistant.library.search("bob", "walrus") == []
    assert assistant.library.roots("bob") == []
    assert assistant.library.chunk_text("bob", f"{folder / 'a.txt'}:1") is None
    assert assistant.library.index("bob").added == 0
    for attempt in (
        lambda: assistant.library.remove("bob", root),
        lambda: assistant.library.index("bob", root),
    ):
        with pytest.raises(LibraryRefused):
            attempt()
    assert assistant.library.search("alice", "walrus")


def test_two_users_may_authorize_the_same_folder_with_their_own_reach(assistant, folder):
    write(folder / "a.txt", "a shared otter")
    assistant.library.add("alice", folder, Reach.MACHINE)
    assistant.library.add("bob", folder, Reach.CLOUD)
    assistant.library.index("alice")
    assistant.library.index("bob")
    assert [item.provenance.reach for item in assistant.library.search("alice", "otter")] == [Reach.MACHINE]
    assert [item.provenance.reach for item in assistant.library.search("bob", "otter")] == [Reach.CLOUD]


def test_adding_twice_and_missing_paths_say_what_is_wrong(assistant, folder, tmp_path):
    assistant.library.add("alice", folder)
    with pytest.raises(LibraryRefused, match="already"):
        assistant.library.add("alice", folder)
    with pytest.raises(LibraryRefused, match="does not exist"):
        assistant.library.add("alice", tmp_path / "nowhere")
    with pytest.raises(LibraryRefused, match="library list"):
        assistant.library.remove("alice", 999)


# Vectors


class FakeEmbedder:
    """Maps a word to a fixed direction, as a stand-in for a real embedding provider."""

    def __init__(self, refuse_reach: Reach | None = None) -> None:
        self.calls: list[tuple[int, Reach]] = []
        self.refuse_reach = refuse_reach

    def can_embed(self) -> bool:
        return True

    def embed(self, texts, reach):
        self.calls.append((len(texts), reach))
        if self.refuse_reach is not None and reach <= self.refuse_reach:
            raise RuntimeError("no embedding provider may see this")
        return [self._vector(text) for text in texts], "fake-embed"

    @staticmethod
    def _vector(text: str) -> list[float]:
        lowered = text.casefold()
        return [float("dog" in lowered or "canine" in lowered), float("cat" in lowered or "feline" in lowered), 0.01]


def test_hybrid_search_finds_what_shares_no_word_with_the_query(assistant, folder):
    write(folder / "a.txt", "All about the canine companion")
    write(folder / "b.txt", "All about the feline companion")
    embedder = FakeEmbedder()
    assistant.router = embedder
    assistant.library.add("alice", folder)
    assert assistant.library.index("alice").embedded == 2
    assert found(assistant.library.search("alice", "dog", limit=1)) == {"a.txt"}
    assert assistant.store.one("SELECT length(vector) AS size FROM library_vectors")["size"] == 12


def test_only_new_chunks_are_embedded(assistant, folder):
    write(folder / "a.txt", "All about the canine")
    assistant.router = FakeEmbedder()
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assert assistant.library.index("alice").embedded == 0


def test_keyword_search_stands_alone_when_nothing_embeds(assistant, folder):
    write(folder / "a.txt", "All about the canine")
    assistant.library.add("alice", folder)
    report = assistant.library.index("alice")
    assert report.embedded == 0 and report.problems == []
    assert found(assistant.library.search("alice", "canine")) == {"a.txt"}
    assert assistant.library.search("alice", "dog") == []


def test_embedding_respects_a_folders_reach(assistant, folder):
    write(folder / "a.txt", "All about the canine")
    embedder = FakeEmbedder(refuse_reach=Reach.MACHINE)
    assistant.router = embedder
    assistant.library.add("alice", folder, Reach.MACHINE)
    report = assistant.library.index("alice")
    assert embedder.calls == [(1, Reach.MACHINE)]
    assert report.embedded == 0 and "Keyword search only" in report.problems[0]
    assert found(assistant.library.search("alice", "canine", reach=Reach.MACHINE)) == {"a.txt"}


def test_vectors_are_kept_per_user(assistant, folder, tmp_path):
    other = tmp_path / "bobs"
    write(other / "b.txt", "A private canine note")
    write(folder / "a.txt", "All about the feline")
    assistant.router = FakeEmbedder()
    assistant.library.add("alice", folder)
    assistant.library.add("bob", other)
    assistant.library.index("alice")
    assistant.library.index("bob")
    assert found(assistant.library.search("alice", "dog", limit=5)) == {"a.txt"}


def test_vectors_of_another_size_are_ignored(assistant, folder):
    write(folder / "a.txt", "All about the canine")
    assistant.router = FakeEmbedder()
    assistant.library.add("alice", folder)
    assistant.library.index("alice")
    assistant.store.execute("UPDATE library_vectors SET vector = ?", (np.zeros(5, dtype=np.float32).tobytes(),))
    assert found(assistant.library.search("alice", "canine")) == {"a.txt"}


def test_reciprocal_rank_fusion_rewards_agreement():
    scores = fuse([[1, 2, 3], [3, 2, 4]])
    assert max(scores, key=scores.get) == 2 or scores[3] > scores[1]
    assert scores[2] > scores[4] and scores[3] > scores[4]


# Commands


def test_commands_add_index_list_search_remove(tmp_path, folder, capsys):
    home = tmp_path / "home"
    write(folder / "a.md", "# Lighthouse\n\nThe keeper's log.\n")
    assert run_cli(home, "library", "add", str(folder), "--reach", "machine") == 0
    assert run_cli(home, "library", "index") == 0
    assert run_cli(home, "library", "list") == 0
    assert run_cli(home, "library", "search", "keeper") == 0
    shown = capsys.readouterr().out
    assert "1 added" in shown and "machine" in shown and f"{folder / 'a.md'}:1" in shown
    assert run_cli(home, "library", "remove", "1") == 0
    assert run_cli(home, "library", "list") == 0
    assert "The library is empty" in capsys.readouterr().out


def test_the_module_exposes_the_documented_skips():
    assert {".git"} <= knowledge.SKIPPED_FOLDERS
    assert {".env*", "*.key", "*.pem", "id_rsa*", "credentials*"} <= set(knowledge.SECRET_NAMES)
