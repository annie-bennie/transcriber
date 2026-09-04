# -*- coding: utf-8 -*-
"""Scoring speaker labelling against a hand-corrected reference.

the reference is an ordinary txt in the same format transcribe.py produces, with the speaker
labels fixed by hand. pieces are separated by a `---` line; numbering does not carry across
pieces, order matters inside one.

    python tools/eval_speakers.py reference.txt "transcripts/meeting.json"

labels are arbitrary: the reference's SPEAKER_00 need not be whoever the pipeline called
that. so both mappings are scored and the better one is taken — what is compared is the
division into speakers, not the names.
"""
import argparse
import json
import re
import sys
from pathlib import Path

LINE = re.compile(
    r"^\[(?P<speaker>\S+)\s+(?P<start>\d+:\d+(?::\d+)?)-(?P<end>\d+:\d+(?::\d+)?)\]"
    r"(?:\s*\([^)]*\))?\s*:\s*(?P<text>.*)$"
)


def parse_time(value: str) -> float:
    parts = [int(p) for p in value.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def read_reference(path: Path):
    """read the reference into a list of pieces, each a list of (start, end, speaker, text)."""
    chunks, current = [], []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("---"):
            if current:
                chunks.append(current)
                current = []
            continue
        match = LINE.match(line)
        if match:
            current.append((parse_time(match["start"]), parse_time(match["end"]),
                            match["speaker"], match["text"].strip()))
    if current:
        chunks.append(current)
    return chunks


def find_segment(segments, start, end):
    """the pipeline segment that best covers a reference line.

    the reference is rounded to a second, so matching goes by overlap, not by equality.
    """
    best, best_overlap = None, 0.0
    for seg in segments:
        overlap = min(end + 1.0, seg["end"]) - max(start, seg["start"])
        if overlap > best_overlap:
            best, best_overlap = seg, overlap
    return best


def evaluate(reference_chunks, segments, verbose=False):
    rows = []
    for number, chunk in enumerate(reference_chunks, 1):
        for start, end, speaker, text in chunk:
            seg = find_segment(segments, start, end)
            rows.append((number, start, speaker, text,
                         seg["speaker"] if seg else None,
                         seg.get("speaker_uncertain", False) if seg else False))

    found = [r for r in rows if r[4]]
    labels = sorted({r[4] for r in found})
    ref_labels = sorted({r[2] for r in found})
    mappings = []
    if len(labels) == 2 and len(ref_labels) == 2:
        mappings = [dict(zip(ref_labels, labels)), dict(zip(ref_labels, labels[::-1]))]
    else:
        mappings = [{r: r for r in ref_labels}]

    best_map, best_hits = None, -1
    for mapping in mappings:
        hits = sum(1 for r in found if mapping.get(r[2]) == r[4])
        if hits > best_hits:
            best_map, best_hits = mapping, hits

    print(f"reference lines: {len(rows)}, matched to segments: {len(found)}")
    print(f"correct labels: {best_hits}/{len(found)} "
          f"({best_hits / max(len(found), 1) * 100:.0f}%)")

    uncertain = [r for r in found if r[5]]
    if uncertain:
        wrong_unc = sum(1 for r in uncertain if best_map.get(r[2]) != r[4])
        print(f"marked uncertain: {len(uncertain)}, wrong among them: {wrong_unc}")
        confident = [r for r in found if not r[5]]
        good = sum(1 for r in confident if best_map.get(r[2]) == r[4])
        print(f"among the confident ones: {good}/{len(confident)} "
              f"({good / max(len(confident), 1) * 100:.0f}%)")

    # transitions: how many speaker changes inside a piece were caught.
    total_switches = correct_switches = 0
    for number in sorted({r[0] for r in found}):
        chunk = [r for r in found if r[0] == number]
        for a, b in zip(chunk, chunk[1:]):
            expected = best_map.get(a[2]) != best_map.get(b[2])
            got = a[4] != b[4]
            total_switches += 1
            correct_switches += int(expected == got)
    print(f"boundaries between neighbouring lines: {correct_switches}/{total_switches} correct")

    if verbose:
        print()
        for number, start, ref, text, got, unc in rows:
            mark = "  " if got and best_map.get(ref) == got else "XX"
            flag = "?" if unc else " "
            print(f"  {mark}{flag} piece {number} {start:7.1f} reference {ref} -> {got}  "
                  f"{text[:44]!r}")
    return best_hits, len(found)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("reference", help="txt with hand-corrected labels")
    parser.add_argument("state", nargs="+", help="json (several may be given, to compare)")
    parser.add_argument("--verbose", "-v", action="store_true", help="print every line")
    args = parser.parse_args()

    chunks = read_reference(Path(args.reference))
    print(f"reference: {sum(len(c) for c in chunks)} lines in {len(chunks)} pieces\n")
    for path in args.state:
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        print(f"=== {Path(path).name}")
        evaluate(chunks, state["segments"], args.verbose)
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
