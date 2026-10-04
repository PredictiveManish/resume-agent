"""Command-line interface for the resume-tailoring agent."""

import argparse
import json
import sys

from .pipeline import tailor


def _print_preview(result) -> None:
    m = result.match
    print("=" * 68)
    print(f"  JD-MATCH SCORE: {m.score}/100   (heuristic — not an official ATS score)")
    print("=" * 68)
    print("  Subscores:")
    for k, v in m.subscores.items():
        print(f"    - {k}: {v}")
    print(f"\n  Matched keywords ({len(m.matched_keywords)}): "
          f"{', '.join(m.matched_keywords[:15]) or '(none)'}")
    print(f"  Missing keywords ({len(m.missing_keywords)}): "
          f"{', '.join(m.missing_keywords[:15]) or '(none)'}")

    if m.gaps:
        print("\n  Gaps:")
        for g in m.gaps:
            print(f"    - {g}")
    if m.suggestions:
        print("\n  Suggestions:")
        for s in m.suggestions:
            print(f"    - {s}")

    if result.proofs:
        print("\n  Proof checks:")
        for p in result.proofs:
            print(f"    [{'OK' if p.ok else 'X '}] {p.url}  {p.detail}")

    if result.changes:
        print(f"\n  Preview of changes ({len(result.changes)}):")
        for c in result.changes:
            print(f"\n    [{c.section}]")
            print(f"      before: {c.before}")
            print(f"      after : {c.after}")
            print(f"      why   : {c.reason}")
    print()


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="resume-tailor",
        description="Tailor a resume to a job description and get ATS-friendly LaTeX.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("tailor", help="Tailor a resume to a job description")
    g = t.add_mutually_exclusive_group(required=True)
    g.add_argument("--resume", help="Path to resume (.tex, .pdf, .txt)")
    g.add_argument("--resume-text", help="Resume as raw text")
    j = t.add_mutually_exclusive_group(required=True)
    j.add_argument("--jd", help="Path to job description text file")
    j.add_argument("--jd-text", help="Job description as raw text")
    t.add_argument("--proof", action="append", default=[],
                   help="Proof URL for a claim, e.g. a GitHub repo (repeatable)")
    t.add_argument("--out", help="Write the tailored .tex file here")
    t.add_argument("--json", help="Write the full result as JSON here")

    args = p.parse_args(argv)

    if args.resume_text:
        resume_arg, is_text = args.resume_text, True
    else:
        resume_arg, is_text = args.resume, False

    if args.jd_text:
        jd_text = args.jd_text
    else:
        with open(args.jd, "r", encoding="utf-8") as fh:
            jd_text = fh.read()

    try:
        result = tailor(resume_arg, jd_text, proof_urls=args.proof, is_text=is_text)
    except Exception as e:  # noqa: BLE001 — surface a clean CLI error
        print(f"Error: {e}", file=sys.stderr)
        return 1

    _print_preview(result)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(result.new_tex)
        print(f"Wrote tailored LaTeX -> {args.out}")
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result.to_dict(), fh, indent=2)
        print(f"Wrote full result  -> {args.json}")
    if not args.out and not args.json:
        print("--- tailored .tex ---\n")
        print(result.new_tex)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
