"""Offline, sentence-level blind translation review and validated vote scoring.

Run ``python -m oh_streamer.subtitles.blind_ab --help``. Existing subtitle
JSON without IDs is matched by exact timestamps AND Japanese source text;
translation order is never used as an alignment signal.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import random
from typing import Mapping

from .core import has_japanese_residue
from .postprocess import taiwan_text
from .sentences import wrap_two_lines


KEY_FORMAT = "oh_streamer.subtitle-blind-key.v1"
VOTE_FORMAT = "oh_streamer.subtitle-blind-votes.v1"
LABELS = "ABC"


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON field: {key}")
        result[key] = value
    return result


def _load(value):
    if isinstance(value, (str, Path)):
        return json.loads(Path(value).read_text("utf-8-sig"), object_pairs_hook=_unique_object)
    return value


def _rows(data):
    data = _load(data)
    rows = data.get("segments") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError("subtitle JSON must be a list or contain a segments list")
    return rows


def _text(row, name):
    value = row.get(name)
    if not isinstance(value, str):
        raise ValueError(f"cue {name} must be text")
    return value


def _time(row):
    values = tuple(row.get(name) for name in ("start", "end"))
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) for v in values):
        raise ValueError("cue timestamps must be finite numbers")
    if values[0] < 0 or values[1] < values[0]:
        raise ValueError("invalid cue timestamp range")
    return values


def _cue_id(row):
    value = row.get("id", row.get("cue_id"))
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int)) or value == "":
        raise ValueError("cue IDs must be nonempty strings or integers")
    if "id" in row and "cue_id" in row and row["id"] != row["cue_id"]:
        raise ValueError("conflicting cue IDs")
    # Avoid collisions between integer 1 and string "1".
    return (type(value).__name__, value)


def _align(source, translations):
    if not isinstance(translations, Mapping) or len(translations) not in (2, 3):
        raise ValueError("provide two or three named translation datasets")
    if any(not isinstance(name, str) or not name.strip() for name in translations):
        raise ValueError("model names must be nonempty text")
    rows = _rows(source)
    if not rows:
        raise ValueError("source has no cues")
    ids, signatures, sources = {}, {}, []
    for row in rows:
        start, end = _time(row)
        text = _text(row, "text")
        explicit_id = _cue_id(row)
        signature = (start, end, text)
        if explicit_id is not None:
            if explicit_id in ids:
                raise ValueError("duplicate source cue ID")
            ids[explicit_id] = len(sources)
        signatures.setdefault(signature, []).append(len(sources))
        sources.append({"start": start, "end": end, "text_ja": text,
                        "source_id": row.get("id", row.get("cue_id"))})
    aligned = {}
    for name, data in translations.items():
        matches, seen_ids = {}, set()
        for row in _rows(data):
            start, end = _time(row)
            japanese = _text(row, "text_ja")
            explicit_id = _cue_id(row)
            if explicit_id is not None:
                if explicit_id in seen_ids:
                    raise ValueError(f"duplicate translation cue ID: {name}")
                seen_ids.add(explicit_id)
            if explicit_id is not None and ids:
                if explicit_id not in ids:
                    raise ValueError(f"unknown translation cue ID: {name}")
                index = ids[explicit_id]
                expected = sources[index]
                if (start, end, japanese) != (expected["start"], expected["end"], expected["text_ja"]):
                    raise ValueError(f"source text or timestamps differ: {name}")
            else:
                candidates = signatures.get((start, end, japanese), [])
                if len(candidates) != 1:
                    raise ValueError(f"source/timestamps mismatch or ambiguous ID-less cue: {name}")
                index = candidates[0]
            if index in matches:
                raise ValueError(f"duplicate translation alignment: {name}")
            text = row.get("text", "")
            if text is None:
                text = ""
            if not isinstance(text, str):
                raise ValueError("translation text must be text or null")
            hidden = any(row.get(flag) is True for flag in ("hidden", "preserved_original"))
            hidden = hidden or row.get("display") is False
            hidden = hidden or "[原文]" in text or has_japanese_residue(text)
            status = "hidden" if hidden else ("present" if text.strip() else "missing")
            matches[index] = {
                "text": wrap_two_lines(taiwan_text(text)) if status == "present" else "",
                "status": status,
            }
        aligned[name] = matches
    return sources, aligned


def build_blind_review(source, translations: Mapping, *, seed=None):
    """Return (standalone HTML, separate answer key), accepting JSON paths/data.

    Omitted rows and hidden failures remain review candidates at their original
    cue IDs. ``seed`` reproduces both order and dataset ID. The dataset hash
    binds votes to the exact source, displayed translations and random mapping.
    """
    sources, aligned = _align(source, translations)
    rng = random.Random(seed)
    cues = []
    for index, source_row in enumerate(sources):
        models = list(translations)
        rng.shuffle(models)
        options = {}
        for label, model in zip(LABELS, models):
            options[label] = {"model": model, **aligned[model].get(index, {"text": "", "status": "missing"})}
        cues.append({"id": f"q{index + 1:06d}", **source_row, "options": options})
    key = {"format": KEY_FORMAT, "models": list(translations), "cues": cues}
    key["dataset_id"] = _digest(key)
    return _render(key), key


def _render(key):
    # Only opaque question IDs and A/B/C labels enter the page. In particular,
    # source paths, model IDs, filenames and answer mappings never enter HTML.
    sections = []
    for cue in key["cues"]:
        qid = cue["id"]
        choices = []
        for label, option in cue["options"].items():
            value = option["text"] if option["status"] == "present" else (
                "（未顯示譯文）" if option["status"] == "hidden" else "（缺少譯文）")
            choices.append(f'<label class="choice"><input type="radio" name="{qid}" value="{label}"> '
                           f'<b>{label}</b><span>{html.escape(value)}</span></label>')
        choices.extend(f'<label><input type="radio" name="{qid}" value="{value}"> {label}</label>'
                       for value, label in (("tie", "平手（全部相同）"), ("skip", "略過")))
        sections.append(f'<fieldset data-cue="{qid}"><legend>{qid} · {cue["start"]:.3f}–{cue["end"]:.3f} s</legend>'
                        f'<p lang="ja">{html.escape(cue["text_ja"])}</p>{"".join(choices)}</fieldset>')
    return '''<!doctype html>
<html lang="zh-Hant"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'">
<title>字幕盲測</title><style>
body{font:18px system-ui,sans-serif;max-width:900px;margin:2rem auto;padding:0 1rem;background:#f5f5f5;color:#222}
fieldset{margin:1rem 0;background:white;border:1px solid #bbb;border-radius:8px;padding:1rem}
label{display:block;padding:.5rem;cursor:pointer}.choice{border-bottom:1px solid #eee}
.choice span{display:block;white-space:pre-wrap;margin:.4rem 0 .3rem 1.7rem}button{font:inherit;padding:.6rem}
header{position:sticky;top:0;background:#f5f5f5;padding:.6rem 0}p{white-space:pre-wrap}
</style><header><h1>字幕盲測</h1><p>每句選出較佳譯文，或選全部平手／略過。未作答會計為棄權。作答會在瀏覽器允許時自動暫存於本機，完成後請匯出投票 JSON。</p>
<button id="export" type="button">匯出投票 JSON</button> <span id="progress" aria-live="polite"></span></header>
<main>''' + "\n".join(sections) + '''</main><script>
const datasetId = "''' + key["dataset_id"] + '''";
const fields = [...document.querySelectorAll('fieldset[data-cue]')];
const storageKey = 'ohStreamer.subtitle.blind.' + datasetId;
try {
  const saved = JSON.parse(localStorage.getItem(storageKey) || '{}');
  if (saved && typeof saved === 'object' && !Array.isArray(saved)) {
    for (const field of fields) {
      for (const input of field.querySelectorAll('input[type=radio]')) {
        input.checked = input.value === saved[field.dataset.cue];
      }
    }
  }
} catch (_) { /* Export still works when local storage is unavailable. */ }
function progress(){document.getElementById('progress').textContent =
  `${fields.filter(f=>f.querySelector('input:checked')).length} / ${fields.length} 已作答`;}
document.addEventListener('change', () => {
  progress();
  const saved = Object.fromEntries(fields.map(f =>
    [f.dataset.cue, f.querySelector('input:checked')?.value ?? null]));
  try { localStorage.setItem(storageKey, JSON.stringify(saved)); } catch (_) {}
}); progress();
document.getElementById('export').addEventListener('click', () => {
  const votes = fields.map(f => ({cue_id:f.dataset.cue,choice:f.querySelector('input:checked')?.value ?? 'skip'}));
  const payload = {format:"''' + VOTE_FORMAT + '''",dataset_id:datasetId,votes};
  const url = URL.createObjectURL(new Blob([JSON.stringify(payload,null,2)],{type:'application/json'}));
  const a = document.createElement('a'); a.href=url; a.download='blind_votes.json';
  document.body.appendChild(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);
});
</script></html>'''


def write_blind_review(source, translations: Mapping, out_dir, *, seed=None):
    """Write blind_ab.html and answer_key.json and return the answer key."""
    page, key = build_blind_review(source, translations, seed=seed)
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "blind_ab.html").write_text(page, "utf-8")
    (destination / "answer_key.json").write_text(json.dumps(key, ensure_ascii=False, indent=2), "utf-8")
    return key


def _validate_key(key):
    if not isinstance(key, dict) or set(key) != {"format", "dataset_id", "models", "cues"} or key["format"] != KEY_FORMAT:
        raise ValueError("invalid answer key format")
    if key["dataset_id"] != _digest({k: v for k, v in key.items() if k != "dataset_id"}):
        raise ValueError("answer key dataset hash mismatch")
    models = key["models"]
    if (not isinstance(models, list) or len(models) not in (2, 3)
            or any(not isinstance(m, str) or not m.strip() for m in models)
            or len(set(models)) != len(models)):
        raise ValueError("invalid answer key models")
    if not isinstance(key["cues"], list) or not key["cues"]:
        raise ValueError("invalid answer key cues")
    seen = set()
    for cue in key["cues"]:
        if not isinstance(cue, dict) or set(cue) != {"id", "source_id", "start", "end", "text_ja", "options"}:
            raise ValueError("invalid answer key cue")
        qid = cue["id"]
        if not isinstance(qid, str) or not qid or qid in seen:
            raise ValueError("duplicate or invalid answer key cue ID")
        seen.add(qid)
        _time(cue)
        _text(cue, "text_ja")
        options = cue["options"]
        if not isinstance(options, dict) or set(options) != set(LABELS[:len(models)]):
            raise ValueError("invalid answer key option labels")
        mapped = []
        for option in options.values():
            if (not isinstance(option, dict) or set(option) != {"model", "text", "status"}
                    or not isinstance(option["model"], str) or not isinstance(option["text"], str)
                    or option["status"] not in ("present", "hidden", "missing")):
                raise ValueError("invalid answer key option")
            mapped.append(option["model"])
        if set(mapped) != set(models):
            raise ValueError("answer key must map every model exactly once per cue")
    return {cue["id"]: cue for cue in key["cues"]}


def score_votes(answer_key, votes):
    """Score validated votes. Ties enter denominator; skip/unanswered abstain.

    win_rate = outright wins / (wins + losses + ties), or None if no rated
    cues. For three candidates a tie means all three tie; selecting a candidate
    gives that candidate one win and each other candidate one loss.
    Hash binding detects mixups and edits, and is not a digital signature.
    """
    key, votes = _load(answer_key), _load(votes)
    cues = _validate_key(key)
    if (not isinstance(votes, dict) or set(votes) != {"format", "dataset_id", "votes"}
            or votes["format"] != VOTE_FORMAT or votes["dataset_id"] != key["dataset_id"]
            or not isinstance(votes["votes"], list)):
        raise ValueError("invalid votes format or dataset binding")
    choices = {}
    for vote in votes["votes"]:
        if not isinstance(vote, dict) or set(vote) != {"cue_id", "choice"}:
            raise ValueError("invalid vote")
        qid, choice = vote["cue_id"], vote["choice"]
        if not isinstance(qid, str) or qid not in cues or qid in choices:
            raise ValueError("unknown or duplicate vote cue ID")
        if not isinstance(choice, str) or choice not in (*cues[qid]["options"], "tie", "skip"):
            raise ValueError("unknown vote choice")
        choices[qid] = choice
    results = {model: {"wins": 0, "losses": 0, "ties": 0, "abstentions": 0,
                       "missing_translations": 0, "hidden_translations": 0} for model in key["models"]}
    skipped = unanswered = ties = decisive = 0
    for qid, cue in cues.items():
        choice = choices.get(qid, "skip")
        if qid not in choices:
            unanswered += 1
        elif choice == "skip":
            skipped += 1
        if choice == "tie":
            ties += 1
        elif choice != "skip":
            decisive += 1
        for label, option in cue["options"].items():
            counts = results[option["model"]]
            if option["status"] in ("missing", "hidden"):
                counts[option["status"] + "_translations"] += 1
            counts["abstentions" if choice == "skip" else "ties" if choice == "tie" else "wins" if choice == label else "losses"] += 1
    for counts in results.values():
        counts["denominator"] = counts["wins"] + counts["losses"] + counts["ties"]
        counts["win_rate"] = counts["wins"] / counts["denominator"] if counts["denominator"] else None
    return {"dataset_id": key["dataset_id"], "total_cues": len(cues), "decisive_votes": decisive,
            "ties": ties, "skipped": skipped, "unanswered": unanswered,
            "rate_definition": "outright wins / (wins + losses + ties); skip and unanswered abstain",
            "models": results}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="create an offline review page and separate answer key")
    generate.add_argument("source")
    generate.add_argument("--translation", action="append", required=True, metavar="MODEL=JSON")
    generate.add_argument("--out", required=True)
    generate.add_argument("--seed", type=int)
    score = sub.add_parser("score", help="score exported votes against their answer key")
    score.add_argument("answer_key")
    score.add_argument("votes")
    score.add_argument("--out")
    args = parser.parse_args(argv)
    try:
        if args.command == "generate":
            translations = {}
            for entry in args.translation:
                model, separator, path = entry.partition("=")
                if not separator or not model or not path or model in translations:
                    raise ValueError("each --translation must be a distinct MODEL=JSON")
                translations[model] = Path(path)
            write_blind_review(args.source, translations, args.out, seed=args.seed)
        else:
            report = json.dumps(score_votes(args.answer_key, args.votes), ensure_ascii=False, indent=2)
            if args.out:
                Path(args.out).write_text(report + "\n", "utf-8")
            else:
                print(report)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
