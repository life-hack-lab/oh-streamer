"""
What the player accepts as "open this" (input from the main app, and the development `--player <input>` argument):

  * a single http(s) URL or local file path (surrounding quotes / whitespace stripped; file:/// URLs become paths)
  * a URL followed by lines of "Header-Name: value" (Referer, User-Agent, Origin, Cookie, ...)
  * DevTools "Copy as cURL": bash flavour (curl 'url' -H 'k: v' ... with \\ line continuations, $'...' quoting)
    and cmd flavour (curl ^"url^" -H ^"k: v^" ... with ^ line continuations), also pasted into PowerShell
    (backtick continuations); -b/--cookie, -A/--user-agent, -e/--referer, -u/--user, --compressed etc.
  * DevTools "Copy as PowerShell" (Invoke-WebRequest -Uri ... -Headers @{...}, $session.Cookies.Add(...))

parse() is pure (no I/O): see test_open_input.py. Errors are OpenInputError with a short Traditional Chinese
message for the OSD.
"""

import base64
import dataclasses
import re
from dataclasses import dataclass, field
from urllib.parse import unquote, urlparse

MAX_INPUT = 1_000_000          # characters; anything longer is not something we can open


class OpenInputError(ValueError):
    pass


@dataclass
class OpenInput:
    target: str                                  # http(s) URL or local path
    headers: dict = field(default_factory=dict)  # extra HTTP headers (URL inputs only)
    kind: str = "url"                            # "url" | "file"
    flavor: str = "plain"                        # plain | headers | curl-bash | curl-cmd | powershell
    notes: list = field(default_factory=list)    # ignored bits, for the log
    text: str = field(default="", compare=False, repr=False)   # parse_many: the pasted text this entry came from

    def describe(self) -> str:
        h = f", headers: {', '.join(self.headers)}" if self.headers else ""
        return f"{self.kind} via {self.flavor}{h}"


# headers that must not be forwarded to ffmpeg / yt-dlp (they describe one particular browser request)
DROP_HEADERS = frozenset({
    "accept-encoding", "content-length", "content-type", "host", "connection", "range", "if-none-match",
    "if-modified-since", "if-range", "te", "upgrade", "transfer-encoding", "expect", "keep-alive",
    "proxy-connection",
    "authority", "method", "path", "scheme",      # HTTP/2 pseudo headers as DevTools' PowerShell export writes them
})

_HEADER_RE = re.compile(r"^([A-Za-z0-9!#$%&'*+.^_`|~-]+)\s*:\s?(.*)$")
_SCHEME_RE = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.-]*://")


def is_http_url(s: str) -> bool:
    return bool(re.match(r"^https?://\S", s, re.IGNORECASE))


def _strip_quotes(s: str) -> str:
    s = s.strip()
    while len(s) >= 2 and (s[0] == s[-1] and s[0] in "\"'" or s[0] + s[-1] in ("“”", "‘’")):
        s = s[1:-1].strip()
    return s


def file_url_to_path(url: str) -> str:
    """file:///C:/a%20b.mp4 -> C:\\a b.mp4 ; file://server/share/x -> \\\\server\\share\\x"""
    p = urlparse(url)
    path = unquote(p.path)
    if p.netloc and p.netloc.lower() != "localhost":
        return "\\\\" + p.netloc + path.replace("/", "\\")
    if re.match(r"^/[A-Za-z]:", path):
        path = path[1:]
    return path.replace("/", "\\") if re.match(r"^[A-Za-z]:", path) else path


def _looks_like_path(s: str) -> bool:
    """C:\\..., C:/..., \\\\server\\share"""
    return bool(re.match(r"^([A-Za-z]:[\\/]|\\\\[^\\])", s))


def _add_header(headers: dict, name: str, value: str, notes: list):
    name, value = name.strip(), value.strip()
    if not name:
        return
    if name.startswith(":"):
        notes.append(f"pseudo header {name} ignored")
        return
    if name.lower() in DROP_HEADERS:
        notes.append(f"header {name} ignored")
        return
    for k in list(headers):
        if k.lower() == name.lower():
            if name.lower() == "cookie" and headers[k] and value:
                headers[k] = headers[k].rstrip("; ") + "; " + value
                return
            del headers[k]
    headers[name] = value


def _target(s: str, flavor: str, headers: dict, notes: list) -> OpenInput:
    s = _strip_quotes(s)
    if not s:
        raise OpenInputError("沒有可開啟的網址或檔案")
    if is_http_url(s):
        if any(c.isspace() for c in s):
            raise OpenInputError("網址含有空白，無法開啟")
        return OpenInput(s, headers, "url", flavor, notes)
    if s.lower().startswith("file://"):
        return OpenInput(file_url_to_path(s), {}, "file", flavor, notes)
    if _SCHEME_RE.match(s):
        raise OpenInputError(f"不支援的網址類型：{s.split(':', 1)[0]}")
    if "\n" in s or len(s) > 1024:
        raise OpenInputError("剪貼簿內容不是網址或檔案路徑")
    if headers:
        notes.append("headers ignored for a local file")
    return OpenInput(s, {}, "file", flavor, notes)


# ---------------------------------------------------------------------------------------------------------------
# tokenizers
# ---------------------------------------------------------------------------------------------------------------

_ANSI_C = {"n": "\n", "r": "\r", "t": "\t", "a": "\a", "b": "\b", "e": "\x1b", "E": "\x1b", "f": "\f", "v": "\v",
           "\\": "\\", "'": "'", '"': '"', "?": "?"}


def split_posix(s: str) -> list[str]:
    """POSIX-shell-like word splitting: '...', "...", $'...' (ANSI-C), backslash escapes, \\<newline> continuation.
    A backtick or caret at the very end of a line (PowerShell / cmd continuation) is treated as a continuation too."""
    out, cur, have = [], [], False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c in " \t\r\n":
            if have:
                out.append("".join(cur))
                cur, have = [], False
            i += 1
        elif c == "\\":
            if i + 1 < n and s[i + 1] == "\n":
                i += 2
            elif i + 2 < n and s[i + 1] == "\r" and s[i + 2] == "\n":
                i += 3
            elif i + 1 < n:
                cur.append(s[i + 1])
                have = True
                i += 2
            else:
                i += 1
        elif c in "`^" and re.match(r"[ \t]*\r?\n", s[i + 1:]) and not have:
            i += 1   # continuation marker standing alone at the end of a line
        elif c == "'":
            j = s.find("'", i + 1)
            if j < 0:
                raise OpenInputError("cURL 指令的引號不完整")
            cur.append(s[i + 1:j])
            have = True
            i = j + 1
        elif c == "$" and i + 1 < n and s[i + 1] == "'":
            i += 2
            while True:
                if i >= n:
                    raise OpenInputError("cURL 指令的引號不完整")
                c = s[i]
                if c == "'":
                    i += 1
                    break
                if c == "\\" and i + 1 < n:
                    e = s[i + 1]
                    if e in _ANSI_C:
                        cur.append(_ANSI_C[e])
                        i += 2
                    elif e in "uU" or e == "x":
                        width = {"u": 4, "U": 8, "x": 2}[e]
                        m = re.match(r"[0-9a-fA-F]{1,%d}" % width, s[i + 2:])
                        if m:
                            cur.append(chr(int(m.group(0), 16)))
                            i += 2 + len(m.group(0))
                        else:
                            cur.append("\\" + e)
                            i += 2
                    elif e in "01234567":
                        m = re.match(r"[0-7]{1,3}", s[i + 1:])
                        cur.append(chr(int(m.group(0), 8)))
                        i += 1 + len(m.group(0))
                    else:
                        cur.append("\\" + e)
                        i += 2
                else:
                    cur.append(c)
                    i += 1
            have = True
        elif c == '"':
            i += 1
            while True:
                if i >= n:
                    raise OpenInputError("cURL 指令的引號不完整")
                c = s[i]
                if c == '"':
                    i += 1
                    break
                if c == "\\" and i + 1 < n and s[i + 1] in '\\"$`\n':
                    if s[i + 1] != "\n":
                        cur.append(s[i + 1])
                    i += 2
                else:
                    cur.append(c)
                    i += 1
            have = True
        else:
            cur.append(c)
            have = True
            i += 1
    if have:
        out.append("".join(cur))
    return out


def _cmd_unescape(s: str) -> str:
    """cmd.exe layer: ^<newline> = line continuation, ^x = literal x ('^\\n\\n' from DevTools = a literal newline)."""
    out, i, n = [], 0, len(s)
    while i < n:
        c = s[i]
        if c == "^" and i + 1 < n:
            nxt = s[i + 1]
            if nxt == "\r" and i + 2 < n and s[i + 2] == "\n":
                i += 3
                if i < n and s[i] == "\n":      # escaped newline (^ + two newlines): keep one
                    out.append("\n")
                    i += 1
                continue
            if nxt == "\n":
                i += 2
                if i < n and s[i] == "\n" and _inside_quotes(out):
                    out.append("\n")
                    i += 1
                continue
            out.append(nxt)
            i += 2
        else:
            out.append(c)
            i += 1
    return "".join(out)


def _inside_quotes(chars: list) -> bool:
    q, bs = False, 0
    for c in chars:
        if c == "\\":
            bs += 1
            continue
        if c == '"' and bs % 2 == 0:
            q = not q
        bs = 0
    return q


def split_windows(s: str) -> list[str]:
    """MS C runtime argument splitting (what curl.exe sees after cmd.exe removed its carets)."""
    out, cur, have, inq = [], [], False, False
    i, n = 0, len(s)
    while i < n:
        c = s[i]
        if c == "\\":
            j = i
            while j < n and s[j] == "\\":
                j += 1
            nbs = j - i
            if j < n and s[j] == '"':
                cur.append("\\" * (nbs // 2))
                if nbs % 2:
                    cur.append('"')
                    i = j + 1
                else:
                    i = j
                have = True
                continue
            cur.append("\\" * nbs)
            have = True
            i = j
            continue
        if c == '"':
            if inq and i + 1 < n and s[i + 1] == '"':   # "" inside quotes = literal quote
                cur.append('"')
                i += 2
                continue
            inq = not inq
            have = True
            i += 1
            continue
        if c in " \t\r\n" and not inq:
            if have:
                out.append("".join(cur))
                cur, have = [], False
            i += 1
            continue
        cur.append(c)
        have = True
        i += 1
    if inq:
        raise OpenInputError("cURL 指令的引號不完整")
    if have:
        out.append("".join(cur))
    return out


# ---------------------------------------------------------------------------------------------------------------
# curl
# ---------------------------------------------------------------------------------------------------------------

# curl options that take a value (long and short forms); everything else starting with "-" is a flag
_ARG_OPTS = {
    "-H", "--header", "-b", "--cookie", "-A", "--user-agent", "-e", "--referer", "-u", "--user", "--url",
    "-X", "--request", "-d", "--data", "--data-raw", "--data-binary", "--data-ascii", "--data-urlencode",
    "-F", "--form", "--form-string", "-o", "--output", "-x", "--proxy", "-m", "--max-time", "--connect-timeout",
    "-c", "--cookie-jar", "-r", "--range", "--retry", "-w", "--write-out", "-T", "--upload-file", "-K", "--config",
    "--cacert", "--capath", "-E", "--cert", "--key", "--resolve", "--connect-to", "--interface", "-U",
    "--proxy-user", "--limit-rate", "-Y", "--speed-limit", "-y", "--speed-time", "--max-redirs", "--json",
    "--oauth2-bearer", "--aws-sigv4", "--happy-eyeballs-timeout-ms", "--expect100-timeout", "--dns-servers",
}


def _is_curl(text: str) -> bool:
    return bool(re.match(r"^\s*curl(\.exe)?\s", text, re.IGNORECASE))


def _curl_flavor(text: str) -> str:
    if re.search(r'\^"', text) or re.search(r"\s\^\r?\n", text):
        return "curl-cmd"
    return "curl-bash"


def parse_curl(text: str) -> OpenInput:
    flavor = _curl_flavor(text)
    if flavor == "curl-cmd":
        tokens = split_windows(_cmd_unescape(text.strip()))
    else:
        tokens = split_posix(text.strip())
    if not tokens or not re.match(r"^curl(\.exe)?$", tokens[0], re.IGNORECASE):
        raise OpenInputError("無法解析 cURL 指令")
    headers, notes, url = {}, [], None
    i = 1
    while i < len(tokens):
        t = tokens[i]
        opt, val = t, None
        if t.startswith("--") and "=" in t and t.split("=", 1)[0] in _ARG_OPTS:
            opt, val = t.split("=", 1)
        elif re.match(r"^-[A-Za-z]", t) and not t.startswith("--") and len(t) > 2 and t[:2] in _ARG_OPTS:
            opt, val = t[:2], t[2:]   # -H'x: y' / -bname=value glued to the option
        if opt in _ARG_OPTS:
            if val is None:
                if i + 1 >= len(tokens):
                    notes.append(f"{opt} without a value")
                    break
                val = tokens[i + 1]
                i += 1
            if opt in ("-H", "--header"):
                m = _HEADER_RE.match(val.strip())
                if m:
                    _add_header(headers, m.group(1), m.group(2), notes)
                elif val.strip().endswith(";"):
                    _add_header(headers, val.strip()[:-1], "", notes)   # curl's "Name;" = empty header
                else:
                    notes.append(f"bad header {val!r} ignored")
            elif opt in ("-b", "--cookie"):
                if "=" in val:
                    _add_header(headers, "Cookie", val, notes)
                else:
                    notes.append("cookie file ignored")
            elif opt in ("-A", "--user-agent"):
                _add_header(headers, "User-Agent", val, notes)
            elif opt in ("-e", "--referer"):
                ref = val[:-5] if val.endswith(";auto") else val
                if ref and ref != "auto":
                    _add_header(headers, "Referer", ref, notes)
            elif opt in ("-u", "--user"):
                _add_header(headers, "Authorization", "Basic " + base64.b64encode(val.encode()).decode(), notes)
            elif opt == "--url":
                url = url or val
            elif opt in ("-X", "--request"):
                if val.upper() not in ("GET", "HEAD"):
                    notes.append(f"request method {val} ignored (GET is used)")
            elif opt.startswith("-d") or opt.startswith("--data") or opt in ("-F", "--form", "--json"):
                notes.append(f"{opt} ignored (request body)")
            else:
                notes.append(f"{opt} ignored")
        elif t.startswith("-") and len(t) > 1:
            if t not in ("--compressed", "-s", "--silent", "-L", "--location", "-g", "--globoff", "-i", "-v",
                         "-k", "--insecure", "--http2", "--http1.1", "-S", "--show-error", "-f", "--fail"):
                notes.append(f"{t} ignored")
            elif t in ("-k", "--insecure"):
                notes.append(f"{t} ignored (certificates are always checked)")
        else:
            url = url or t
        i += 1
    if not url:
        raise OpenInputError("cURL 指令中找不到網址")
    return _target(url, flavor, headers, notes)


# ---------------------------------------------------------------------------------------------------------------
# PowerShell (DevTools "Copy as PowerShell")
# ---------------------------------------------------------------------------------------------------------------

def _ps_string(s: str) -> str:
    """Body of a PowerShell "..." string: `" `` `n `t escapes, "" = literal quote."""
    s = s.replace('""', '"')
    return re.sub(r"`(.)", lambda m: {"n": "\n", "t": "\t", "r": "\r", "0": "\0"}.get(m.group(1), m.group(1)), s)


_PS_STR = r'"((?:[^"`]|`.|"")*)"' + r"|'((?:[^']|'')*)'"


def _ps_value(m, base: int) -> str:
    if m.group(base) is not None:
        return _ps_string(m.group(base))
    return m.group(base + 1).replace("''", "'")


def _ps_block_end(text: str, i: int) -> int:
    """Index of the "}" closing the @{ block whose body starts at i (quoted strings are skipped)."""
    depth, n = 1, len(text)
    while i < n:
        c = text[i]
        if c in "\"'":
            i += 1
            while i < n:
                if c == '"' and text[i] == "`":
                    i += 2
                    continue
                if text[i] == c:
                    if text[i + 1:i + 2] == c:   # doubled quote = literal quote
                        i += 2
                        continue
                    break
                i += 1
        elif c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    return n


def parse_powershell(text: str) -> OpenInput:
    notes, headers = [], {}
    m = re.search(r"-Uri\s+(?:" + _PS_STR + ")", text, re.IGNORECASE)
    if not m:
        raise OpenInputError("PowerShell 指令中找不到網址")
    url = _ps_value(m, 1)
    for ua in re.finditer(r"\.UserAgent\s*=\s*(?:" + _PS_STR + ")", text):
        _add_header(headers, "User-Agent", _ps_value(ua, 1), notes)
    ua = re.search(r"-UserAgent\s+(?:" + _PS_STR + ")", text, re.IGNORECASE)
    if ua:
        _add_header(headers, "User-Agent", _ps_value(ua, 1), notes)
    hm = re.search(r"-Headers\s+@\{", text, re.IGNORECASE)
    if hm:
        body = text[hm.end():_ps_block_end(text, hm.end())]
        pair = re.compile(r"(?:" + _PS_STR + r"|([A-Za-z0-9_-]+))\s*=\s*(?:" + _PS_STR + ")")
        for p in pair.finditer(body):
            name = _ps_value(p, 1) if (p.group(1) is not None or p.group(2) is not None) else p.group(3)
            _add_header(headers, name, _ps_value(p, 4), notes)
    cookies = []
    for c in re.finditer(r"System\.Net\.Cookie\(\s*(?:" + _PS_STR + r")\s*,\s*(?:" + _PS_STR + ")", text):
        cookies.append(f"{_ps_value(c, 1)}={_ps_value(c, 3)}")
    if cookies:
        _add_header(headers, "Cookie", "; ".join(cookies), notes)
    if re.search(r"-Method\s+[\"']?(POST|PUT|PATCH|DELETE)", text, re.IGNORECASE):
        notes.append("request method ignored (GET is used)")
    return _target(url, "powershell", headers, notes)


# ---------------------------------------------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------------------------------------------

def parse(text: str) -> OpenInput:
    """Clipboard / dropped / forwarded text -> OpenInput (target + headers). Raises OpenInputError."""
    if text is None:
        raise OpenInputError("剪貼簿沒有文字")
    if len(text) > MAX_INPUT:
        raise OpenInputError("剪貼簿內容太長")
    text = text.replace("\ufeff", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise OpenInputError("剪貼簿沒有文字")
    if _is_curl(text):
        return parse_curl(text)
    if re.search(r"Invoke-(WebRequest|RestMethod)\b", text, re.IGNORECASE):
        return parse_powershell(text)
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) == 1 or _looks_like_path(_strip_quotes(lines[0])):
        return _target(lines[0], "plain", {}, ["extra lines ignored"] if len(lines) > 1 else [])
    # URL (first URL-looking line) + "Name: value" lines
    headers, notes, url = {}, [], None
    for ln in lines:
        s = _strip_quotes(ln)
        if url is None and (is_http_url(s) or s.lower().startswith("file://")):
            url = s
            continue
        m = re.match(r"^(:[a-z]+)\s*:\s?(.*)$", ln) if ln.startswith(":") else _HEADER_RE.match(ln)
        if m and not _looks_like_path(s):
            _add_header(headers, m.group(1), m.group(2), notes)
        else:
            notes.append(f"line ignored: {ln[:60]!r}")
    if url is None:
        raise OpenInputError("找不到網址（第一行應為網址）" if headers else "剪貼簿內容不是網址或檔案路徑")
    return _target(url, "headers" if headers else "plain", headers, notes)


def _is_plain_line(ln: str) -> bool:
    """A line that stands alone as an input: an http(s) / file:// URL or a file path, not a "Name: value" header."""
    s = _strip_quotes(ln)
    if is_http_url(s) or s.lower().startswith("file://") or _looks_like_path(s):
        return True
    return not _HEADER_RE.match(s) and not _SCHEME_RE.match(s)


def parse_many(text: str, errors: list | None = None) -> list[OpenInput]:
    """Like parse(), but a pasted list becomes one OpenInput per line.

    cURL / PowerShell text is always one entry. Otherwise, with >= 2 non-empty lines that are all URLs or file paths
    (no "Name: value" header line among them) every line is its own entry (the same target twice is kept once);
    anything else goes to parse() (a URL followed by header lines = one entry). A bad line of a list is appended to
    `errors` (a short message each) when given, else raised as OpenInputError. Raises OpenInputError for text that is
    not an input at all, like parse()."""
    if text is None:
        raise OpenInputError("剪貼簿沒有文字")
    if len(text) > MAX_INPUT:
        raise OpenInputError("剪貼簿內容太長")
    text = text.replace("﻿", "").replace("\r\n", "\n").replace("\r", "\n").strip()
    if not text:
        raise OpenInputError("剪貼簿沒有文字")
    if _is_curl(text) or re.search(r"Invoke-(WebRequest|RestMethod)\b", text, re.IGNORECASE):
        return [dataclasses.replace(parse(text), text=text)]
    lines = [ln.strip() for ln in text.split("\n")]
    lines = [ln for ln in lines if ln]
    if len(lines) < 2 or not all(_is_plain_line(ln) for ln in lines):
        return [dataclasses.replace(parse(text), text=text)]
    out, seen = [], set()
    for ln in lines:
        try:
            oi = _target(ln, "plain", {}, [])
        except OpenInputError as e:
            if errors is None:
                raise
            errors.append(f"{ln[:60]}：{e}")
            continue
        if oi.target not in seen:
            seen.add(oi.target)
            out.append(dataclasses.replace(oi, text=ln))
    return out
