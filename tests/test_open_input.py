"""Offline unit tests for open_input.py (run: python streamer/tests/test_open_input.py)."""
# Keep flat streamer modules importable during discovery and direct execution.
from pathlib import Path as _Path
import sys as _sys

_STREAMER_ROOT = _Path(__file__).resolve().parents[1]
if str(_STREAMER_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_STREAMER_ROOT))
_SOURCE_ROOT = _STREAMER_ROOT / "src"
if str(_SOURCE_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_SOURCE_ROOT))

import os
import sys
import unittest
import oh_streamer.player.open_input as O  # noqa: E402

URL = "https://cdn.example.com/hls/v1/index.m3u8?token=a%20b&exp=17"

CHROME_BASH = r"""curl 'https://cdn.example.com/hls/v1/index.m3u8?token=a%20b&exp=17' \
  -H 'accept: */*' \
  -H 'accept-language: zh-TW,zh;q=0.9,en;q=0.8' \
  -b 'sid=abc123; theme=dark' \
  -H 'origin: https://www.example.com' \
  -H 'priority: u=1, i' \
  -H 'referer: https://www.example.com/watch/42' \
  -H 'sec-ch-ua: "Chromium";v="130", "Google Chrome";v="130"' \
  -H 'sec-fetch-mode: cors' \
  -H 'user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36' \
  --compressed"""

CHROME_CMD = r"""curl ^"https://cdn.example.com/hls/v1/index.m3u8?token=a^%^20b^&exp=17^" ^
  -H ^"accept: */*^" ^
  -H ^"accept-language: zh-TW,zh;q=0.9,en;q=0.8^" ^
  -b ^"sid=abc123; theme=dark^" ^
  -H ^"origin: https://www.example.com^" ^
  -H ^"referer: https://www.example.com/watch/42^" ^
  -H ^"sec-ch-ua: ^\^"Chromium^\^";v=^\^"130^\^", ^\^"Google Chrome^\^";v=^\^"130^\^"^" ^
  -H ^"user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36^" ^
  --compressed"""

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
SEC_CH_UA = '"Chromium";v="130", "Google Chrome";v="130"'

POWERSHELL = r"""$session = New-Object Microsoft.PowerShell.Commands.WebRequestSession
$session.UserAgent = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/130.0 Safari/537.36"
$session.Cookies.Add((New-Object System.Net.Cookie("sid", "abc123", "/", ".example.com")))
$session.Cookies.Add((New-Object System.Net.Cookie("theme", "dark", "/", ".example.com")))
Invoke-WebRequest -UseBasicParsing -Uri "https://cdn.example.com/hls/v1/index.m3u8?token=a%20b&exp=17" `
-WebSession $session `
-Headers @{
"authority"="cdn.example.com"
  "method"="GET"
  "path"="/hls/v1/index.m3u8"
  "scheme"="https"
  "accept"="*/*"
  "accept-encoding"="gzip, deflate, br"
  "origin"="https://www.example.com"
  "referer"="https://www.example.com/watch/42"
  "sec-ch-ua"="`"Chromium`";v=`"130`""
}"""


def hdr(oi, name):
    for k, v in oi.headers.items():
        if k.lower() == name.lower():
            return v
    return None


class PlainTest(unittest.TestCase):
    def test_url(self):
        oi = O.parse("  " + URL + "\n")
        self.assertEqual((oi.target, oi.kind, oi.headers, oi.flavor), (URL, "url", {}, "plain"))

    def test_quoted_url(self):
        self.assertEqual(O.parse(f'"{URL}"').target, URL)
        self.assertEqual(O.parse(f"'{URL}'").target, URL)

    def test_windows_path_with_quotes(self):
        oi = O.parse('"C:\\Videos\\my clip (1).mp4"')
        self.assertEqual((oi.target, oi.kind), ("C:\\Videos\\my clip (1).mp4", "file"))

    def test_unc_and_forward_slash_paths(self):
        self.assertEqual(O.parse(r"\\nas\share\a.mkv").target, r"\\nas\share\a.mkv")
        self.assertEqual(O.parse("D:/x/y.mp4").kind, "file")

    def test_file_url(self):
        oi = O.parse("file:///C:/Videos/a%20b.mp4")
        self.assertEqual((oi.target, oi.kind), ("C:\\Videos\\a b.mp4", "file"))

    def test_bom_and_crlf(self):
        self.assertEqual(O.parse("\ufeff" + URL + "\r\n").target, URL)

    def test_empty(self):
        for t in ("", "   \n  ", None):
            with self.assertRaises(O.OpenInputError):
                O.parse(t)

    def test_other_scheme_rejected(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("rtmp://example.com/live")

    def test_too_long(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("x" * (O.MAX_INPUT + 1))

    def test_multiple_paths_take_first(self):
        oi = O.parse("C:\\a.mp4\nC:\\b.mp4")
        self.assertEqual(oi.target, "C:\\a.mp4")

    def test_prose_is_not_input(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("hello there\nthis is just some text\nnothing to open")


class HeaderLinesTest(unittest.TestCase):
    def test_url_plus_headers(self):
        oi = O.parse(f"{URL}\nReferer: https://www.example.com/\nUser-Agent: {UA}\nOrigin: https://www.example.com\n"
                     "Cookie: a=1; b=2")
        self.assertEqual(oi.target, URL)
        self.assertEqual(oi.flavor, "headers")
        self.assertEqual(oi.headers, {"Referer": "https://www.example.com/", "User-Agent": UA,
                                      "Origin": "https://www.example.com", "Cookie": "a=1; b=2"})

    def test_blank_lines_and_spaces(self):
        oi = O.parse(f"\n  {URL}  \n\n  referer:https://r.example/  \n")
        self.assertEqual(oi.headers, {"referer": "https://r.example/"})

    def test_dropped_and_pseudo_headers(self):
        oi = O.parse(f"{URL}\n:authority: cdn.example.com\n:path: /x\nAccept-Encoding: gzip\nRange: bytes=0-\n"
                     "Host: cdn.example.com\nAccept: */*")
        self.assertEqual(oi.headers, {"Accept": "*/*"})

    def test_duplicate_header_last_wins_cookie_merges(self):
        oi = O.parse(f"{URL}\nReferer: a\nreferer: b\nCookie: x=1\ncookie: y=2")
        self.assertEqual(hdr(oi, "referer"), "b")
        self.assertEqual(hdr(oi, "cookie"), "x=1; y=2")
        self.assertEqual(sum(1 for k in oi.headers if k.lower() == "referer"), 1)

    def test_headers_before_url(self):
        oi = O.parse(f"Referer: https://r.example/\n{URL}")
        self.assertEqual((oi.target, hdr(oi, "Referer")), (URL, "https://r.example/"))

    def test_headers_without_url(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("Referer: a\nUser-Agent: b")

    def test_header_value_with_colons(self):
        oi = O.parse(f"{URL}\nReferer: https://x.example:8443/a?b=c:d")
        self.assertEqual(hdr(oi, "Referer"), "https://x.example:8443/a?b=c:d")


class CurlBashTest(unittest.TestCase):
    def test_chrome_bash(self):
        oi = O.parse(CHROME_BASH)
        self.assertEqual((oi.target, oi.flavor, oi.kind), (URL, "curl-bash", "url"))
        self.assertEqual(hdr(oi, "Cookie"), "sid=abc123; theme=dark")
        self.assertEqual(hdr(oi, "referer"), "https://www.example.com/watch/42")
        self.assertEqual(hdr(oi, "user-agent"), UA)
        self.assertEqual(hdr(oi, "sec-ch-ua"), SEC_CH_UA)
        self.assertEqual(hdr(oi, "origin"), "https://www.example.com")
        self.assertIsNone(hdr(oi, "accept-encoding"))

    def test_ansi_c_quoting(self):
        oi = O.parse(r"curl $'https://e.example/a.mp4' -H $'cookie: n=it\'s\u0021; m=1' -H $'x-tab: a\tb'")
        self.assertEqual(oi.target, "https://e.example/a.mp4")
        self.assertEqual(hdr(oi, "cookie"), "n=it's!; m=1")
        self.assertEqual(hdr(oi, "x-tab"), "a\tb")

    def test_double_quotes_and_escapes(self):
        oi = O.parse('curl "https://e.example/a.mp4" -H "Referer: https://r.example/\\"q\\"" -H X-A:\\ b')
        self.assertEqual(hdr(oi, "Referer"), 'https://r.example/"q"')
        self.assertEqual(hdr(oi, "X-A"), "b")

    def test_short_options_and_glued_values(self):
        oi = O.parse("curl -s -L -A 'UA/1' -e 'https://ref.example/;auto' -b a=1 -H'X-Y: z' "
                     "--cookie 'b=2' 'https://e.example/v.mp4' --compressed -k")
        self.assertEqual(oi.target, "https://e.example/v.mp4")
        self.assertEqual(hdr(oi, "User-Agent"), "UA/1")
        self.assertEqual(hdr(oi, "Referer"), "https://ref.example/")
        self.assertEqual(hdr(oi, "Cookie"), "a=1; b=2")
        self.assertEqual(hdr(oi, "X-Y"), "z")
        self.assertTrue(any("insecure" in n or "-k" in n for n in oi.notes))

    def test_long_option_equals_and_url_option(self):
        oi = O.parse("curl --url=https://e.example/a.m3u8 --header='Origin: https://o.example' --user-agent=UA2")
        self.assertEqual((oi.target, hdr(oi, "Origin"), hdr(oi, "User-Agent")),
                         ("https://e.example/a.m3u8", "https://o.example", "UA2"))

    def test_value_options_are_skipped(self):
        oi = O.parse("curl -X GET -o out.bin --max-time 30 -d 'a=b' https://e.example/a.mp4")
        self.assertEqual(oi.target, "https://e.example/a.mp4")

    def test_cookie_file_ignored(self):
        oi = O.parse("curl -b cookies.txt https://e.example/a.mp4")
        self.assertIsNone(hdr(oi, "Cookie"))

    def test_basic_auth(self):
        oi = O.parse("curl -u user:pw https://e.example/a.mp4")
        self.assertEqual(hdr(oi, "Authorization"), "Basic dXNlcjpwdw==")

    def test_curl_exe_and_crlf(self):
        oi = O.parse("curl.exe 'https://e.example/a.mp4' \\\r\n  -H 'Referer: r'\r\n")
        self.assertEqual((oi.target, hdr(oi, "Referer")), ("https://e.example/a.mp4", "r"))

    def test_powershell_backtick_continuation(self):
        oi = O.parse("curl 'https://e.example/a.mp4' `\n  -H 'Referer: r' `\n  --compressed")
        self.assertEqual((oi.target, hdr(oi, "Referer")), ("https://e.example/a.mp4", "r"))

    def test_no_url(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("curl -H 'Referer: r' --compressed")

    def test_unbalanced_quote(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("curl 'https://e.example/a.mp4")

    def test_curl_file_url(self):
        oi = O.parse("curl 'file:///D:/v/a.mp4'")
        self.assertEqual((oi.kind, oi.target), ("file", "D:\\v\\a.mp4"))


class CurlCmdTest(unittest.TestCase):
    def test_chrome_cmd(self):
        oi = O.parse(CHROME_CMD)
        self.assertEqual((oi.target, oi.flavor), (URL, "curl-cmd"))
        self.assertEqual(hdr(oi, "Cookie"), "sid=abc123; theme=dark")
        self.assertEqual(hdr(oi, "referer"), "https://www.example.com/watch/42")
        self.assertEqual(hdr(oi, "user-agent"), UA)
        self.assertEqual(hdr(oi, "sec-ch-ua"), SEC_CH_UA)

    def test_cmd_crlf(self):
        oi = O.parse(CHROME_CMD.replace("\n", "\r\n"))
        self.assertEqual(oi.target, URL)
        self.assertEqual(hdr(oi, "sec-ch-ua"), SEC_CH_UA)

    def test_cmd_escaped_specials(self):
        oi = O.parse('curl ^"https://e.example/a.mp4?x=1^&y=^%^7E^" -H ^"X-P: a^|b^<c^>^^d^"')
        self.assertEqual(oi.target, "https://e.example/a.mp4?x=1&y=%7E")
        self.assertEqual(hdr(oi, "X-P"), "a|b<c>^d")

    def test_cmd_literal_newline_inside_value(self):
        # DevTools writes a newline inside a value as ^ + two newlines; such a header is invalid and is dropped,
        # but it must not break the rest of the command apart
        oi = O.parse('curl ^"https://e.example/a.mp4^" ^\n  -H ^"X-M: a^\n\nb^" ^\n  -H ^"Referer: r^"')
        self.assertEqual((oi.target, hdr(oi, "X-M"), hdr(oi, "Referer")), ("https://e.example/a.mp4", None, "r"))

    def test_cmd_plain_double_quotes(self):
        # hand-written Windows command (no carets): only the CRT quoting rules apply
        oi = O.parse('curl "https://e.example/a.mp4" ^\n -H "Referer: https://r.example/" -b "k=v"')
        self.assertEqual((oi.target, hdr(oi, "Referer"), hdr(oi, "Cookie")),
                         ("https://e.example/a.mp4", "https://r.example/", "k=v"))


class PowerShellTest(unittest.TestCase):
    def test_chrome_powershell(self):
        oi = O.parse(POWERSHELL)
        self.assertEqual((oi.target, oi.flavor), (URL, "powershell"))
        self.assertEqual(hdr(oi, "User-Agent"), UA)
        self.assertEqual(hdr(oi, "Cookie"), "sid=abc123; theme=dark")
        self.assertEqual(hdr(oi, "referer"), "https://www.example.com/watch/42")
        self.assertEqual(hdr(oi, "sec-ch-ua"), '"Chromium";v="130"')
        for pseudo in ("authority", "method", "path", "scheme", "accept-encoding"):
            self.assertIsNone(hdr(oi, pseudo), pseudo)

    def test_single_quoted_uri(self):
        oi = O.parse("Invoke-WebRequest -Uri 'https://e.example/a.mp4' -Headers @{ 'Referer' = 'r' ; Origin='o' }")
        self.assertEqual((oi.target, hdr(oi, "Referer"), hdr(oi, "Origin")), ("https://e.example/a.mp4", "r", "o"))

    def test_no_uri(self):
        with self.assertRaises(O.OpenInputError):
            O.parse("Invoke-WebRequest -Headers @{}")


class ParseManyTest(unittest.TestCase):
    def test_list_of_urls_and_paths_is_one_entry_per_line(self):
        found = O.parse_many("https://a.example/1\r\n\r\n  \"C:\\videos\\a b.mp4\"\nhttps://a.example/2?x=1\nrel/clip.mp4\n")
        self.assertEqual([(o.kind, o.target) for o in found],
                         [("url", "https://a.example/1"), ("file", "C:\\videos\\a b.mp4"),
                          ("url", "https://a.example/2?x=1"), ("file", "rel/clip.mp4")])
        self.assertEqual([o.text for o in found], ["https://a.example/1", "\"C:\\videos\\a b.mp4\"",
                                                   "https://a.example/2?x=1", "rel/clip.mp4"])
        self.assertTrue(all(not o.headers for o in found))

    def test_same_target_twice_is_kept_once(self):
        self.assertEqual([o.target for o in O.parse_many("https://a.example/1\nhttps://a.example/1\n")],
                         ["https://a.example/1"])

    def test_header_lines_make_it_a_single_entry(self):
        found = O.parse_many("https://a.example/1\nReferer: https://a.example/\nCookie: a=b")
        self.assertEqual(len(found), 1)
        self.assertEqual((found[0].target, found[0].headers), ("https://a.example/1",
                                                               {"Referer": "https://a.example/", "Cookie": "a=b"}))
        self.assertEqual(found[0].text, "https://a.example/1\nReferer: https://a.example/\nCookie: a=b")
        # a Windows path on one line is not a header ("C:" looks like "Name:")
        self.assertEqual(len(O.parse_many("C:\\a.mp4\nC:\\b.mp4")), 2)

    def test_curl_and_powershell_stay_single(self):
        found = O.parse_many(CHROME_BASH)
        self.assertEqual((len(found), found[0].flavor, found[0].text), (1, "curl-bash", CHROME_BASH))
        multi = "curl 'https://a.example/1' -H 'Referer: r'\nhttps://a.example/2"
        self.assertEqual([o.target for o in O.parse_many(multi)], ["https://a.example/1"])
        ps = "Invoke-WebRequest -Uri 'https://e.example/a.mp4' -Headers @{ 'Referer' = 'r' }\nhttps://other.example/"
        self.assertEqual([o.target for o in O.parse_many(ps)], ["https://e.example/a.mp4"])

    def test_single_line_and_errors_match_parse(self):
        self.assertEqual(O.parse_many("  https://a.example/1  ")[0].target, O.parse("https://a.example/1").target)
        for bad in ("", "   \n ", None):
            with self.assertRaises(O.OpenInputError):
                O.parse_many(bad)
        with self.assertRaises(O.OpenInputError):
            O.parse_many("Referer: x\nCookie: y")

    def test_bad_line_goes_to_errors_or_raises(self):
        text = "https://a.example/1\nhttps://a.example/with space\nhttps://a.example/3"
        errors = []
        found = O.parse_many(text, errors)
        self.assertEqual([o.target for o in found], ["https://a.example/1", "https://a.example/3"])
        self.assertEqual(len(errors), 1)
        with self.assertRaises(O.OpenInputError):
            O.parse_many(text)

    def test_parse_is_unchanged(self):
        self.assertEqual(O.parse("https://a.example/1\nhttps://a.example/2").target, "https://a.example/1")
        self.assertEqual(O.parse("https://a.example/1") , O.parse_many("https://a.example/1")[0])   # text is not compared


class WindowsPathTest(unittest.TestCase):
    """Backslashes of local Windows paths must survive parse() / parse_many() untouched."""

    def targets(self, text):
        return [o.target for o in O.parse_many(text)]

    def test_plain_quoted_unc(self):
        for text, want in [(r"C:\Users\me\clip.mp4", r"C:\Users\me\clip.mp4"),
                           (r'"C:\a b\c.mp4"', r"C:\a b\c.mp4"),
                           ("'C:\\a b\\c.mp4'", r"C:\a b\c.mp4"),
                           ("\u201cC:\\a b\\c.mp4\u201d", r"C:\a b\c.mp4"),
                           ("\\\\server\\share\\x.mp4", "\\\\server\\share\\x.mp4"),
                           (r"C:\temp\new\table.mp4", r"C:\temp\new\table.mp4"),
                           (r"C:\curl\x.mp4", r"C:\curl\x.mp4")]:
            self.assertEqual(self.targets(text), [want])
            self.assertEqual(O.parse(text).target, want)
            self.assertEqual(O.parse(text).kind, "file")

    def test_multi_line(self):
        text = "C:\\a\\b.mp4\r\n\"D:\\x y\\d.mp4\"\n\\\\srv\\sh\\e.mp4\n"
        self.assertEqual(self.targets(text), ["C:\\a\\b.mp4", "D:\\x y\\d.mp4", "\\\\srv\\sh\\e.mp4"])

    def test_entry_text_reparses(self):
        for o in O.parse_many('"C:\\a b\\c.mp4"\nD:\\q\\r.mp4'):
            self.assertEqual(O.parse(o.text).target, o.target)

    def test_curl_continuations_still_work(self):
        bash = "curl 'https://a.example/v.m3u8' \\\n  -H 'Referer: https://a.example/'"
        cmd = 'curl "https://a.example/v.m3u8" ^\n  -H "Referer: https://a.example/"'
        for text in (bash, cmd):
            found = O.parse_many(text)
            self.assertEqual(len(found), 1)
            self.assertEqual(found[0].target, "https://a.example/v.m3u8")
            self.assertEqual(found[0].headers.get("Referer"), "https://a.example/")


class TokenizerTest(unittest.TestCase):
    def test_split_windows_backslashes(self):
        self.assertEqual(O.split_windows(r'a\\\"b "c d" e\f'), ['a\\"b', "c d", "e\\f"])
        self.assertEqual(O.split_windows(r'"x\\" y'), ["x\\", "y"])

    def test_split_posix(self):
        self.assertEqual(O.split_posix("a 'b c' \"d\\\"e\" f\\ g $'h\\x41'"), ["a", "b c", 'd"e', "f g", "hA"])


if __name__ == "__main__":
    unittest.main(verbosity=1)
