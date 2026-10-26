"""Browser selection and launch shared by the bundled page resolvers."""
import sys


def browser_options(platform=None):
    platform = platform or sys.platform
    if platform == "darwin":
        return (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            ["chrome", "msedge", None],
        )
    return (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        ["msedge", "chrome", None],
    )


def launch_browser(playwright, *, channels=None, profile_dir=None, user_agent=None, args=None):
    """Try system browsers before bundled Chromium; optionally keep a profile."""
    channels = browser_options()[1] if channels is None else channels
    options = {"headless": False, "args": args if args is not None else ["--window-position=-2000,-2000"]}
    if user_agent:
        options["user_agent"] = user_agent
    last_error = None
    for channel in channels:
        try:
            kwargs = dict(options)
            if channel:
                kwargs["channel"] = channel
            if profile_dir is not None:
                return playwright.chromium.launch_persistent_context(user_data_dir=profile_dir, **kwargs)
            return playwright.chromium.launch(**kwargs)
        except Exception as exc:
            last_error = exc
    names = ", ".join(channel or "Chromium" for channel in channels)
    raise RuntimeError(f"Could not launch browser (tried {names}): {last_error}")
