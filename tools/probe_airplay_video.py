"""Opt-in real-device MP4 probe. Uses existing credentials without modifying them."""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from oh_streamer.platform.airplay_cast import CredentialStore, failure_details


async def main(args):
    import pyatv
    from pyatv.const import Protocol
    from pyatv.settings import AirPlayVersion, MrpTunnel
    from pyatv.storage.memory_storage import MemoryStorage
    from pyatv.support.http import HttpConnection
    from pyatv.protocols.airplay.utils import get_protocol_version

    # Raw pyatv debug logging may contain credentials or media URLs.
    logging.getLogger("pyatv").setLevel(logging.CRITICAL)
    original = HttpConnection.send_and_receive

    async def traced(self, method, uri, *pos, **kw):
        path = uri.split("?", 1)[0]
        safe = path if path in {"/play", "/playback-info", "/pair-verify", "/info",
                               "/feedback", "/rate", "/setProperty"} else "<redacted>"
        try:
            response = await original(self, method, uri, *pos, **kw)
            print(f"{method} {safe}: {response.code}", flush=True)
            return response
        except Exception as error:
            print(f"{method} {safe}: {type(error).__name__}; {failure_details(error)}", flush=True)
            if args.headers and uri == "/playback-info":
                from pyatv.protocols.raop.protocols.airplayv2 import HEADERS
                for label, options in (
                    ("AirPlay session headers", {"headers": HEADERS}),
                    ("RTSP plus session headers", {"headers": HEADERS, "protocol": "RTSP/1.0"}),
                ):
                    try:
                        response = await original(self, method, uri, **options)
                        print(f"{label}: {response.code}", flush=True)
                        return response
                    except Exception as retry_error:
                        print(f"{label}: {failure_details(retry_error)}", flush=True)
            raise

    entry = CredentialStore(args.credentials).get(args.device)
    if not entry:
        raise RuntimeError("No saved credentials for selected device")
    HttpConnection.send_and_receive = traced
    try:
        for preferred in ((AirPlayVersion.V2,) if args.headers else (AirPlayVersion.V1, AirPlayVersion.V2)):
            print(f"--- AirPlay {preferred.value}, local test MP4 ---", flush=True)
            atv = None
            try:
                configs = await pyatv.scan(asyncio.get_running_loop(), hosts=[entry["address"]], timeout=4)
                config = next(c for c in configs if str(c.identifier) == args.device)
                config.set_credentials(Protocol.AirPlay, entry["airplay"])
                service = config.get_service(Protocol.AirPlay)
                print(f"automatic protocol: {get_protocol_version(service, AirPlayVersion.Auto).name}", flush=True)
                for svc in config.services:
                    if svc.protocol != Protocol.AirPlay:
                        svc.enabled = False
                storage = MemoryStorage()
                settings = await storage.get_settings(config)
                settings.protocols.airplay.mrp_tunnel = MrpTunnel.Disable
                settings.protocols.raop.protocol_version = preferred
                atv = await pyatv.connect(config, asyncio.get_running_loop(), storage=storage)
                await asyncio.wait_for(atv.stream.play_url(str(Path(args.video).resolve())), 20)
                print("play_url returned normally (check TV for visible playback)", flush=True)
            except Exception as error:
                print(f"RESULT: {type(error).__name__}; {failure_details(error)}", flush=True)
            finally:
                if atv:
                    pending = atv.close()
                    if pending:
                        await asyncio.wait(pending, timeout=3)
            await asyncio.sleep(2)
    finally:
        HttpConnection.send_and_receive = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", required=True)
    parser.add_argument("--credentials", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--headers", action="store_true", help="Test playback-info session headers on AirPlay 2")
    asyncio.run(main(parser.parse_args()))
