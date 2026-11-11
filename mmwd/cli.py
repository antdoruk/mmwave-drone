"""Shared command-line plumbing: every script can run on the real board,
on the simulator, or on a recorded .bin file with the same flags."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import parse_cfg
from .radar import BoardSource, ReplaySource, SimSource, find_cli_and_data, send_cfg


def add_source_args(ap: argparse.ArgumentParser):
    ap.add_argument("--cfg", default="configs/md_1tx.cfg", help="chirp configuration file")
    ap.add_argument("--sim", metavar="SCENE", help="no board: simulate empty | drone | human | fan")
    ap.add_argument("--distance", type=float, default=2.0, help="target distance for --sim (m)")
    ap.add_argument("--replay", metavar="BIN", help="play back a raw .bin recording")
    ap.add_argument("--cli", help="CLI serial port (default: auto-detect)")
    ap.add_argument("--data", help="data serial port (default: auto-detect)")
    ap.add_argument("--no-config", action="store_true",
                    help="board is already configured and running, do not send the cfg again")


def open_source(args, raw_path=None):
    params = parse_cfg(args.cfg)
    if args.sim:
        return SimSource(params, args.sim, args.distance, raw_path=raw_path), params, None
    if args.replay:
        return ReplaySource(args.replay, fps=params.fps), params, None
    try:
        cli, data = (args.cli, args.data) if (args.cli and args.data) else find_cli_and_data()
    except RuntimeError as e:
        print(f"\nBOARD NOT FOUND: {e}")
        print("Reconnect the USB cable, allow the accessory if macOS asks, "
              "then check that `ls /dev/cu.*` lists two usbserial ports.")
        sys.exit(1)
    print(f"CLI port : {cli}\ndata port: {data}")
    if not args.no_config:
        errs = send_cfg(cli, args.cfg)
        if errs:
            print("\nThe board rejected some lines:")
            for e in errs:
                print("  " + e)
            print("Check the configuration file. Stopping.")
            sys.exit(2)
    return BoardSource(data, raw_path=raw_path), params, cli


def cfg_text(path) -> str:
    return Path(path).read_text()
