# -*- coding: utf-8 -*-
"""Whether the model hub can be reached — decided once, before anything tries.

every model here except whisper's own comes from hugging face. once the weights are in the
local cache nothing needs the network, but the libraries do not know that: huggingface_hub asks
the hub for the newest revision on every load and falls back to the cache only after its
retries run out. measured in a container with no network: five minutes of a six-minute run went
to those retries (70 s for the voice model, 230 s for the emotion scales), on a two-minute clip
that otherwise takes 38 s.

the libraries read HF_HUB_OFFLINE once, when they are imported, and transformers copies it into
its own module at that moment. so this has to run before any of them is imported, which is the
first thing the command line does.
"""
import os
import socket

HUB_HOST = "huggingface.co"
HUB_PORT = 443
PROBE_TIMEOUT_SEC = 2.0


def hub_reachable(host=HUB_HOST, port=HUB_PORT, timeout=PROBE_TIMEOUT_SEC,
                  connect=socket.create_connection) -> bool:
    """One tcp connection, nothing sent over it. dns failure, refusal and timeout all count
    as unreachable — they are all OSError."""
    try:
        connect((host, port), timeout=timeout).close()
        return True
    except OSError:
        return False


def prefer_local_models(environ=None, reachable=hub_reachable):
    """Switch the hub libraries to the local cache when the hub cannot be reached.

    returns a line worth logging, or None when nothing changed. an explicit HF_HUB_OFFLINE or
    TRANSFORMERS_OFFLINE, in either direction, is the user's decision: left alone, not probed.
    """
    environ = os.environ if environ is None else environ
    if "HF_HUB_OFFLINE" in environ or "TRANSFORMERS_OFFLINE" in environ:
        return None
    if reachable():
        return None
    environ["HF_HUB_OFFLINE"] = "1"
    return (f"{HUB_HOST} is unreachable — models are read from the local cache only "
            f"(HF_HUB_OFFLINE=1). a model that was never downloaded will fail to load.")
