"""Talk to a local LM Studio server through its OpenAI-compatible API (free, nothing leaves the machine)."""
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import requests

import store

THINK_RE = re.compile(r"<think>.*?</think>", re.S)
CHUNK_TIMEOUT = 1200  # replies stream in, so this only limits the wait between tokens; reading a long
# prompt on a CPU (or waiting behind another request) can take many minutes before the first one
CHECK_TIMEOUT = 3
DEFAULT_CONTEXT = 4096  # LM Studio's usual default when the server doesn't say
CHARS_PER_TOKEN = 3  # rough and on the safe side for English with numbers


class LLMError(RuntimeError):
    pass


class Stopped(LLMError):
    pass


def _base_url():
    return store.get_settings()["lmstudio_url"].rstrip("/")


def list_models(base_url=None):
    """Chat models the server has available (embedding models are left out)."""
    try:
        res = requests.get(f"{(base_url or _base_url()).rstrip('/')}/models", timeout=CHECK_TIMEOUT)
        res.raise_for_status()
    except requests.RequestException as err:
        raise LLMError(f"Couldn't reach LM Studio at {base_url or _base_url()}. Is its server running?") from err
    return [m["id"] for m in res.json().get("data", []) if "embed" not in m["id"].lower()]


def status():
    settings = store.get_settings()
    try:
        models = list_models(settings["lmstudio_url"])
    except LLMError as err:
        return {"online": False, "message": str(err), "models": [], "model": settings["lmstudio_model"]}
    model = settings["lmstudio_model"] if settings["lmstudio_model"] in models else (models[0] if models else "")
    return {"online": True, "models": models, "model": model,
            "message": f"Connected · {len(models)} chat model{'s' if len(models) != 1 else ''}" if models
            else "Connected, but no chat model is downloaded. Load one in LM Studio."}


def context_length(model):
    """Tokens the loaded model can hold (prompt + reply), from LM Studio's own REST API."""
    root = re.sub(r"/v1/?$", "", _base_url())
    try:
        res = requests.get(f"{root}/api/v0/models/{model}", timeout=CHECK_TIMEOUT)
        res.raise_for_status()
        info = res.json()
    except (requests.RequestException, ValueError):
        return DEFAULT_CONTEXT
    return int(info.get("loaded_context_length") or info.get("max_context_length") or DEFAULT_CONTEXT)


def estimate_tokens(text):
    return len(text) // CHARS_PER_TOKEN + 1


def _error_text(payload):
    """LM Studio nests the engine's error JSON inside its own message; pull out the readable part."""
    message = str((payload.get("error") or {}).get("message") or payload.get("message") or payload)
    inner = re.search(r'\\?"message\\?"\s*:\s*\\?"(.*?)\\?"', message)
    message = inner.group(1) if inner else message
    if "context size" in message or "context length" in message:
        message += (". Raise the model's context length in LM Studio (model load settings), "
                    "or analyze a smaller portfolio.")
    return message


def active_model():
    s = status()
    if not s["online"]:
        raise LLMError(s["message"])
    if not s["model"]:
        raise LLMError(s["message"])
    return s["model"]


def chat(messages, model=None, temperature=0.3, max_tokens=-1, on_progress=None, should_stop=None):
    """Send a chat and return the reply text, with any reasoning stripped out.

    The reply is streamed so a slow local model never hits a request timeout. on_progress(phase, tokens) is
    called as tokens arrive (phase "thinking" for a reasoning model's reasoning, then "writing"), and
    should_stop() is checked between tokens to cancel.
    """
    model = model or active_model()
    try:
        res = requests.post(f"{_base_url()}/chat/completions", timeout=(CHECK_TIMEOUT, CHUNK_TIMEOUT), stream=True, json={
            "model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens, "stream": True,
        })
    except requests.RequestException as err:
        raise LLMError(f"LM Studio request failed: {err}") from err
    if not res.ok:
        raise LLMError(f"LM Studio returned {res.status_code}: {res.text[:300]}")
    parts, counts, in_think = [], {"thinking": 0, "writing": 0}, False
    try:
        for line in res.iter_lines():
            if should_stop and should_stop():
                raise Stopped("Stopped.")
            if not line.startswith(b"data: ") or line == b"data: [DONE]":
                continue
            chunk = json.loads(line[6:])
            if "error" in chunk or not chunk.get("choices"):
                if "error" in chunk:
                    raise LLMError(f"LM Studio: {_error_text(chunk)}")
                continue
            delta = chunk["choices"][0].get("delta", {})
            # LM Studio sends a reasoning model's thoughts separately; others inline them in <think> tags
            content = delta.get("content") or ""
            if content:
                parts.append(content)
                in_think = ("<think>" in content or in_think) and "</think>" not in content
            if not content and not delta.get("reasoning_content"):
                continue
            phase = "thinking" if delta.get("reasoning_content") or in_think else "writing"
            counts[phase] += 1
            if on_progress:
                on_progress(phase, counts[phase])
    except requests.RequestException as err:
        raise LLMError(f"LM Studio stopped responding: {err}") from err
    finally:
        res.close()
    text = THINK_RE.sub("", "".join(parts))
    if "</think>" in text:  # some reasoning models omit the opening tag
        text = text.split("</think>", 1)[1]
    if not text.strip():
        raise LLMError("The model finished without writing an answer, usually because it ran out of context while thinking. "
                       "Raise its context length in LM Studio, or pick a different model in Connections.")
    return text.strip()


def launch():
    """Start LM Studio with the command from Connections (e.g. ~/bin/lmstudio), detached from the app."""
    command = store.get_settings()["lmstudio_command"]
    args = shlex.split(command)
    if not args:
        raise LLMError("Set the command that starts LM Studio first.")
    args[0] = os.path.expanduser(args[0])
    if not Path(args[0]).exists() and "/" in args[0]:
        raise LLMError(f"{args[0]} doesn't exist.")
    try:
        subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL,
                         start_new_session=True)
    except OSError as err:
        raise LLMError(f"Couldn't start LM Studio: {err}") from err
