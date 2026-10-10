"""The provider seam: detection is explicit, construction goes through it, and
a host with nothing supported gets a sentence rather than an ImportError."""

from __future__ import annotations

import sys

import pytest

from grip.agent import provider
from grip.agent.collect import NvmlProvider
from grip.agent.provider import (
    AcceleratorProvider,
    detect_vendor,
    provider_class,
    provider_for_host,
)


def test_detect_nvidia_by_cli_or_sdk(monkeypatch):
    monkeypatch.setattr(provider.importlib.util, "find_spec", lambda name: None)
    assert (
        detect_vendor(which=lambda n: "/usr/bin/nvidia-smi" if n == "nvidia-smi" else None)
        == "nvidia"
    )
    assert detect_vendor(which=lambda n: None) is None
    monkeypatch.setattr(provider.importlib.util, "find_spec", lambda name: object())
    assert detect_vendor(which=lambda n: None) == "nvidia"


def test_provider_class_maps_known_vendors_and_rejects_the_rest():
    from grip.agent.amd import AmdSmiProvider

    assert provider_class("nvidia") is NvmlProvider
    assert provider_class("amd") is AmdSmiProvider  # experimental, but a real branch now
    with pytest.raises(RuntimeError, match="no accelerator provider"):
        provider_class("intel")


def test_no_supported_stack_is_a_plain_sentence(monkeypatch):
    monkeypatch.setattr(provider, "detect_vendor", lambda which=None: None)
    with pytest.raises(RuntimeError, match="no supported accelerator SDK"):
        provider_for_host()


def test_nvml_provider_satisfies_the_protocol(monkeypatch):
    from test_agent_collect import FakeNvml

    monkeypatch.setitem(sys.modules, "pynvml", FakeNvml(n=1))
    p = NvmlProvider()
    assert isinstance(p, AcceleratorProvider)
    assert p.vendor == "nvidia" and p.count == 1
