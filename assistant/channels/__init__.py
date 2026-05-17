"""Chat channels module with plugin architecture."""

from assistant.channels.base import BaseChannel
from assistant.channels.manager import ChannelManager

__all__ = ["BaseChannel", "ChannelManager"]
