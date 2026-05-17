"""Message bus module for decoupled channel-agent communication."""

from assistant.bus.events import InboundMessage, OutboundMessage
from assistant.bus.queue import MessageBus

__all__ = ["MessageBus", "InboundMessage", "OutboundMessage"]
