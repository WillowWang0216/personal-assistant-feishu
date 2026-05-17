"""
Entry point for running personal-assistant as a module: python -m assistant
"""

from assistant.cli.commands import app

if __name__ == "__main__":
    app()
