from __future__ import annotations

import fire

from .cli import MovementCLI


def main() -> None:
    fire.Fire(MovementCLI)


if __name__ == "__main__":
    main()
