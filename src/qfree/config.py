"""Runtime configuration loaded from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    tushare_token: str | None
    data_dir: Path = Path("data")
    min_amount: float = 100_000.0

    @classmethod
    def from_env(cls) -> Settings:
        minimum = os.getenv("QFREE_MIN_AMOUNT", "100000")
        token = os.getenv("TUSHARE_TOKEN") or None
        token_file = os.getenv("QFREE_TOKEN_FILE", "token")
        if token is None:
            path = Path(token_file)
            if path.is_file():
                token = path.read_text(encoding="utf-8").strip() or None
        return cls(
            tushare_token=token,
            data_dir=Path(os.getenv("QFREE_DATA_DIR", "data")),
            min_amount=float(minimum),
        )

    def require_token(self) -> str:
        if not self.tushare_token:
            raise RuntimeError("TUSHARE_TOKEN is required for network data commands")
        return self.tushare_token
