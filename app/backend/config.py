"""Application configuration using pydantic-settings.

Configuration is loaded from environment variables and .env files.
All settings have sensible defaults for local development.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database - using port 5433 to avoid conflicts with other PostgreSQL instances
    database_url: str = "postgresql+asyncpg://lancer:lancer@localhost:5433/lancer"

    # Server
    debug: bool = True
    
    # CORS origins as comma-separated string (parsed to list via property)
    cors_origins_raw: str = Field(
        default="http://localhost:5173,http://localhost:3000,http://127.0.0.1:5173,http://127.0.0.1:3000",
        validation_alias="cors_origins",
    )

    # API
    api_prefix: str = "/api"

    @property
    def cors_origins(self) -> list[str]:
        """Parse CORS origins from comma-separated string."""
        return [origin.strip() for origin in self.cors_origins_raw.split(",") if origin.strip()]


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience export
settings = get_settings()
