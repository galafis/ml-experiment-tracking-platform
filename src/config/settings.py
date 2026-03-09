# -*- coding: utf-8 -*-
"""
Platform configuration using Pydantic Settings.

Supports environment variables, .env files, and YAML configuration.
All secrets can be injected via environment variables at deployment time.
"""

from functools import lru_cache
from pathlib import Path
from typing import Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatabaseConfig(BaseSettings):
    """PostgreSQL connection configuration."""

    model_config = SettingsConfigDict(env_prefix="POSTGRES_")

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5432, description="PostgreSQL port")
    user: str = Field(default="mltrack", description="Database user")
    password: str = Field(default="mltrack_secret", description="Database password")
    database: str = Field(default="ml_experiments", description="Database name")
    pool_size: int = Field(default=10, description="Connection pool size")
    max_overflow: int = Field(default=20, description="Max overflow connections")
    echo_sql: bool = Field(default=False, description="Echo SQL statements")

    @property
    def dsn(self) -> str:
        """Build PostgreSQL DSN string."""
        return (
            f"postgresql+asyncpg://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )

    @property
    def sync_dsn(self) -> str:
        """Build synchronous PostgreSQL DSN string."""
        return (
            f"postgresql://{self.user}:{self.password}"
            f"@{self.host}:{self.port}/{self.database}"
        )


class MongoConfig(BaseSettings):
    """MongoDB connection configuration."""

    model_config = SettingsConfigDict(env_prefix="MONGO_")

    host: str = Field(default="localhost", description="MongoDB host")
    port: int = Field(default=27017, description="MongoDB port")
    user: Optional[str] = Field(default=None, description="MongoDB user")
    password: Optional[str] = Field(default=None, description="MongoDB password")
    database: str = Field(default="ml_artifacts", description="Database name")
    auth_source: str = Field(default="admin", description="Auth database")
    max_pool_size: int = Field(default=50, description="Max connection pool size")

    @property
    def uri(self) -> str:
        """Build MongoDB connection URI."""
        if self.user and self.password:
            return (
                f"mongodb://{self.user}:{self.password}"
                f"@{self.host}:{self.port}/{self.database}"
                f"?authSource={self.auth_source}"
            )
        return f"mongodb://{self.host}:{self.port}/{self.database}"


class RedisConfig(BaseSettings):
    """Redis connection configuration."""

    model_config = SettingsConfigDict(env_prefix="REDIS_")

    host: str = Field(default="localhost", description="Redis host")
    port: int = Field(default=6379, description="Redis port")
    password: Optional[str] = Field(default=None, description="Redis password")
    db: int = Field(default=0, description="Redis database index")
    default_ttl: int = Field(default=3600, description="Default TTL in seconds")
    max_connections: int = Field(default=20, description="Max connections")

    @property
    def url(self) -> str:
        """Build Redis URL."""
        auth = f":{self.password}@" if self.password else ""
        return f"redis://{auth}{self.host}:{self.port}/{self.db}"


class APIConfig(BaseSettings):
    """FastAPI application configuration."""

    model_config = SettingsConfigDict(env_prefix="API_")

    title: str = Field(
        default="ML Experiment Tracking Platform",
        description="API title",
    )
    description: str = Field(
        default="Track, compare, and manage ML experiments with full data versioning",
        description="API description",
    )
    version: str = Field(default="1.0.0", description="API version")
    host: str = Field(default="0.0.0.0", description="Server bind host")
    port: int = Field(default=8000, description="Server bind port")
    debug: bool = Field(default=False, description="Debug mode")
    cors_origins: list[str] = Field(
        default=["*"], description="Allowed CORS origins"
    )
    api_prefix: str = Field(default="/api/v1", description="API route prefix")
    workers: int = Field(default=4, description="Uvicorn worker count")

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v


class PlatformSettings(BaseSettings):
    """Root settings aggregating all platform configuration."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    environment: str = Field(default="development", description="Runtime environment")
    log_level: str = Field(default="INFO", description="Global log level")
    log_json: bool = Field(default=False, description="Use JSON log format")
    data_dir: Path = Field(
        default=Path("./data"), description="Local data directory"
    )

    postgres: DatabaseConfig = Field(default_factory=DatabaseConfig)
    mongo: MongoConfig = Field(default_factory=MongoConfig)
    redis: RedisConfig = Field(default_factory=RedisConfig)
    api: APIConfig = Field(default_factory=APIConfig)

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"

    @property
    def is_testing(self) -> bool:
        return self.environment.lower() == "testing"


@lru_cache(maxsize=1)
def get_settings() -> PlatformSettings:
    """Return cached singleton platform settings."""
    return PlatformSettings()
