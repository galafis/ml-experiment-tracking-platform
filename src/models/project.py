# -*- coding: utf-8 -*-
"""
Project and team domain models.

Provides dataclasses for organising experiments into projects and
managing team-based collaboration with role-based access control.
"""

from __future__ import annotations

import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return uuid.uuid4().hex


class CollaboratorRole(str, Enum):
    """Roles a collaborator may hold within a project."""

    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


@dataclass
class Collaborator:
    """A user participating in a project.

    Attributes
    ----------
    id : str
        Unique collaborator identifier.
    user_id : str
        External user system identifier.
    username : str
        Display name.
    email : str
        Contact email address.
    role : CollaboratorRole
        Permission role within the project.
    joined_at : datetime
        When the collaborator joined.
    """

    user_id: str
    username: str
    email: str
    role: CollaboratorRole = CollaboratorRole.MEMBER
    id: str = field(default_factory=_new_id)
    joined_at: datetime = field(default_factory=_utcnow)

    @property
    def is_admin(self) -> bool:
        return self.role in (CollaboratorRole.OWNER, CollaboratorRole.ADMIN)

    @property
    def can_write(self) -> bool:
        return self.role != CollaboratorRole.VIEWER

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["role"] = self.role.value
        data["joined_at"] = self.joined_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Collaborator:
        data = dict(data)
        if isinstance(data.get("role"), str):
            data["role"] = CollaboratorRole(data["role"])
        if isinstance(data.get("joined_at"), str):
            data["joined_at"] = datetime.fromisoformat(data["joined_at"])
        return cls(**data)


@dataclass
class Team:
    """A named group of collaborators.

    Attributes
    ----------
    id : str
        Unique team identifier.
    name : str
        Team display name.
    description : str
        Team purpose or scope.
    members : list[Collaborator]
        Team members.
    created_at : datetime
        Creation timestamp.
    """

    name: str
    description: str = ""
    members: list[Collaborator] = field(default_factory=list)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)

    def add_member(self, collaborator: Collaborator) -> None:
        """Add a collaborator to the team."""
        if not any(m.user_id == collaborator.user_id for m in self.members):
            self.members.append(collaborator)

    def remove_member(self, user_id: str) -> bool:
        """Remove a member by user_id. Returns True if removed."""
        before = len(self.members)
        self.members = [m for m in self.members if m.user_id != user_id]
        return len(self.members) < before

    def get_member(self, user_id: str) -> Optional[Collaborator]:
        for m in self.members:
            if m.user_id == user_id:
                return m
        return None

    @property
    def member_count(self) -> int:
        return len(self.members)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "members": [m.to_dict() for m in self.members],
            "created_at": self.created_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Team:
        data = dict(data)
        data["members"] = [
            Collaborator.from_dict(m) for m in data.get("members", [])
        ]
        if isinstance(data.get("created_at"), str):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        return cls(**data)


@dataclass
class Project:
    """Top-level organisational unit for experiments.

    Attributes
    ----------
    id : str
        Unique project identifier.
    name : str
        Project display name.
    description : str
        Project purpose.
    team : Team | None
        Assigned team.
    experiment_ids : list[str]
        IDs of experiments belonging to this project.
    tags : dict[str, str]
        Arbitrary metadata tags.
    created_at : datetime
        Creation timestamp.
    updated_at : datetime
        Last modification timestamp.
    """

    name: str
    description: str = ""
    team: Optional[Team] = None
    experiment_ids: list[str] = field(default_factory=list)
    tags: dict[str, str] = field(default_factory=dict)
    id: str = field(default_factory=_new_id)
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def add_experiment(self, experiment_id: str) -> None:
        """Register an experiment with this project."""
        if experiment_id not in self.experiment_ids:
            self.experiment_ids.append(experiment_id)
            self.updated_at = _utcnow()

    def remove_experiment(self, experiment_id: str) -> bool:
        """Remove an experiment reference. Returns True if removed."""
        if experiment_id in self.experiment_ids:
            self.experiment_ids.remove(experiment_id)
            self.updated_at = _utcnow()
            return True
        return False

    @property
    def total_experiments(self) -> int:
        return len(self.experiment_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "team": self.team.to_dict() if self.team else None,
            "experiment_ids": list(self.experiment_ids),
            "tags": dict(self.tags),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Project:
        data = dict(data)
        team_data = data.pop("team", None)
        if team_data:
            data["team"] = Team.from_dict(team_data)
        for dt_field in ("created_at", "updated_at"):
            val = data.get(dt_field)
            if isinstance(val, str):
                data[dt_field] = datetime.fromisoformat(val)
        return cls(**data)
