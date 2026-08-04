"""Service module for Role-based Model Routing and Multi-Provider Configurations."""

from __future__ import annotations

import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

from sqlalchemy.orm import Session
from api.database import UserLLMConfigDB, ProviderDB, ModelProfileDB, RoleBindingDB
from api.services.auth_service import decrypt_secret, encrypt_secret

logger = logging.getLogger(__name__)

# Role Groups Definition
ROLE_GROUPS = {
    "analysts": ["market", "social", "news", "fundamentals", "macro", "smart_money", "volume_price"],
    "researchers": ["bull_researcher", "bear_researcher"],
    "arbiter": ["research_manager"],
    "trader": ["trader"],
    "risk": ["aggressive_analyst", "neutral_analyst", "conservative_analyst", "risk_manager"],
}

# Reverse Mapping: Role -> Group
ROLE_TO_GROUP = {role: group for group, roles in ROLE_GROUPS.items() for role in roles}

# Role Default Tier Mapping
ROLE_DEFAULT_TIERS = {
    "market": "quick",
    "social": "quick",
    "news": "quick",
    "fundamentals": "quick",
    "macro": "quick",
    "smart_money": "quick",
    "volume_price": "quick",
    "bull_researcher": "quick",
    "bear_researcher": "quick",
    "research_manager": "deep",
    "trader": "quick",
    "aggressive_analyst": "quick",
    "neutral_analyst": "quick",
    "conservative_analyst": "quick",
    "risk_manager": "deep",
}

ALL_ROLES = list(ROLE_DEFAULT_TIERS.keys())


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _mask_api_key(api_key: Optional[str]) -> Optional[str]:
    if not api_key:
        return None
    if len(api_key) <= 8:
        return "****"
    return f"{api_key[:4]}****{api_key[-4:]}"


def migrate_legacy_user_llm_config(db: Session, user_id: str) -> None:
    """Migrate single-provider legacy UserLLMConfigDB record to ProviderDB and ModelProfileDB."""
    existing_providers = db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()
    if existing_providers:
        return  # Already migrated or user has new multi-provider configs

    legacy_cfg = db.query(UserLLMConfigDB).filter(UserLLMConfigDB.user_id == user_id).first()
    if not legacy_cfg:
        return  # No config to migrate

    provider_type = legacy_cfg.llm_provider or "openai"
    base_url = legacy_cfg.backend_url
    api_key_encrypted = legacy_cfg.api_key_encrypted
    quick_llm = legacy_cfg.quick_think_llm or "gpt-4o-mini"
    deep_llm = legacy_cfg.deep_think_llm or "gpt-4o"

    now = _utcnow()

    # Create Provider
    provider_id = uuid4().hex
    provider = ProviderDB(
        id=provider_id,
        user_id=user_id,
        provider_type=provider_type,
        base_url=base_url,
        api_key_encrypted=api_key_encrypted,
        display_name=f"默认厂商 ({provider_type})",
        enabled=True,
        created_at=now,
        updated_at=now,
    )
    db.add(provider)

    # Create Quick ModelProfile
    quick_profile_id = uuid4().hex
    quick_profile = ModelProfileDB(
        id=quick_profile_id,
        user_id=user_id,
        provider_id=provider_id,
        model_name=quick_llm,
        display_name=f"常规模型 ({quick_llm})",
        tier="quick",
        is_default=True,
        created_at=now,
        updated_at=now,
    )
    db.add(quick_profile)

    # Create Deep ModelProfile
    deep_profile_id = uuid4().hex
    deep_profile = ModelProfileDB(
        id=deep_profile_id,
        user_id=user_id,
        provider_id=provider_id,
        model_name=deep_llm,
        display_name=f"推理模型 ({deep_llm})",
        tier="deep",
        is_default=False,
        created_at=now,
        updated_at=now,
    )
    db.add(deep_profile)

    try:
        db.commit()
        logger.info(f"Successfully migrated legacy LLM config for user {user_id}")
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to migrate legacy LLM config for user {user_id}: {e}")


def resolve_role_model_config(
    db: Optional[Session],
    user_id: Optional[str],
    role_key: str,
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve model configuration for a specific agent role following the Fallback Chain:

    1. Explicit RoleBinding for role_key
    2. Group RoleBinding for role's group
    3. Tier default ModelProfile (quick/deep)
    4. Global default ModelProfile (is_default=True or first)
    5. System fallback (runtime_config / env vars)
    """
    runtime_config = runtime_config or {}
    default_tier = ROLE_DEFAULT_TIERS.get(role_key, "quick")

    fallback_info = {
        "role_key": role_key,
        "resolved_via": "system_fallback",
        "fallback_used": False,
        "provider_type": runtime_config.get("llm_provider") or "openai",
        "model_name": runtime_config.get(f"{default_tier}_think_llm") or runtime_config.get("quick_think_llm") or "gpt-4o-mini",
        "base_url": runtime_config.get("backend_url"),
        "api_key": runtime_config.get("api_key"),
        "temperature": None,
        "max_tokens": None,
        "profile_id": None,
        "provider_id": None,
        "profile_display_name": None,
        "provider_display_name": None,
    }

    if not db or not user_id:
        return fallback_info

    # 1. Run migration if needed
    migrate_legacy_user_llm_config(db, user_id)

    # Helper to resolve Profile & Provider DB objects
    def _build_resolution(profile: ModelProfileDB, provider: ProviderDB, via: str) -> Dict[str, Any]:
        api_key = decrypt_secret(provider.api_key_encrypted) if provider and provider.api_key_encrypted else runtime_config.get("api_key")
        base_url = provider.base_url if provider and provider.base_url else runtime_config.get("backend_url")
        provider_type = provider.provider_type if provider and provider.provider_type else (runtime_config.get("llm_provider") or "openai")
        
        is_fallback = (via != "role_binding")

        return {
            "role_key": role_key,
            "resolved_via": via,
            "fallback_used": is_fallback,
            "provider_type": provider_type,
            "model_name": profile.model_name,
            "base_url": base_url,
            "api_key": api_key,
            "temperature": profile.temperature,
            "max_tokens": profile.max_tokens,
            "profile_id": profile.id,
            "provider_id": provider.id,
            "profile_display_name": profile.display_name,
            "provider_display_name": provider.display_name if provider else provider_type,
        }

    profiles_by_id = {p.id: p for p in db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).all()}
    providers_by_id = {pr.id: pr for pr in db.query(ProviderDB).filter(ProviderDB.user_id == user_id, ProviderDB.enabled == True).all()}

    # Step 1: Check Explicit Role Binding
    role_binding = db.query(RoleBindingDB).filter(
        RoleBindingDB.user_id == user_id,
        RoleBindingDB.target_type == "role",
        RoleBindingDB.target_key == role_key,
    ).first()

    if role_binding and role_binding.model_profile_id in profiles_by_id:
        profile = profiles_by_id[role_binding.model_profile_id]
        if profile.provider_id in providers_by_id:
            return _build_resolution(profile, providers_by_id[profile.provider_id], "role_binding")

    # Step 2: Check Group Role Binding
    group_key = ROLE_TO_GROUP.get(role_key)
    if group_key:
        group_binding = db.query(RoleBindingDB).filter(
            RoleBindingDB.user_id == user_id,
            RoleBindingDB.target_type == "group",
            RoleBindingDB.target_key == group_key,
        ).first()

        if group_binding and group_binding.model_profile_id in profiles_by_id:
            profile = profiles_by_id[group_binding.model_profile_id]
            if profile.provider_id in providers_by_id:
                return _build_resolution(profile, providers_by_id[profile.provider_id], "group_binding")

    # Step 3: Tier Default Profile (quick / deep)
    tier_profile = db.query(ModelProfileDB).filter(
        ModelProfileDB.user_id == user_id,
        ModelProfileDB.tier == default_tier,
    ).first()

    if tier_profile and tier_profile.provider_id in providers_by_id:
        return _build_resolution(tier_profile, providers_by_id[tier_profile.provider_id], "tier_default")

    # Step 4: Global Default Profile
    default_profile = db.query(ModelProfileDB).filter(
        ModelProfileDB.user_id == user_id,
        ModelProfileDB.is_default == True,
    ).first() or db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).first()

    if default_profile and default_profile.provider_id in providers_by_id:
        return _build_resolution(default_profile, providers_by_id[default_profile.provider_id], "global_default")

    # Step 5: System Fallback
    fallback_info["fallback_used"] = True
    logger.warning(f"Role '{role_key}' for user '{user_id}' failed to resolve custom ModelProfile, falling back to system default.")
    return fallback_info


def resolve_all_roles(
    db: Optional[Session],
    user_id: Optional[str],
    runtime_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Dict[str, Any]]:
    """Resolve model configuration for all 15 agent roles."""
    return {role: resolve_role_model_config(db, user_id, role, runtime_config) for role in ALL_ROLES}


# --- Provider CRUD Helpers ---

def list_providers(db: Session, user_id: str) -> List[Dict[str, Any]]:
    migrate_legacy_user_llm_config(db, user_id)
    providers = db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()
    res = []
    for p in providers:
        raw_key = decrypt_secret(p.api_key_encrypted) if p.api_key_encrypted else None
        res.append({
            "id": p.id,
            "user_id": p.user_id,
            "provider_type": p.provider_type,
            "display_name": p.display_name,
            "base_url": p.base_url,
            "has_api_key": bool(raw_key),
            "api_key_masked": _mask_api_key(raw_key),
            "enabled": p.enabled,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
        })
    return res


def create_provider(
    db: Session,
    user_id: str,
    provider_type: str,
    display_name: str,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enabled: bool = True,
) -> Dict[str, Any]:
    now = _utcnow()
    provider_id = uuid4().hex
    encrypted_key = encrypt_secret(api_key) if api_key else None

    provider = ProviderDB(
        id=provider_id,
        user_id=user_id,
        provider_type=provider_type,
        display_name=display_name,
        base_url=base_url,
        api_key_encrypted=encrypted_key,
        enabled=enabled,
        created_at=now,
        updated_at=now,
    )
    db.add(provider)
    db.commit()
    db.refresh(provider)

    return {
        "id": provider.id,
        "user_id": provider.user_id,
        "provider_type": provider.provider_type,
        "display_name": provider.display_name,
        "base_url": provider.base_url,
        "has_api_key": bool(api_key),
        "api_key_masked": _mask_api_key(api_key),
        "enabled": provider.enabled,
        "created_at": provider.created_at,
        "updated_at": provider.updated_at,
    }


def update_provider(
    db: Session,
    user_id: str,
    provider_id: str,
    display_name: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    enabled: Optional[bool] = None,
    clear_api_key: bool = False,
) -> Optional[Dict[str, Any]]:
    provider = db.query(ProviderDB).filter(ProviderDB.id == provider_id, ProviderDB.user_id == user_id).first()
    if not provider:
        return None

    if display_name is not None:
        provider.display_name = display_name
    if base_url is not None:
        provider.base_url = base_url
    if enabled is not None:
        provider.enabled = enabled
    if clear_api_key:
        provider.api_key_encrypted = None
    elif api_key is not None:
        provider.api_key_encrypted = encrypt_secret(api_key)

    provider.updated_at = _utcnow()
    db.commit()
    db.refresh(provider)

    raw_key = decrypt_secret(provider.api_key_encrypted) if provider.api_key_encrypted else None
    return {
        "id": provider.id,
        "user_id": provider.user_id,
        "provider_type": provider.provider_type,
        "display_name": provider.display_name,
        "base_url": provider.base_url,
        "has_api_key": bool(raw_key),
        "api_key_masked": _mask_api_key(raw_key),
        "enabled": provider.enabled,
        "created_at": provider.created_at,
        "updated_at": provider.updated_at,
    }


def delete_provider(db: Session, user_id: str, provider_id: str) -> bool:
    provider = db.query(ProviderDB).filter(ProviderDB.id == provider_id, ProviderDB.user_id == user_id).first()
    if not provider:
        return False

    # Also delete model profiles associated with this provider
    profiles = db.query(ModelProfileDB).filter(ModelProfileDB.provider_id == provider_id, ModelProfileDB.user_id == user_id).all()
    profile_ids = [p.id for p in profiles]
    if profile_ids:
        db.query(RoleBindingDB).filter(RoleBindingDB.user_id == user_id, RoleBindingDB.model_profile_id.in_(profile_ids)).delete(synchronize_session=False)
        db.query(ModelProfileDB).filter(ModelProfileDB.provider_id == provider_id, ModelProfileDB.user_id == user_id).delete(synchronize_session=False)

    db.delete(provider)
    db.commit()
    return True


# --- ModelProfile CRUD Helpers ---

def list_model_profiles(db: Session, user_id: str) -> List[Dict[str, Any]]:
    migrate_legacy_user_llm_config(db, user_id)
    profiles = db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).all()
    providers_by_id = {pr.id: pr for pr in db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()}

    res = []
    for p in profiles:
        provider = providers_by_id.get(p.provider_id)
        res.append({
            "id": p.id,
            "user_id": p.user_id,
            "provider_id": p.provider_id,
            "provider_display_name": provider.display_name if provider else None,
            "provider_type": provider.provider_type if provider else None,
            "model_name": p.model_name,
            "display_name": p.display_name,
            "temperature": p.temperature,
            "max_tokens": p.max_tokens,
            "extra_params": p.extra_params,
            "tier": p.tier,
            "is_default": p.is_default,
            "created_at": p.created_at,
            "updated_at": p.updated_at,
        })
    return res


def sync_model_profiles_from_names(
    db: Session,
    user_id: str,
    model_names: List[str],
    provider_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Batch ensure ModelProfiles exist for a list of model names for the user."""
    migrate_legacy_user_llm_config(db, user_id)
    if not provider_id:
        prov = (
            db.query(ProviderDB)
            .filter(ProviderDB.user_id == user_id, ProviderDB.enabled == True)
            .first()
            or db.query(ProviderDB).filter(ProviderDB.user_id == user_id).first()
        )
        if not prov:
            now = _utcnow()
            prov = ProviderDB(
                id=uuid4().hex,
                user_id=user_id,
                provider_type="openai",
                base_url=None,
                display_name="默认厂商",
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            db.add(prov)
            db.commit()
            db.refresh(prov)
        provider_id = prov.id

    existing_profiles = db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).all()
    existing_names = {p.model_name for p in existing_profiles}

    now = _utcnow()
    added = False
    for m in model_names:
        m_clean = (m or "").strip()
        if m_clean and m_clean not in existing_names:
            profile_id = uuid4().hex
            new_profile = ModelProfileDB(
                id=profile_id,
                user_id=user_id,
                provider_id=provider_id,
                model_name=m_clean,
                display_name=f"{m_clean}",
                is_default=False,
                created_at=now,
                updated_at=now,
            )
            db.add(new_profile)
            existing_names.add(m_clean)
            added = True

    if added:
        db.commit()

    return list_model_profiles(db, user_id)


def create_model_profile(
    db: Session,
    user_id: str,
    model_name: str,
    display_name: Optional[str] = None,
    provider_id: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    tier: Optional[str] = None,
    is_default: bool = False,
) -> Dict[str, Any]:
    now = _utcnow()
    migrate_legacy_user_llm_config(db, user_id)
    if not provider_id:
        prov = (
            db.query(ProviderDB)
            .filter(ProviderDB.user_id == user_id, ProviderDB.enabled == True)
            .first()
            or db.query(ProviderDB).filter(ProviderDB.user_id == user_id).first()
        )
        if not prov:
            prov = ProviderDB(
                id=uuid4().hex,
                user_id=user_id,
                provider_type="openai",
                display_name="默认厂商",
                enabled=True,
                created_at=now,
                updated_at=now,
            )
            db.add(prov)
            db.commit()
            db.refresh(prov)
        provider_id = prov.id

    profile_id = uuid4().hex

    if is_default:
        # Clear default flag on existing profiles for this user
        db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).update({"is_default": False})

    profile = ModelProfileDB(
        id=profile_id,
        user_id=user_id,
        provider_id=provider_id,
        model_name=model_name,
        display_name=display_name,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_params=extra_params,
        tier=tier,
        is_default=is_default,
        created_at=now,
        updated_at=now,
    )
    db.add(profile)
    db.commit()
    db.refresh(profile)

    provider = db.query(ProviderDB).filter(ProviderDB.id == provider_id).first()
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "provider_id": profile.provider_id,
        "provider_display_name": provider.display_name if provider else None,
        "provider_type": provider.provider_type if provider else None,
        "model_name": profile.model_name,
        "display_name": profile.display_name,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "extra_params": profile.extra_params,
        "tier": profile.tier,
        "is_default": profile.is_default,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def update_model_profile(
    db: Session,
    user_id: str,
    profile_id: str,
    provider_id: Optional[str] = None,
    model_name: Optional[str] = None,
    display_name: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    extra_params: Optional[Dict[str, Any]] = None,
    tier: Optional[str] = None,
    is_default: Optional[bool] = None,
) -> Optional[Dict[str, Any]]:
    profile = db.query(ModelProfileDB).filter(ModelProfileDB.id == profile_id, ModelProfileDB.user_id == user_id).first()
    if not profile:
        return None

    if is_default is True:
        db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).update({"is_default": False})
        profile.is_default = True
    elif is_default is False:
        profile.is_default = False

    if provider_id is not None:
        profile.provider_id = provider_id
    if model_name is not None:
        profile.model_name = model_name
    if display_name is not None:
        profile.display_name = display_name
    if temperature is not None:
        profile.temperature = temperature
    if max_tokens is not None:
        profile.max_tokens = max_tokens
    if extra_params is not None:
        profile.extra_params = extra_params
    if tier is not None:
        profile.tier = tier

    profile.updated_at = _utcnow()
    db.commit()
    db.refresh(profile)

    provider = db.query(ProviderDB).filter(ProviderDB.id == profile.provider_id).first()
    return {
        "id": profile.id,
        "user_id": profile.user_id,
        "provider_id": profile.provider_id,
        "provider_display_name": provider.display_name if provider else None,
        "provider_type": provider.provider_type if provider else None,
        "model_name": profile.model_name,
        "display_name": profile.display_name,
        "temperature": profile.temperature,
        "max_tokens": profile.max_tokens,
        "extra_params": profile.extra_params,
        "tier": profile.tier,
        "is_default": profile.is_default,
        "created_at": profile.created_at,
        "updated_at": profile.updated_at,
    }


def delete_model_profile(db: Session, user_id: str, profile_id: str) -> bool:
    profile = db.query(ModelProfileDB).filter(ModelProfileDB.id == profile_id, ModelProfileDB.user_id == user_id).first()
    if not profile:
        return False

    db.query(RoleBindingDB).filter(RoleBindingDB.user_id == user_id, RoleBindingDB.model_profile_id == profile_id).delete(synchronize_session=False)
    db.delete(profile)
    db.commit()
    return True


# --- RoleBinding CRUD Helpers ---

def get_role_bindings(db: Session, user_id: str) -> List[Dict[str, Any]]:
    migrate_legacy_user_llm_config(db, user_id)
    bindings = db.query(RoleBindingDB).filter(RoleBindingDB.user_id == user_id).all()
    profiles_by_id = {p.id: p for p in db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id).all()}
    providers_by_id = {pr.id: pr for pr in db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()}

    res = []
    for b in bindings:
        profile = profiles_by_id.get(b.model_profile_id)
        provider = providers_by_id.get(profile.provider_id) if profile else None
        res.append({
            "id": b.id,
            "target_type": b.target_type,
            "target_key": b.target_key,
            "model_profile_id": b.model_profile_id,
            "model_profile_display_name": profile.display_name if profile else None,
            "model_name": profile.model_name if profile else None,
            "provider_type": provider.provider_type if provider else None,
        })
    return res


def update_role_bindings(db: Session, user_id: str, bindings: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    migrate_legacy_user_llm_config(db, user_id)
    
    # Delete existing bindings for user and re-insert
    db.query(RoleBindingDB).filter(RoleBindingDB.user_id == user_id).delete(synchronize_session=False)
    now = _utcnow()

    for item in bindings:
        rb = RoleBindingDB(
            id=uuid4().hex,
            user_id=user_id,
            target_type=item["target_type"],
            target_key=item["target_key"],
            model_profile_id=item["model_profile_id"],
            created_at=now,
            updated_at=now,
        )
        db.add(rb)

    db.commit()
    return get_role_bindings(db, user_id)


def apply_role_preset(
    db: Session,
    user_id: str,
    preset_mode: str,
    bull_profile_id: Optional[str] = None,
    bear_profile_id: Optional[str] = None,
    manager_profile_id: Optional[str] = None,
    quick_profile_id: Optional[str] = None,
    deep_profile_id: Optional[str] = None,
) -> Dict[str, Any]:
    migrate_legacy_user_llm_config(db, user_id)
    
    if quick_profile_id:
        db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id, ModelProfileDB.tier == "quick").update({"tier": None})
        db.query(ModelProfileDB).filter(ModelProfileDB.id == quick_profile_id, ModelProfileDB.user_id == user_id).update({"tier": "quick", "is_default": True})

    if deep_profile_id:
        db.query(ModelProfileDB).filter(ModelProfileDB.user_id == user_id, ModelProfileDB.tier == "deep").update({"tier": None})
        db.query(ModelProfileDB).filter(ModelProfileDB.id == deep_profile_id, ModelProfileDB.user_id == user_id).update({"tier": "deep"})

    db.query(RoleBindingDB).filter(RoleBindingDB.user_id == user_id).delete(synchronize_session=False)
    now = _utcnow()
    new_bindings = []

    if preset_mode == "single":
        # All roles fall back to quick / deep tier defaults
        pass
    elif preset_mode == "bull_bear_hetero":
        if bull_profile_id:
            new_bindings.append(RoleBindingDB(id=uuid4().hex, user_id=user_id, target_type="role", target_key="bull_researcher", model_profile_id=bull_profile_id, created_at=now, updated_at=now))
        if bear_profile_id:
            new_bindings.append(RoleBindingDB(id=uuid4().hex, user_id=user_id, target_type="role", target_key="bear_researcher", model_profile_id=bear_profile_id, created_at=now, updated_at=now))
    elif preset_mode == "three_way_hetero":
        if bull_profile_id:
            new_bindings.append(RoleBindingDB(id=uuid4().hex, user_id=user_id, target_type="role", target_key="bull_researcher", model_profile_id=bull_profile_id, created_at=now, updated_at=now))
        if bear_profile_id:
            new_bindings.append(RoleBindingDB(id=uuid4().hex, user_id=user_id, target_type="role", target_key="bear_researcher", model_profile_id=bear_profile_id, created_at=now, updated_at=now))
        if manager_profile_id:
            new_bindings.append(RoleBindingDB(id=uuid4().hex, user_id=user_id, target_type="role", target_key="research_manager", model_profile_id=manager_profile_id, created_at=now, updated_at=now))
    else:
        raise ValueError(f"Unknown preset mode: {preset_mode}")

    for rb in new_bindings:
        db.add(rb)

    db.commit()
    return {
        "preset_mode": preset_mode,
        "bindings": get_role_bindings(db, user_id),
        "resolved_roles": resolve_all_roles(db, user_id),
    }
