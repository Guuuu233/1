"""Settings-page URL/Key must cascade onto providers used by role routing."""
from uuid import uuid4

from api.database import Base, ProviderDB, UserDB, UserLLMConfigDB
from api.services import auth_service


def _make_session():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    return TestingSessionLocal()


def _seed_user_with_stale_providers(db, *, other_user=False):
    user_id = uuid4().hex
    db.add(UserDB(id=user_id, email=f"{user_id}@test.local"))
    db.add(
        UserLLMConfigDB(
            user_id=user_id,
            backend_url="http://100.65.130.33:8317/v1",
            llm_provider="openai",
        )
    )
    stale = "http://92.119.124.146:8317/v1"
    for i in range(2 if not other_user else 1):
        db.add(
            ProviderDB(
                id=uuid4().hex,
                user_id=user_id,
                provider_type="openai",
                base_url=stale,
                display_name=f"默认厂商 {i}",
                enabled=True,
            )
        )
    db.commit()
    return user_id


def test_saving_backend_url_cascades_to_all_user_providers():
    db = _make_session()
    try:
        user_id = _seed_user_with_stale_providers(db)
        other_id = _seed_user_with_stale_providers(db, other_user=True)
        new_url = "http://100.65.130.33:8317/v1"

        auth_service.upsert_user_llm_config(db, user_id, backend_url=new_url)

        mine = [p.base_url for p in db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()]
        other = [p.base_url for p in db.query(ProviderDB).filter(ProviderDB.user_id == other_id).all()]
        assert mine == [new_url, new_url]
        assert other == ["http://92.119.124.146:8317/v1"]
        cfg = db.query(UserLLMConfigDB).filter(UserLLMConfigDB.user_id == user_id).one()
        assert cfg.backend_url == new_url
        assert cfg.backend_url is not None
    finally:
        db.close()


def test_saving_api_key_cascades_to_all_user_providers():
    db = _make_session()
    try:
        user_id = _seed_user_with_stale_providers(db)
        auth_service.upsert_user_llm_config(db, user_id, api_key="sk-user-new")

        cfg = db.query(UserLLMConfigDB).filter(UserLLMConfigDB.user_id == user_id).one()
        providers = db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()
        assert cfg.api_key_encrypted
        assert all(p.api_key_encrypted == cfg.api_key_encrypted for p in providers)
    finally:
        db.close()


def test_clear_api_key_clears_provider_keys():
    db = _make_session()
    try:
        user_id = _seed_user_with_stale_providers(db)
        auth_service.upsert_user_llm_config(db, user_id, api_key="sk-user-new")
        auth_service.upsert_user_llm_config(db, user_id, clear_api_key=True)

        cfg = db.query(UserLLMConfigDB).filter(UserLLMConfigDB.user_id == user_id).one()
        providers = db.query(ProviderDB).filter(ProviderDB.user_id == user_id).all()
        assert cfg.api_key_encrypted is None
        assert all(p.api_key_encrypted is None for p in providers)
    finally:
        db.close()
