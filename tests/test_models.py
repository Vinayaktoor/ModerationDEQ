def test_model_import():
    from engine.model import ModerationDEQ
    model = ModerationDEQ()
    assert model is not None
