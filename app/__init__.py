def __getattr__(name):
    if name == "app":
        from app.main import app
        return app
    raise AttributeError(f"module 'app' has no attribute {name!r}")
