from legal_review_env.server.app import app, main as _main

__all__ = ["app", "main"]


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    _main(host=host, port=port)


if __name__ == "__main__":
    main()