from src.bot import messages


def test_file_too_large_message() -> None:
    message = messages.file_too_large(70)

    assert "слишком большой" in message
    assert "70" in message
