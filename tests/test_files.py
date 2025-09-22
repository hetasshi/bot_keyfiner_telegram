from src.utils.files import resolve_extension


def test_resolve_extension_defaults_to_bin() -> None:
    assert resolve_extension(None, None) == ".bin"


def test_resolve_extension_from_filename() -> None:
    assert resolve_extension("Song.MP3", None) == ".mp3"


def test_resolve_extension_from_mime() -> None:
    assert resolve_extension(None, "audio/ogg") == ".ogg"
