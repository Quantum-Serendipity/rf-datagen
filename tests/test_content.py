"""Unit tests for rf_datagen.content modules (ham_text, typing, images)."""

import re

from rf_datagen.content.ham_text import (
    gen_contest_qso,
    get_text_for_mode,
    gen_speech_text,
    gen_ft8_message,
    gen_wspr_message,
    gen_packet_content,
    gen_callsign,
    CALLSIGNS,
)
from rf_datagen.content.typing import (
    CWFistModel,
    TypingCadenceModel,
    text_to_varicode_bits,
    text_to_morse_elements,
)
from rf_datagen.content.images import random_image


# ---------------------------------------------------------------------------
# ham_text.py
# ---------------------------------------------------------------------------

def test_contest_qso_returns_string():
    assert type(gen_contest_qso()) is str


def test_contest_qso_contains_callsign():
    qso = gen_contest_qso()
    # Procedural callsigns follow DE <CALLSIGN> pattern
    assert re.search(r"DE [A-Z0-9]{2,8}", qso)


def test_gen_callsign_format():
    for _ in range(50):
        call = gen_callsign()
        assert re.match(r"^[A-Z0-9]{2,8}$", call), f"Bad callsign: {call}"


def test_get_text_reaches_target_length():
    output = get_text_for_mode("CW", target_chars=500)
    assert len(output) >= 500


def test_navtex_contains_zczc():
    output = get_text_for_mode("NAVTEX")
    assert "ZCZC" in output


def test_cw_is_uppercase():
    output = get_text_for_mode("CW")
    assert output.upper() == output


def test_speech_text_returns_tuple():
    result = gen_speech_text()
    assert isinstance(result, tuple)
    assert len(result) == 2
    assert isinstance(result[0], str)
    assert isinstance(result[1], str)


def test_speech_text_style_is_valid():
    _, style = gen_speech_text()
    assert style in {"casual", "contest", "net", "dx"}


def test_ft8_is_string():
    assert type(gen_ft8_message()) is str


def test_ft8_length_bounded():
    assert len(gen_ft8_message()) < 30


def test_wspr_three_fields():
    parts = gen_wspr_message().split()
    assert len(parts) == 3


def test_wspr_power_valid():
    parts = gen_wspr_message().split()
    power = int(parts[2])
    assert power in {10, 20, 23, 27, 30, 33, 37}


def test_packet_line_count():
    output = gen_packet_content(n_packets=10)
    lines = [ln for ln in output.split("\n") if ln]
    assert len(lines) == 10


def test_packet_has_separator():
    output = gen_packet_content(n_packets=10)
    lines = [ln for ln in output.split("\n") if ln]
    for line in lines:
        assert ">" in line


# ---------------------------------------------------------------------------
# typing.py
# ---------------------------------------------------------------------------

def test_cw_timing_positive():
    fist = CWFistModel("electronic")
    assert fist.timing("dit", 0.06) > 0


def test_cw_dah_longer_than_dit():
    fist = CWFistModel("electronic")
    unit_dur = 0.06
    n = 50
    dit_total = sum(fist.timing("dit", unit_dur) for _ in range(n))
    dah_total = sum(fist.timing("dah", unit_dur) for _ in range(n))
    avg_dit = dit_total / n
    avg_dah = dah_total / n
    assert avg_dah > 2 * avg_dit


def test_cw_rise_fall_positive():
    fist = CWFistModel("electronic")
    assert fist.rise_fall("dit") > 0


def test_typing_char_delay_positive():
    model = TypingCadenceModel("touch_typist")
    assert model.char_delay() > 0


def test_typing_should_pause_returns_bool():
    model = TypingCadenceModel("touch_typist")
    assert isinstance(model.should_pause("a"), bool)


def test_varicode_returns_list_of_ints():
    bits = text_to_varicode_bits("AB")
    assert len(bits) > 0
    assert all(isinstance(b, int) for b in bits)


def test_varicode_bits_are_binary():
    bits = text_to_varicode_bits("AB")
    assert all(b in (0, 1) for b in bits)


def test_varicode_separator_between_chars():
    bits = text_to_varicode_bits("AB")
    # The [0, 0] separator must appear somewhere in the middle portion
    # (not just at the very end appended after the last char).
    # Find the first occurrence of [0, 0] that is not in the trailing two bits.
    found = False
    for i in range(len(bits) - 3):
        if bits[i] == 0 and bits[i + 1] == 0:
            found = True
            break
    assert found


def test_morse_returns_list_of_tuples():
    fist = CWFistModel("electronic")
    elements = text_to_morse_elements("E", fist, 0.06)
    assert len(elements) > 0
    for elem in elements:
        assert isinstance(elem, tuple)
        assert len(elem) == 3
        dur, is_on, etype = elem
        assert isinstance(dur, float)
        assert isinstance(is_on, bool)
        assert etype is None or isinstance(etype, str)


def test_morse_e_is_single_dit():
    fist = CWFistModel("electronic")
    elements = text_to_morse_elements("E", fist, 0.06)
    on_dits = [e for e in elements if e[1] is True and e[2] == "dit"]
    on_dahs = [e for e in elements if e[1] is True and e[2] == "dah"]
    assert len(on_dits) == 1
    assert len(on_dahs) == 0


def test_morse_t_is_single_dah():
    fist = CWFistModel("electronic")
    elements = text_to_morse_elements("T", fist, 0.06)
    on_dahs = [e for e in elements if e[1] is True and e[2] == "dah"]
    on_dits = [e for e in elements if e[1] is True and e[2] == "dit"]
    assert len(on_dahs) == 1
    assert len(on_dits) == 0


# ---------------------------------------------------------------------------
# images.py
# ---------------------------------------------------------------------------

def test_random_image_dimensions():
    img = random_image(320, 256)
    assert img.size == (320, 256)


def test_random_image_is_rgb():
    img = random_image(320, 256)
    assert img.mode == "RGB"
