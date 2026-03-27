"""CW fist model, typing cadence model, Morse table, and Varicode encoding."""

import numpy as np


class CWFistModel:
    """Models individual CW operator keying characteristics ('fist').

    Parameters vary by keying type to simulate different operator equipment
    and skill levels.
    """
    PROFILES = {
        'electronic': {'weight': 0.40, 'dit_dah_range': (2.8, 3.2),
                       'jitter_std': (0.0, 0.05), 'farnsworth': (1.0, 1.2),
                       'rise_fall_ms': (5, 8)},
        'paddle':     {'weight': 0.30, 'dit_dah_range': (2.7, 3.3),
                       'jitter_std': (0.02, 0.10), 'farnsworth': (1.0, 1.5),
                       'rise_fall_ms': (3, 6)},
        'straight_key': {'weight': 0.20, 'dit_dah_range': (2.5, 3.5),
                         'jitter_std': (0.05, 0.15), 'farnsworth': (1.0, 2.0),
                         'rise_fall_ms': (1, 5)},
        'bug':        {'weight': 0.10, 'dit_dah_range': (2.5, 3.5),
                       'jitter_std': (0.03, 0.12), 'farnsworth': (1.0, 1.8),
                       'rise_fall_ms_dit': (5, 5), 'rise_fall_ms_dah': (1, 3)},
    }

    def __init__(self, keying_type=None):
        if keying_type is None:
            types = list(self.PROFILES.keys())
            weights = [self.PROFILES[t]['weight'] for t in types]
            keying_type = np.random.choice(types, p=weights)
        self.keying_type = keying_type
        p = self.PROFILES[keying_type]
        self.dit_dah_ratio = np.random.uniform(*p['dit_dah_range'])
        self.jitter_std = np.random.uniform(*p['jitter_std'])
        self.farnsworth = np.random.uniform(*p['farnsworth'])
        if keying_type == 'bug':
            self.rise_fall_dit = np.random.uniform(*p['rise_fall_ms_dit']) / 1000.0
            self.rise_fall_dah = np.random.uniform(*p['rise_fall_ms_dah']) / 1000.0
        else:
            rf = np.random.uniform(*p['rise_fall_ms']) / 1000.0
            self.rise_fall_dit = rf
            self.rise_fall_dah = rf

    def timing(self, element_type, unit_dur):
        """Return jittered duration for a dit, dah, or gap element."""
        jitter = np.random.normal(1.0, self.jitter_std)
        jitter = max(0.5, min(1.5, jitter))  # clamp
        if element_type == 'dit':
            return unit_dur * jitter
        elif element_type == 'dah':
            return unit_dur * self.dit_dah_ratio * jitter
        elif element_type == 'intra_char':
            return unit_dur * jitter
        elif element_type == 'inter_char':
            return unit_dur * 3 * self.farnsworth * jitter
        elif element_type == 'word':
            return unit_dur * 7 * self.farnsworth * jitter
        return unit_dur * jitter

    def rise_fall(self, element_type):
        """Return rise/fall time in seconds for the given element."""
        if element_type == 'dah':
            return self.rise_fall_dah
        return self.rise_fall_dit


class TypingCadenceModel:
    """Models realistic operator typing patterns for fldigi keyboard modes."""
    PROFILES = {
        'touch_typist':  {'weight': 0.30, 'wpm': (50, 80), 'pause_prob': 0.05,
                          'pause_range': (0.3, 1.5), 'typo_rate': 0.02},
        'hunt_and_peck': {'weight': 0.30, 'wpm': (15, 30), 'pause_prob': 0.15,
                          'pause_range': (1.0, 5.0), 'typo_rate': 0.05},
        'copy_paste':    {'weight': 0.20, 'wpm': (200, 500), 'pause_prob': 0.30,
                          'pause_range': (3.0, 8.0), 'typo_rate': 0.01},
        'mixed':         {'weight': 0.20, 'wpm': (20, 60), 'pause_prob': 0.10,
                          'pause_range': (0.5, 4.0), 'typo_rate': 0.04},
    }

    def __init__(self, profile=None):
        if profile is None:
            names = list(self.PROFILES.keys())
            weights = [self.PROFILES[n]['weight'] for n in names]
            profile = np.random.choice(names, p=weights)
        self.profile = profile
        p = self.PROFILES[profile]
        self.wpm = np.random.uniform(*p['wpm'])
        self.pause_prob = p['pause_prob']
        self.pause_range = p['pause_range']
        self.typo_rate = p['typo_rate']

    def char_delay(self):
        """Return delay in seconds before next character."""
        cps = self.wpm * 5.0 / 60.0
        base = 1.0 / max(1.0, cps)
        return base * np.random.lognormal(0, 0.3)

    def should_pause(self, char):
        """Return True if operator should pause after this character."""
        if char in '.?!\n':
            return np.random.random() < (self.pause_prob * 3)
        return np.random.random() < self.pause_prob

    def pause_duration(self):
        """Return think-pause duration in seconds."""
        return np.random.uniform(*self.pause_range)

    def should_typo(self):
        """Return True if operator makes a typo here."""
        return np.random.random() < self.typo_rate


MORSE_TABLE = {
    'A': '.-',    'B': '-...',  'C': '-.-.',  'D': '-..',   'E': '.',
    'F': '..-.',  'G': '--.',   'H': '....',  'I': '..',    'J': '.---',
    'K': '-.-',   'L': '.-..',  'M': '--',    'N': '-.',    'O': '---',
    'P': '.--.',  'Q': '--.-',  'R': '.-.',   'S': '...',   'T': '-',
    'U': '..-',   'V': '...-',  'W': '.--',   'X': '-..-',  'Y': '-.--',
    'Z': '--..',
    '0': '-----', '1': '.----', '2': '..---', '3': '...--', '4': '....-',
    '5': '.....', '6': '-....', '7': '--...', '8': '---..', '9': '----.',
    '.': '.-.-.-', ',': '--..--', '?': '..--..', '/': '-..-.',
    '=': '-...-',  '+': '.-.-.',
}

PROSIGNS = {
    'AR': '.-.-.',   # end of message
    'SK': '...-.-',  # end of QSO
    'BT': '-...-',   # break / new paragraph
    'KN': '-.--.',   # go ahead, named station only
}

VARICODE = {
    '\x00': '1010101011', '\x01': '1011011011', '\x02': '1011101101',
    '\x03': '1101110111', '\x04': '1011101011', '\x05': '1101011111',
    '\x06': '1011101111', '\x07': '1011111101', '\x08': '1011111111',
    '\t':   '11101111',   '\n':   '11101',
    '\x0b': '1101101111', '\x0c': '1011011101',
    '\r':   '11111',
    '\x0e': '1101110101', '\x0f': '1110101011',
    '\x10': '1011110111', '\x11': '1011110101', '\x12': '1110101101',
    '\x13': '1110101111', '\x14': '1101011011', '\x15': '1101101011',
    '\x16': '1101101101', '\x17': '1101010111', '\x18': '1101111011',
    '\x19': '1101111101', '\x1a': '1110110111', '\x1b': '1101010101',
    '\x1c': '1101011101', '\x1d': '1110111011', '\x1e': '1011111011',
    '\x1f': '1101111111',
    ' ':    '1',
    '!':    '111111111',  '"':    '101011111', '#':    '111110101',
    '$':    '111011011',  '%':    '1011010101','&':    '1010111011',
    "'":    '101111111',  '(':    '11111011',  ')':    '11110111',
    '*':    '101101111',  '+':    '111011111', ',':    '1110101',
    '-':    '110101',     '.':    '1010111',   '/':    '110101111',
    '0':    '10110111',   '1':    '10111101',  '2':    '11101101',
    '3':    '11111111',   '4':    '101110111', '5':    '101011011',
    '6':    '101101011',  '7':    '110101101', '8':    '110101011',
    '9':    '110110111',
    ':':    '11110101',   ';':    '110111101', '<':    '111101101',
    '=':    '1010101',    '>':    '111010111', '?':    '1010101111',
    '@':    '1010111101',
    'A':    '1111101',    'B':    '11101011',  'C':    '10101101',
    'D':    '10110101',   'E':    '1110111',   'F':    '11011011',
    'G':    '11111101',   'H':    '101010101', 'I':    '1111111',
    'J':    '111111101',  'K':    '101111101', 'L':    '11010111',
    'M':    '10111011',   'N':    '11011101',  'O':    '10101011',
    'P':    '11010101',   'Q':    '111011101', 'R':    '10101111',
    'S':    '1101111',    'T':    '1101101',   'U':    '101010111',
    'V':    '110110101',  'W':    '101011101', 'X':    '101110101',
    'Y':    '101111011',  'Z':    '1010101101',
    '[':    '111110111',  '\\':   '111101111', ']':    '111111011',
    '^':    '1010111111', '_':    '101101101',
    '`':    '1011011111',
    'a':    '1011',       'b':    '1011111',   'c':    '101111',
    'd':    '101101',     'e':    '11',        'f':    '111101',
    'g':    '1011011',    'h':    '101011',    'i':    '1101',
    'j':    '111101011',  'k':    '10111111',  'l':    '11011',
    'm':    '111011',     'n':    '1111',      'o':    '111',
    'p':    '111111',     'q':    '110111111', 'r':    '10101',
    's':    '10111',      't':    '101',       'u':    '110111',
    'v':    '1111011',    'w':    '1101011',   'x':    '11011111',
    'y':    '1011101',    'z':    '111010101',
    '{':    '1010110111', '|':    '110111011', '}':    '1010110101',
    '~':    '1011010111','\x7f': '1110110101',
}


def text_to_varicode_bits(text):
    """Encode text to PSK31 Varicode bit stream."""
    bits = []
    for char in text:
        code = VARICODE.get(char)
        if code is None:
            continue
        bits.extend(int(b) for b in code)
        bits.extend([0, 0])  # inter-character separator
    return bits


def text_to_morse_elements(text, fist, unit_dur):
    """Convert text string to Morse keying elements using fist model.

    Returns list of (duration_sec, is_on, element_type) tuples.
    """
    elements = []
    text = text.upper()

    for ci, char in enumerate(text):
        if char == ' ':
            elements.append((fist.timing('word', unit_dur), False, None))
            continue

        morse = MORSE_TABLE.get(char)
        if morse is None:
            continue

        for ei, symbol in enumerate(morse):
            if symbol == '.':
                elements.append((fist.timing('dit', unit_dur), True, 'dit'))
            elif symbol == '-':
                elements.append((fist.timing('dah', unit_dur), True, 'dah'))
            if ei < len(morse) - 1:
                elements.append((fist.timing('intra_char', unit_dur), False, None))

        elements.append((fist.timing('inter_char', unit_dur), False, None))

    return elements
