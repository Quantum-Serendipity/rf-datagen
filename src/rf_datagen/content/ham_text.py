"""Ham radio text content — procedural QSO, contest, and message generation.

All text generators produce realistic ham radio content with high diversity
for training signal classifiers.  Callsigns, grid squares, names, QTH, RST,
and equipment are procedurally generated so that 70K+ unique windows per mode
see minimal repetition.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Procedural building blocks
# ---------------------------------------------------------------------------

# ITU callsign prefix rules: (prefix_pattern, suffix_digits, suffix_letters)
_CALLSIGN_RULES = [
    # US: W, K, N, AA-AL + digit + 1-3 letters
    ("W", 1, (1, 3)),
    ("K", 1, (1, 3)),
    ("N", 1, (1, 3)),
    # US extra-class 2x1, 1x2
    ("W", 1, (1, 1)),
    ("K", 1, (2, 2)),
    # Canada
    ("VE", 1, (2, 3)),
    ("VA", 1, (2, 3)),
    # UK
    ("G", 1, (2, 3)),
    ("M", 1, (2, 3)),
    ("2E", 1, (2, 3)),
    # Germany
    ("DL", 1, (2, 3)),
    ("DK", 1, (2, 3)),
    ("DJ", 1, (2, 3)),
    # Japan
    ("JA", 1, (2, 3)),
    ("JH", 1, (2, 3)),
    ("JR", 1, (2, 3)),
    # Australia
    ("VK", 1, (2, 3)),
    # New Zealand
    ("ZL", 1, (2, 3)),
    # Brazil
    ("PY", 1, (2, 3)),
    ("PU", 1, (2, 3)),
    # Spain
    ("EA", 1, (2, 3)),
    # France
    ("F", 1, (2, 3)),
    # Italy
    ("I", 1, (2, 3)),
    ("IK", 1, (2, 3)),
    # Finland
    ("OH", 1, (2, 3)),
    # Sweden
    ("SM", 1, (2, 3)),
    ("SA", 1, (2, 3)),
    # Russia
    ("UA", 1, (2, 3)),
    ("RV", 1, (2, 3)),
    # Argentina
    ("LU", 1, (2, 3)),
    # South Africa
    ("ZS", 1, (2, 3)),
    # India
    ("VU", 1, (2, 3)),
    # South Korea
    ("HL", 1, (2, 3)),
    # Poland
    ("SP", 1, (2, 3)),
    # Czech Republic
    ("OK", 1, (2, 3)),
    # Netherlands
    ("PA", 1, (2, 3)),
    ("PD", 1, (2, 3)),
    # Belgium
    ("ON", 1, (2, 3)),
    # Portugal
    ("CT", 1, (2, 3)),
]

_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

_FIRST_NAMES = [
    "Bob", "Jim", "Tom", "Bill", "Dave", "John", "Mike", "Steve", "Dan",
    "Rick", "Joe", "Ed", "Mark", "Paul", "Pete", "Ken", "Don", "Ron",
    "Fred", "George", "Frank", "Larry", "Gary", "Jerry", "Ray", "Al",
    "Art", "Chuck", "Hank", "Herb", "Jack", "Lee", "Len", "Lou",
    "Ned", "Phil", "Ralph", "Roy", "Sam", "Ted", "Vince", "Walt",
    "Wayne", "Hans", "Klaus", "Yoshi", "Taro", "Pierre", "Jean",
    "Carlos", "Miguel", "Oleg", "Igor", "Sven", "Erik", "Pekka",
    "Marco", "Luca", "Andrzej", "Karel", "Jan", "Pedro", "Rui",
]

_US_STATES = [
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
    "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
    "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
    "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
    "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
]

_US_CITIES = [
    "New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
    "Philadelphia", "San Antonio", "San Diego", "Dallas", "Austin",
    "Denver", "Portland", "Seattle", "Atlanta", "Boston",
    "Nashville", "Tucson", "Omaha", "Raleigh", "Tampa",
    "Newington", "Dayton", "Orlando", "Sacramento", "Charlotte",
]

_DX_QTH = [
    "London", "Munich", "Paris", "Tokyo", "Sydney", "Auckland",
    "Sao Paulo", "Madrid", "Rome", "Helsinki", "Stockholm",
    "Moscow", "Buenos Aires", "Cape Town", "Mumbai", "Seoul",
    "Warsaw", "Prague", "Amsterdam", "Brussels", "Lisbon",
    "Berlin", "Hamburg", "Osaka", "Melbourne", "Toronto",
    "Vancouver", "Montreal", "Ottawa", "Calgary",
]

_RIGS = [
    "IC7300", "IC7610", "IC7851", "IC9700",
    "FT991A", "FTDX10", "FTDX101D", "FT710",
    "TS590SG", "TS890S", "TS990S",
    "K3S", "K4", "KX3", "KX2",
    "FLEX6600", "FLEX6700",
    "SDR1000", "HERMES LITE 2",
]

_ANTENNAS = [
    "dipole", "vertical", "3 el yagi", "4 el yagi", "5 el yagi",
    "hex beam", "fan dipole", "end fed", "G5RV", "loop",
    "OCF dipole", "delta loop", "inverted V", "long wire",
    "cobweb", "SteppIR", "log periodic", "quad",
    "mag loop", "Buddipole", "random wire", "doublet",
]

_BANDS = [
    "160m", "80m", "60m", "40m", "30m", "20m", "17m", "15m",
    "12m", "10m", "6m", "2m", "70cm",
]

_POWERS = [
    "5W", "10W", "25W", "50W", "100W", "200W", "400W",
    "500W", "1KW", "1.5KW", "QRP 5W",
]

_WX_CONDITIONS = [
    "clear", "partly cloudy", "cloudy", "overcast", "rainy",
    "snowing", "foggy", "windy", "hazy", "sunny",
]


def gen_callsign():
    """Generate a random realistic ham radio callsign."""
    prefix, n_digits, (min_let, max_let) = _CALLSIGN_RULES[
        np.random.randint(0, len(_CALLSIGN_RULES))]
    digit = str(np.random.randint(0, 10))
    n_suffix = np.random.randint(min_let, max_let + 1)
    suffix = "".join(np.random.choice(list(_LETTERS), n_suffix))
    return f"{prefix}{digit}{suffix}"


def gen_grid_square():
    """Generate a random valid 4-character Maidenhead grid square."""
    # Field: A-R (18 lon) x A-R (18 lat)
    lon_field = chr(ord("A") + np.random.randint(0, 18))
    lat_field = chr(ord("A") + np.random.randint(0, 18))
    # Square: 0-9 x 0-9
    lon_sq = str(np.random.randint(0, 10))
    lat_sq = str(np.random.randint(0, 10))
    return f"{lon_field}{lat_field}{lon_sq}{lat_sq}"


def gen_rst():
    """Generate a realistic RST signal report.

    Not always 599 — varies readability (1-5), strength (1-9), tone (1-9).
    Weighted toward good reports (common on air).
    """
    # Readability: mostly 4-5
    r = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
    # Strength: weighted toward 5-9
    s = np.random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9],
                         p=[0.01, 0.02, 0.05, 0.07, 0.15, 0.15, 0.2, 0.2, 0.15])
    # Tone: mostly 9
    t = np.random.choice([7, 8, 9], p=[0.05, 0.15, 0.8])
    return f"{r}{s}{t}"


def gen_name():
    """Generate a random operator first name."""
    return np.random.choice(_FIRST_NAMES)


def gen_qth():
    """Generate a random QTH (location)."""
    if np.random.random() < 0.5:
        return np.random.choice(_US_STATES)
    return np.random.choice(_DX_QTH)


def gen_temp():
    """Generate a random temperature string."""
    temp = np.random.randint(10, 100)
    return f"{temp}F"


# ---------------------------------------------------------------------------
# Keep a small static pool for backward compatibility (tests import CALLSIGNS)
# ---------------------------------------------------------------------------

CALLSIGNS = [
    "W1AW", "K3LR", "N0CALL", "W6ABC", "VE3XYZ", "G4ABC",
    "JA1ABC", "DL1ABC", "VK3DEF", "ZL1GHI", "PY2JKL", "EA3MNO",
    "F5PQR", "I2STU", "OH3VWX", "SM5YZA", "UA3BCD", "LU4EFG",
]

GRID_SQUARES = [
    "EM48", "FN31", "EN91", "DM79", "CM87", "IO91",
    "PM95", "QM06", "RE78", "OF37", "KP20", "GF15",
]


# ---------------------------------------------------------------------------
# Procedural QSO generation
# ---------------------------------------------------------------------------

def gen_cq():
    """Generate a CQ call with procedural callsign."""
    call = gen_callsign()
    r = np.random.random()
    if r < 0.3:
        return f"CQ CQ CQ DE {call} {call} {call} K"
    elif r < 0.5:
        return f"CQ DX CQ DX DE {call} {call} K"
    elif r < 0.7:
        return f"CQ TEST CQ TEST DE {call} {call} K"
    elif r < 0.85:
        return f"CQ CONTEST CQ CONTEST DE {call} {call} K"
    else:
        return f"QRZ QRZ DE {call} {call} K"


def gen_qso():
    """Generate a full multi-exchange QSO between two stations."""
    c1 = gen_callsign()
    c2 = gen_callsign()
    n1 = gen_name()
    n2 = gen_name()
    q1 = gen_qth()
    q2 = gen_qth()
    rst1 = gen_rst()
    rst2 = gen_rst()
    rig = np.random.choice(_RIGS)
    ant = np.random.choice(_ANTENNAS)
    pwr = np.random.choice(_POWERS)

    exchanges = [
        f"CQ CQ CQ DE {c1} {c1} {c1} K",
        f"{c1} {c1} DE {c2} {c2} K",
        f"{c2} DE {c1} GM ES TNX FER CALL UR RST {rst1} {rst1} "
        f"QTH {q1} NAME {n1} {n1} HW CPY K",
        f"{c1} DE {c2} R R TU UR RST {rst2} {rst2} QTH {q2} "
        f"NAME {n2} {n2} RIG HR {rig} PWR {pwr} ANT {ant} BK",
        f"{c2} DE {c1} R FB {n2} SOLID CPY VY 73 ES HPE CUAGN "
        f"DE {c1} SK",
        f"{c1} DE {c2} 73 {n1} GUD DX DE {c2} SK",
    ]
    return "\n".join(exchanges)


def gen_qso_short():
    """Generate a short 2-3 exchange QSO."""
    c1 = gen_callsign()
    c2 = gen_callsign()
    rst = gen_rst()
    name = gen_name()
    qth = gen_qth()

    r = np.random.random()
    if r < 0.5:
        return (f"DE {c1} UR RST {rst} {rst} QTH {qth} NAME {name} K\n"
                f"R R TNX FER RPT 73 DE {c2} SK")
    else:
        return (f"{c2} DE {c1} UR {rst} NAME {name} QTH {qth} K\n"
                f"TU {c1} DE {c2} 73 SK")


def gen_contest_qso():
    """Generate a realistic contest QSO exchange."""
    c1 = gen_callsign()
    c2 = gen_callsign()
    exch = gen_contest_exchange()
    return (f"CQ TEST DE {c1} {c1} K "
            f"{c2} {c2} DE {c1} UR {exch} K "
            f"TU {c1} DE {c2} 73 K")


def gen_contest_exchange():
    """Generate a varied contest exchange."""
    contest_type = np.random.random()
    if contest_type < 0.25:
        # CQ WW style: RST + zone
        rst = gen_rst()
        zone = np.random.randint(1, 41)
        return f"{rst} {zone:02d}"
    elif contest_type < 0.50:
        # ARRL SS style: serial + precedence + call + check + section
        serial = np.random.randint(1, 3000)
        prec = np.random.choice(["A", "B", "Q", "M", "S", "U"])
        check = np.random.randint(50, 99)
        section = np.random.choice(_US_STATES[:20])
        return f"{serial} {prec} {check} {section}"
    elif contest_type < 0.70:
        # Field Day: class + section
        n_tx = np.random.randint(1, 10)
        cat = np.random.choice(["A", "B", "C", "D", "E", "F"])
        section = np.random.choice(_US_STATES[:20])
        return f"{n_tx}{cat} {section}"
    elif contest_type < 0.85:
        # State QSO party: RST + state
        rst = gen_rst()
        state = np.random.choice(_US_STATES)
        return f"{rst} {state}"
    else:
        # Simple serial number
        rst = gen_rst()
        serial = np.random.randint(1, 2000)
        return f"{rst} {serial:03d}"


def gen_wx_report():
    """Generate a weather report string (common in QSOs)."""
    cond = np.random.choice(_WX_CONDITIONS)
    temp = gen_temp()
    wind_dir = np.random.choice(["N", "NE", "E", "SE", "S", "SW", "W", "NW"])
    wind_spd = np.random.randint(0, 35)
    return f"WX {cond} TEMP {temp} WIND {wind_dir} {wind_spd}"


def gen_station_info():
    """Generate station equipment description."""
    rig = np.random.choice(_RIGS)
    ant = np.random.choice(_ANTENNAS)
    pwr = np.random.choice(_POWERS)
    height = np.random.choice([20, 25, 30, 35, 40, 45, 50, 55, 60, 70, 80])
    return f"RIG {rig} PWR {pwr} ANT {ant} AT {height} FT"


def gen_propagation_report():
    """Generate a propagation conditions snippet."""
    sfi = np.random.randint(70, 250)
    a_idx = np.random.randint(0, 30)
    k_idx = np.random.randint(0, 7)
    band = np.random.choice(_BANDS[:10])
    cond = np.random.choice(["excellent", "good", "fair", "poor", "dead"])
    return f"SFI {sfi} A {a_idx} K {k_idx} {band} {cond}"


# ---------------------------------------------------------------------------
# Ragchew / conversation topics (procedurally varied)
# ---------------------------------------------------------------------------

_RAGCHEW_TEMPLATES = [
    "Built a new {ant} last weekend and hung it at {h} feet between two trees. "
    "Results have been amazing on {band}. Worked {n} stations first day.",
    "Got my license back in {year}. Started on CW with a {rig}. "
    "Have been at it for over {years} years now. Still love it.",
    "The grandkids were visiting and the youngest wanted to know what all "
    "the knobs on the {rig} were for. Let them listen to some {band} signals.",
    "Working on DXCC, at {n} countries confirmed now. Still need a few "
    "Pacific islands and some of the smaller African countries.",
    "This {rig} has been great. The noise blanker really helps with "
    "the power line noise from the transformer up the street.",
    "We had a great Field Day this year. Set up {n} stations and made "
    "over {contacts} contacts. Weather cooperated for once.",
    "My neighbor asked about amateur radio so I showed him the {rig}. "
    "He was amazed we could talk to {qth} without internet.",
    "Propagation on {band} has been {cond} lately with the solar cycle. "
    "SFI around {sfi}. Ten meters wide open to South America afternoons.",
    "Just finished building a {ant} for {band}. "
    "Took about {hours} hours in the workshop. Very happy with the SWR.",
    "Been experimenting with FT8 on {band}. Worked {n} countries in "
    "one afternoon with just {pwr} to a {ant}.",
    "Club meeting last week was about emergency communications. "
    "We practiced setting up a portable station with battery power.",
    "Upgraded to {rig} from my old rig. The difference in receiver "
    "performance is night and day. Much better on a crowded band.",
    "Planning a POTA activation this weekend at the state park. "
    "Taking the {rig} and a {ant}. Should be fun.",
    "Worked the ISS on {band} last week. Had a nice contact "
    "with the astronaut. Signal was about {rst}.",
]


def gen_ragchew():
    """Generate a procedurally varied ragchew conversation segment."""
    tmpl = np.random.choice(_RAGCHEW_TEMPLATES)
    return tmpl.format(
        ant=np.random.choice(_ANTENNAS),
        h=np.random.choice([20, 25, 30, 35, 40, 50, 60, 70]),
        band=np.random.choice(_BANDS[:10]),
        n=np.random.randint(5, 150),
        year=np.random.randint(1965, 2020),
        years=np.random.randint(5, 55),
        rig=np.random.choice(_RIGS),
        contacts=np.random.randint(200, 3000),
        qth=np.random.choice(_DX_QTH),
        cond=np.random.choice(["excellent", "good", "fair", "poor"]),
        sfi=np.random.randint(70, 250),
        hours=np.random.randint(2, 20),
        pwr=np.random.choice(_POWERS),
        rst=gen_rst(),
    )


# ---------------------------------------------------------------------------
# Static pools (kept for voice modes and backward compat, but supplemented
# by procedural generators above)
# ---------------------------------------------------------------------------

HAM_PHRASES = [
    "CQ CQ CQ DE W1AW W1AW W1AW PSE K",
    "CQ DX CQ DX DE K3LR K3LR K",
    "CQ TEST CQ TEST DE N0CALL N0CALL K",
    "DE W1AW UR RST 599 599 QTH CT CT NAME BOB BOB HW CPY K",
    "R R TU FER RPT HR UR 599 QTH NY NAME JIM JIM BK",
    "VY 73 ES GUD DX DE W1AW SK",
    "QRZ QRZ DE W1AW W1AW K",
    "AGN AGN PSE RPT UR CALL DE W1AW K",
    "CQ CQ CQ DE VE3ABC VE3ABC VE3ABC K",
    "DE DL1ABC UR RST 579 579 QTH MUNICH NAME HANS HW K",
    "R R TNX FER FB QSO DR OM 73 DE JA1XYZ SK",
    "CQ CONTEST CQ CONTEST DE W6DEF W6DEF K",
]

# Replace pangrams/lorem with ham-specific content (procedural fills the gap)
COMMON_TEXTS = [
    "Weather forecast for the northeast region: partly cloudy with highs near "
    "72 degrees. Winds from the southwest at 10 to 15 miles per hour.",
    "Station W1AW located in Newington, Connecticut. Operating on 14.070 MHz. "
    "Antenna is a 3 element yagi at 60 feet.",
    "Good morning from New York City. The temperature here is 65 degrees and "
    "rising. Conditions on 20 meters are excellent today.",
    "Thank you for the nice contact. This is my first time trying this digital "
    "mode. The software seems to be working very well today.",
    "Running 100 watts to a dipole antenna at 35 feet. The band has been very "
    "active today with lots of DX stations.",
    "Amateur radio provides a reliable means of communication during "
    "emergencies when other systems may fail.",
    "Solar flux index today is 145 with an A index of 5 and K index of 1. "
    "Conditions are favorable for long distance propagation on 20 meters.",
    "The Amateur Radio Emergency Service provides critical communications "
    "support during natural disasters and public events.",
    "Current propagation conditions show the MUF at approximately 28 MHz for "
    "paths between North America and Europe.",
    "Field Day is the most popular operating event in amateur radio, held "
    "annually on the fourth full weekend of June.",
]

NAVTEX_TEXTS = [
    "ZCZC BA01 NAVIGATIONAL WARNING NR 123 ENGLISH CHANNEL. UNLIT BUOY "
    "IN POSITION 50-30N 001-15W. MARINERS NAVIGATE WITH CAUTION. NNNN",
    "ZCZC BA02 METEOROLOGICAL WARNING. GALE WARNING ISSUED FOR SEA AREA "
    "PORTLAND. SOUTHWEST GALE FORCE 8 EXPECTED SOON. NNNN",
    "ZCZC QA03 SAR COORDINATION. ALL VESSELS IN VICINITY OF 48-20N 005-30W "
    "REQUESTED TO KEEP SHARP LOOKOUT AND REPORT ANY SIGHTING. NNNN",
    "ZCZC BA04 NAVIGATIONAL WARNING. CABLE LAYING OPERATIONS IN PROGRESS "
    "BETWEEN 51-10N 001-20E AND 51-15N 001-30E. WIDE BERTH REQUESTED. NNNN",
    "ZCZC BA05 METEOROLOGICAL FORECAST. NORTH SEA SOUTH. WIND SW 4 TO 5 "
    "INCREASING 6 TO 7 LATER. SEA STATE MODERATE BECOMING ROUGH. NNNN",
]

CONTEST_EXCHANGES = [
    "5NN 001", "5NN 002", "5NN 003", "5NN 004", "5NN 005",
    "599 CT", "599 NY", "599 CA", "599 TX", "599 FL",
    "59 001", "59 002", "59 003", "59 004", "59 005",
]

PSK_TEXTS = [
    "CQ CQ CQ de W1AW W1AW pse k",
    "CQ DX de K3LR K3LR k",
    "de W1AW ur rst 599 599 qth CT name Bob hw? k",
    "r r tu fer rpt hr ur 599 qth NY name Jim bk",
    "vy 73 es gud dx de W1AW sk",
    "Running 100 watts to a dipole at 35 feet.",
    "Band conditions on 20 meters are excellent today.",
    "Solar flux index today is 145 with A index of 5.",
    "Thank you for the nice contact on PSK31 today.",
    "Weather here is partly cloudy with temp around 72.",
    "Station W1AW in Newington Connecticut on 14070 kHz.",
    "Very nice to meet you on the air 73 and good DX.",
]

CW_PHRASES = [
    "CQ CQ CQ DE W1AW W1AW K",
    "CQ DX CQ DX DE K3LR K",
    "CQ TEST DE N0CALL K",
    "DE W1AW UR RST 599 QTH CT NAME BOB K",
    "R R TU UR 599 QTH NY NAME JIM BK",
    "VY 73 ES GUD DX DE W1AW SK",
    "QRZ DE W1AW K",
    "AGN PSE RPT UR CALL K",
    "DE VE3ABC UR 559 QTH ON K",
    "CQ CQ DE DL1ABC DL1ABC K",
    "R TNX FER FB QSO 73 DE JA1XYZ SK",
    "CQ CONTEST DE W6DEF K",
    "5NN TU DE W1AW",
    "HR WX CLEAR TEMP 72 ES WARM",
    "RIG HR IC7300 PWR 100W ANT DIPOLE",
    "CQ CQ DE VK3DEF VK3DEF K",
    "UR RST 579 579 QTH MUNICH NAME HANS K",
    "GE OM TNX FER CALL UR RST 599 K",
    "WX HR CLOUDY TEMP 55F WIND SW 10",
    "PSE QSL VIA BURO 73 DE G4ABC SK",
    "TEST TEST DE EA3MNO EA3MNO K",
    "R R SOLID COPY DR OM 73 ES DX SK",
]

# Analog voice content pools
HAM_QSO_TEXTS = [
    "CQ CQ CQ this is Whiskey One Alpha Whiskey calling CQ and standing by",
    "Good morning, this is Kilo Three Lima Romeo in Pennsylvania, over",
    "Roger roger, you are five nine here in Connecticut, name is Bob, over",
    "Thank you for the contact, your signal is very strong today",
    "CQ DX CQ DX this is Victor Echo Three X-ray Yankee Zulu",
    "This is November Zero Charlie Alpha Lima Lima calling CQ contest",
    "Your report is five seven, I copy you loud and clear",
    "Running one hundred watts into a three element yagi at sixty feet",
    "The weather here is partly cloudy with temperatures in the mid seventies",
    "Band conditions on twenty meters are excellent this morning",
    "I am portable today, operating from a hilltop with a wire antenna",
    "This is my first contact on this new antenna, very happy with results",
    "Solar flux index is one forty five, A index five, K index one",
    "Very nice to meet you on the air, seventy three and good DX",
    "QSL via bureau or direct, my address is on QRZ dot com",
    "The International Space Station passes overhead at twenty forty five UTC",
    "This is a special event station operating for Field Day weekend",
    "Our club station is located at the community center downtown",
    "I have been a ham radio operator for over thirty years now",
    "Propagation to Europe is excellent, worked several stations already",
]

BROADCAST_TEXTS = [
    "Good evening and welcome to the evening news. Our top story tonight.",
    "Weather forecast for the northeast region. Partly cloudy skies expected.",
    "The temperature today reached a high of eighty five degrees Fahrenheit.",
    "In international news, leaders gathered for a summit on trade policy.",
    "Sports update. The home team won last night's game by a score of five to three.",
    "Traffic advisory. Expect delays on the interstate due to construction.",
    "Time now is seven thirty PM eastern standard time. Stay tuned for more.",
    "This station broadcasts on the frequency of seven point two megahertz.",
    "Coming up next, our special program on amateur radio history.",
    "Market update. Trading was mixed today with technology shares leading gains.",
]

EMERGENCY_TEXTS = [
    "This is an emergency communication. All stations please stand by.",
    "ARES net is now active. All stations check in with your callsign.",
    "Priority traffic. Severe thunderstorm warning for the following counties.",
    "Skywarn net is active. Report severe weather observations on this frequency.",
    "Emergency traffic only. Please clear the frequency for emergency use.",
]

CONTEST_TEXTS = [
    "five nine zero three",
    "five nine fourteen",
    "five nine oh five",
    "contest, Whiskey One Alpha Whiskey",
    "CQ contest CQ contest Whiskey One Alpha Whiskey contest",
    "five nine Georgia",
    "five nine zero zero seven",
    "thanks, QRZ?",
    "you're five nine, number two forty seven",
    "copy, seventy three",
    "again?",
    "Whiskey One? again please",
    "QSO before?",
    "roger five nine, thanks QRZ",
    "five nine zero one, QSL?",
    "november four alpha foxtrot, five nine thirteen",
    "five nine Pennsylvania",
    "CQ test kilo three lima romeo",
    "you are five eight, QTH?",
    "zone fourteen, five nine",
]

NET_CHECKIN_TEXTS = [
    "Net control, Whiskey One Alpha Whiskey, checking in, no traffic",
    "Kilo Three Lima Romeo, checking in, I have one piece of traffic",
    "This is November Zero Charlie Alpha Lima Lima, present, no traffic",
    "Victor Echo Three X-ray Yankee Zulu, late check-in, no traffic",
    "Whiskey Four Bravo Papa, checking in from mobile",
    "Net control acknowledges, go ahead with your traffic",
    "All stations, this is the Monday evening two meter net",
    "Are there any additional check-ins? Going once, going twice",
    "The net is now closed, seventy three to all",
    "This is a directed net. Please wait to be recognized.",
    "Any relays? Any station with emergency or priority traffic?",
    "Stand by for the weekly announcements from the club secretary",
    "Alpha Alpha Four Romeo Golf, short time check-in, listening only",
    "Kilo Eight Juliet Mike, back to net control",
    "Net control, request to be excused",
]

DX_PILEUP_TEXTS = [
    "The station is Foxtrot Oscar Hotel, Foxtrot Oscar Hotel",
    "Oscar Hotel? Oscar Hotel? Go ahead",
    "You are five nine, QSL, QRZ?",
    "Whiskey One only, Whiskey One go ahead",
    "India? India station, go ahead please",
    "Up five, listening up five",
    "QRZ QRZ? the frequency is in use",
    "Listening, go ahead caller",
    "The callsign again? say again your call",
    "Golf three? was that Golf three?",
]

RAGCHEW_TEXTS = [
    "So I built this antenna last weekend, it's a fan dipole for forty and "
    "twenty meters, and I hung it between two trees in the backyard. My wife "
    "wasn't thrilled about the feedline running across the patio but the "
    "results have been amazing.",
    "I remember back in the seventies when I got my novice license, all we "
    "could do was CW on the low end of the bands. Had an old Heathkit "
    "receiver and a crystal-controlled transmitter.",
    "The grandkids were visiting this weekend and the youngest one, she's "
    "about eight, she wanted to know what all the knobs on the radio were "
    "for. I let her listen to some signals and she thought it was the "
    "coolest thing.",
    "I've been working on getting my DXCC confirmed, I'm at about ninety "
    "seven countries now. Still need a few Pacific islands and some of the "
    "smaller African countries.",
    "This new transceiver has been great. The noise blanker works really well "
    "for the power line noise I get from the transformer up the street. "
    "Used to be almost impossible to operate on eighty meters.",
    "We had a great field day this year. Set up four stations, two on HF "
    "and one on VHF and one on satellite. The weather cooperated for once "
    "and we made over two thousand contacts.",
    "My neighbor asked me about amateur radio last week so I invited him "
    "over for a demo. He was amazed that we could talk to someone in "
    "Australia without the internet or any infrastructure.",
    "The propagation has been weird lately with the solar cycle ramping up. "
    "Ten meters has been wide open to South America in the afternoons but "
    "twenty has been dead after dark.",
]

PHONETIC_WORDS = [
    "alpha", "bravo", "charlie", "delta", "echo", "foxtrot",
    "golf", "hotel", "india", "juliet", "kilo", "lima",
    "mike", "november", "oscar", "papa", "quebec", "romeo",
    "sierra", "tango", "uniform", "victor", "whiskey",
    "xray", "yankee", "zulu",
]

APRS_MESSAGES = [
    ">Hello World via packet radio",
    ">Testing 1200 baud packet",
    ">QSL via bureau or direct",
    ">Portable operation today",
    ">Field Day station active",
    ">New antenna installed",
    ">Band conditions excellent",
    ">Thanks for the contact",
    ">Running direwolf TNC",
    ">Good morning from New York",
    "!4903.50N/07201.75W-PHG2360/Home station",
    "@092345z4903.50N/07201.75W_090/010g020t077",
    ":W1AW    :Testing{001",
    ">Solar flux 145 A=5 K=1",
    ">ARES net check-in",
]


# JS8Call message patterns
JS8_MESSAGES = [
    "@HB HEARTBEAT",
    "@APRSIS GRID",
    "CQ CQ CQ DE W1AW W1AW",
    "CQ CQ CQ DE K3LR K3LR",
    "CQ DX DE VE3XYZ VE3XYZ",
    "W1AW: K3LR SNR +05",
    "K3LR: W1AW SNR -02",
    "W1AW: K3LR ACK",
    "N0CALL: CQ CQ CQ DE N0CALL",
    "W6ABC: DL1ABC RR 73",
    "@HB AUTO RELAY SPOT",
    "@APRSIS CMD :W1AW :GRID?",
    "VE3XYZ: QSL VIA LOTW 73",
    "DL1ABC: W1AW INFO? ",
    "W1AW: DL1ABC INFO QTH CT PWR 100W ANT 3 EL YAGI",
    "JA1ABC: CQ ASIA DE JA1ABC",
    "@HB HB W1AW EM48",
    "K3LR: MSG W1AW NICE QSO 73",
    "W1AW: HEARING K3LR VE3XYZ DL1ABC JA1ABC",
    "N0CALL: @ALLCALL? QST NET AT 0100Z ON 7078",
]

JS8_HEARTBEAT_TEMPLATES = [
    "@HB HEARTBEAT {call} {grid}",
    "@HB HB {call} {grid}",
]

JS8_DIRECTED_TEMPLATES = [
    "{call1}: {call2} SNR {snr}",
    "{call1}: {call2} ACK",
    "{call1}: {call2} RR 73",
    "{call1}: {call2} GRID?",
    "{call1}: {call2} INFO?",
    "{call1}: {call2} MSG TU FER QSO 73",
    "{call1}: {call2} HEARING {call3}",
]


def gen_js8_message():
    """Generate a random JS8Call message (heartbeat, directed, or CQ)."""
    r = np.random.random()
    c1 = gen_callsign()
    c2 = gen_callsign()
    grid = gen_grid_square()

    if r < 0.25:
        tmpl = np.random.choice(JS8_HEARTBEAT_TEMPLATES)
        return tmpl.format(call=c1, grid=grid)
    elif r < 0.60:
        c3 = gen_callsign()
        snr = np.random.choice(["-15", "-10", "-05", "+00", "+05", "+10", "+15"])
        tmpl = np.random.choice(JS8_DIRECTED_TEMPLATES)
        return tmpl.format(call1=c1, call2=c2, call3=c3, snr=snr)
    elif r < 0.80:
        return f"CQ CQ CQ DE {c1} {c1}"
    else:
        return np.random.choice(JS8_MESSAGES)


# SAME/EAS constants
SAME_ORIGINATORS = ["PEP", "CIV", "WXR", "EAS"]
SAME_EVENT_CODES = [
    "TOR", "SVR", "FFW", "SVS", "SMW", "SPS", "FRW", "EWW",
    "HWA", "HUA", "HUW", "TSA", "TSW", "WSW", "BZW", "WCW",
    "RWT", "DMO", "ADR", "NIC", "NPT", "RMT", "EAN", "EAT",
]
SAME_FIPS = [
    "029510", "017031", "048201", "006037", "036061",
    "012086", "042101", "025025", "039049", "051810",
    "013121", "024005", "055079", "034013", "008031",
]
SAME_CALLSIGNS = [
    "KEAX/NWS", "KLOX/NWS", "WACN/NWS", "KMOX/NWS",
    "KGYX/NWS", "WJON/NWS", "KLBX/NWS", "KMPX/NWS",
]


def gen_same_message():
    """Generate a random SAME/EAS header string."""
    orig = np.random.choice(SAME_ORIGINATORS)
    event = np.random.choice(SAME_EVENT_CODES)
    n_areas = np.random.randint(1, 4)
    areas = "-".join(np.random.choice(SAME_FIPS, n_areas, replace=False))
    hours = np.random.randint(0, 3)
    minutes = np.random.choice([0, 15, 30, 45])
    duration = f"{hours:02d}{minutes:02d}"
    callsign = np.random.choice(SAME_CALLSIGNS)
    return f"ZCZC-{orig}-{event}-{areas}+{duration}-{callsign}-"


def gen_minimodem_text(mode):
    """Generate text content for minimodem modes."""
    if mode == "RTTY":
        return get_text_for_mode("RTTY", target_chars=2000)
    parts = []
    for _ in range(np.random.randint(5, 15)):
        parts.append(gen_ragchew())
    return "\n".join(parts)


def gen_navtex_message():
    """Generate a procedural NAVTEX message."""
    station = chr(ord("A") + np.random.randint(0, 26))
    msg_type = chr(ord("A") + np.random.randint(0, 8))
    msg_num = np.random.randint(1, 100)
    lat_d = np.random.randint(30, 65)
    lat_m = np.random.randint(0, 60)
    lon_d = np.random.randint(0, 30)
    lon_m = np.random.randint(0, 60)
    ns = np.random.choice(["N", "S"])
    ew = np.random.choice(["E", "W"])

    msg_types = [
        f"NAVIGATIONAL WARNING NR {msg_num}. UNLIT BUOY IN POSITION "
        f"{lat_d}-{lat_m:02d}{ns} {lon_d:03d}-{lon_m:02d}{ew}. "
        "MARINERS NAVIGATE WITH CAUTION.",
        f"METEOROLOGICAL WARNING. GALE WARNING ISSUED. "
        f"WIND {np.random.choice(['SW', 'NW', 'NE', 'SE', 'N', 'S', 'W', 'E'])} "
        f"FORCE {np.random.randint(6, 12)} EXPECTED.",
        f"CABLE LAYING OPERATIONS IN PROGRESS BETWEEN "
        f"{lat_d}-{lat_m:02d}{ns} {lon_d:03d}-{lon_m:02d}{ew} AND "
        f"{lat_d+1}-{np.random.randint(0,60):02d}{ns} "
        f"{lon_d:03d}-{np.random.randint(0,60):02d}{ew}. WIDE BERTH REQUESTED.",
        f"METEOROLOGICAL FORECAST. WIND "
        f"{np.random.choice(['SW', 'NW', 'NE', 'SE'])} "
        f"{np.random.randint(3,6)} TO {np.random.randint(6,9)} "
        f"INCREASING {np.random.randint(7,12)} LATER. "
        f"SEA STATE {np.random.choice(['MODERATE', 'ROUGH', 'VERY ROUGH'])}.",
    ]
    body = np.random.choice(msg_types)
    return f"ZCZC {station}{msg_type}{msg_num:02d} {body} NNNN"


# ---------------------------------------------------------------------------
# Main text generation entry points
# ---------------------------------------------------------------------------

def get_text_for_mode(mode, target_chars=3000):
    """Generate varied text content appropriate for a given digital mode.

    Uses procedural generators for high diversity — each call produces
    unique content from combinatorial callsign/QSO/contest generation.
    """
    parts = []
    current_len = 0

    while current_len < target_chars:
        if mode == "NAVTEX":
            parts.append(gen_navtex_message())
        elif mode in ("CW", "RTTY", "HELLSCHREIBER"):
            r = np.random.random()
            if r < 0.25:
                parts.append(gen_cq().upper())
            elif r < 0.50:
                parts.append(gen_contest_qso().upper())
            elif r < 0.70:
                parts.append(gen_qso_short().upper())
            elif r < 0.85:
                parts.append(gen_wx_report().upper())
            else:
                parts.append(gen_station_info().upper())
        elif mode == "PACKET":
            parts.append(gen_packet_content(n_packets=5))
        elif mode == "THROB":
            r = np.random.random()
            if r < 0.4:
                parts.append(gen_cq().upper())
            elif r < 0.7:
                parts.append(gen_qso_short().upper())
            else:
                parts.append(gen_wx_report().upper())
        elif mode == "FSQ":
            c1 = gen_callsign()
            c2 = gen_callsign()
            r = np.random.random()
            if r < 0.3:
                parts.append(f"{c1}: CQ CQ CQ de {c1} {c1} K")
            elif r < 0.5:
                parts.append(f"{c1}: {c2} UR {gen_rst()} NAME {gen_name()} "
                             f"QTH {gen_qth()} K")
            elif r < 0.7:
                parts.append(f"{c1}: {c2} 73 de {c1}")
            else:
                parts.append(f"{c1}: {gen_wx_report()}")
        else:
            # PSK31, PSK63, OLIVIA, CONTESTIA, MFSK, MT63, DOMINOEX, etc.
            r = np.random.random()
            if r < 0.20:
                parts.append(gen_cq())
            elif r < 0.40:
                parts.append(gen_contest_qso())
            elif r < 0.60:
                parts.append(gen_qso_short())
            elif r < 0.75:
                parts.append(gen_ragchew())
            elif r < 0.85:
                parts.append(gen_wx_report())
            elif r < 0.95:
                parts.append(gen_station_info())
            else:
                parts.append(gen_propagation_report())

        current_len = sum(len(t) for t in parts)

    return "\n".join(parts)


def gen_speech_text():
    """Generate random speech text from diverse ham radio scenarios.

    Returns (text, style) where style is one of:
        "casual", "contest", "net", "dx"
    """
    r = np.random.random()
    if r < 0.20:
        return np.random.choice(HAM_QSO_TEXTS), "casual"
    elif r < 0.35:
        return np.random.choice(CONTEST_TEXTS), "contest"
    elif r < 0.50:
        return np.random.choice(NET_CHECKIN_TEXTS), "net"
    elif r < 0.60:
        return np.random.choice(DX_PILEUP_TEXTS), "dx"
    elif r < 0.75:
        return np.random.choice(RAGCHEW_TEXTS), "casual"
    elif r < 0.85:
        return np.random.choice(BROADCAST_TEXTS), "casual"
    elif r < 0.95:
        return np.random.choice(EMERGENCY_TEXTS), "net"
    else:
        n = np.random.randint(5, 15)
        words = np.random.choice(PHONETIC_WORDS, n)
        return " ".join(words), "casual"


def gen_ft8_message():
    """Generate a random valid FT8/FT4 message."""
    msg_type = np.random.random()
    c1 = gen_callsign()
    c2 = gen_callsign()
    grid = gen_grid_square()
    if msg_type < 0.3:
        return f"CQ {c1} {grid}"
    elif msg_type < 0.6:
        rpt = np.random.choice(["-15", "-10", "-05", "+00", "+05", "+10"])
        return f"{c1} {c2} {rpt}"
    elif msg_type < 0.8:
        return f"{c1} {c2} R{np.random.choice(['-10', '-05', '+00', '+05'])}"
    else:
        return f"{c1} {c2} RR73"


def gen_wspr_message():
    """Generate a random WSPR message: callsign grid power."""
    call = gen_callsign()
    grid = gen_grid_square()[:4]
    power = np.random.choice([10, 20, 23, 27, 30, 33, 37])
    return f"{call} {grid} {power}"


def gen_packet_content(n_packets=50):
    """Generate packet content suitable for gen_packets input."""
    lines = []
    for _ in range(n_packets):
        src = gen_callsign()
        dst = "APRS" if np.random.random() < 0.5 else gen_callsign()
        r = np.random.random()
        if r < 0.3:
            msg = np.random.choice(APRS_MESSAGES)
        elif r < 0.6:
            # Position report
            lat_d = np.random.randint(25, 50)
            lat_m = np.random.uniform(0, 60)
            lon_d = np.random.randint(65, 125)
            lon_m = np.random.uniform(0, 60)
            msg = f"!{lat_d}{lat_m:05.2f}N/{lon_d:03d}{lon_m:05.2f}W-"
        else:
            # Status message
            msg = f">{gen_wx_report()}"
        lines.append(f"{src}>{dst}:{msg}")
    return "\n".join(lines)
