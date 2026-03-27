"""Ham radio text content pools — QSO phrases, callsigns, contest exchanges."""

import numpy as np


CALLSIGNS = [
    "W1AW", "K3LR", "N0CALL", "W6ABC", "VE3XYZ", "G4ABC",
    "JA1ABC", "DL1ABC", "VK3DEF", "ZL1GHI", "PY2JKL", "EA3MNO",
    "F5PQR", "I2STU", "OH3VWX", "SM5YZA", "UA3BCD", "LU4EFG",
]

GRID_SQUARES = [
    "EM48", "FN31", "EN91", "DM79", "CM87", "IO91",
    "PM95", "QM06", "RE78", "OF37", "KP20", "GF15",
]

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

COMMON_TEXTS = [
    "The quick brown fox jumped over the lazy dog.",
    "Now is the time for all good men to come to the aid of their country.",
    "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod "
    "tempor incididunt ut labore et dolore magna aliqua.",
    "Pack my box with five dozen liquor jugs.",
    "How vexingly quick daft zebras jump.",
    "The five boxing wizards jump quickly.",
    "Sphinx of black quartz, judge my vow.",
    "Two driven jocks help fax my big quiz.",
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
    "The International Space Station passes overhead at approximately 2045 UTC "
    "tonight on a northeast heading. Elevation 67 degrees maximum.",
    "Solar flux index today is 145 with an A index of 5 and K index of 1. "
    "Conditions are favorable for long distance propagation on 20 meters.",
    "This is a test transmission for digital mode evaluation. Testing one two "
    "three four five six seven eight nine zero.",
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
    "The quick brown fox jumped over the lazy dog.",
    "Now is the time for all good men to come to the aid.",
    "Running 100 watts to a dipole at 35 feet.",
    "Band conditions on 20 meters are excellent today.",
    "Solar flux index today is 145 with A index of 5.",
    "Thank you for the nice contact on PSK31 today.",
    "This is my first time trying this digital mode.",
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
    "THE QUICK BROWN FOX JUMPED OVER THE LAZY DOG",
    "NOW IS THE TIME FOR ALL GOOD MEN",
    "PACK MY BOX WITH FIVE DOZEN LIQUOR JUGS",
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


def gen_contest_qso():
    """Generate a realistic contest QSO exchange."""
    c1, c2 = np.random.choice(CALLSIGNS, 2, replace=False)
    exch = np.random.choice(CONTEST_EXCHANGES)
    return (f"CQ TEST DE {c1} {c1} K "
            f"{c2} {c2} DE {c1} UR {exch} {exch} K "
            f"TU {c1} DE {c2} 73 K")


def get_text_for_mode(mode, target_chars=3000):
    """Generate varied text content appropriate for a given digital mode."""
    parts = []
    current_len = 0

    while current_len < target_chars:
        if mode == "NAVTEX":
            parts.append(np.random.choice(NAVTEX_TEXTS))
        elif mode in ("CW", "RTTY", "HELLSCHREIBER"):
            r = np.random.random()
            if r < 0.3:
                parts.append(np.random.choice(HAM_PHRASES).upper())
            elif r < 0.5:
                parts.append(gen_contest_qso().upper())
            else:
                parts.append(np.random.choice(COMMON_TEXTS).upper())
        elif mode == "PACKET":
            parts.append(f">{np.random.choice(COMMON_TEXTS)[:80]}")
        elif mode == "THROB":
            r = np.random.random()
            if r < 0.4:
                parts.append(np.random.choice(HAM_PHRASES).upper())
            else:
                parts.append(np.random.choice(COMMON_TEXTS).upper()[:60])
        elif mode == "FSQ":
            c1, c2 = np.random.choice(CALLSIGNS, 2, replace=False)
            r = np.random.random()
            if r < 0.3:
                parts.append(f"{c1}: CQ CQ CQ de {c1} {c1} K")
            elif r < 0.6:
                parts.append(f"{c1}: {np.random.choice(COMMON_TEXTS)[:80]}")
            else:
                parts.append(f"{c1}: {c2} 73 de {c1}")
        else:
            r = np.random.random()
            if r < 0.2:
                parts.append(np.random.choice(HAM_PHRASES))
            elif r < 0.4:
                parts.append(gen_contest_qso())
            else:
                parts.append(np.random.choice(COMMON_TEXTS))

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
    c1 = np.random.choice(CALLSIGNS)
    c2 = np.random.choice(CALLSIGNS)
    grid = np.random.choice(GRID_SQUARES)
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
    call = np.random.choice(CALLSIGNS)
    grid = np.random.choice(GRID_SQUARES)[:4]
    power = np.random.choice([10, 20, 23, 27, 30, 33, 37])
    return f"{call} {grid} {power}"


def gen_packet_content(n_packets=50):
    """Generate packet content suitable for gen_packets input."""
    lines = []
    for _ in range(n_packets):
        src = np.random.choice(CALLSIGNS)
        dst = "APRS" if np.random.random() < 0.5 else np.random.choice(CALLSIGNS)
        msg = np.random.choice(APRS_MESSAGES)
        lines.append(f"{src}>{dst}:{msg}")
    return "\n".join(lines)


