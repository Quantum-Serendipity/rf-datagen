"""Protocol sync word constants from published specifications."""

import numpy as np

# DMR Base Station Voice Sync: 0x755FD7DF75F7 (ETSI TS 102 361-1 Table 9.2)
DMR_BS_VOICE_SYNC = np.array([
    0,1,1,1, 0,1,0,1, 0,1,0,1, 1,1,1,1,  # 0x755F
    1,1,0,1, 0,1,1,1, 1,1,0,1, 1,1,1,1,  # 0xD7DF
    0,1,1,1, 0,1,0,1, 1,1,1,1, 0,1,1,1,  # 0x75F7
], dtype=np.uint8)

# DMR BS Data Sync: 0xDFF57D75DF5D (ETSI TS 102 361-1 Table 9.2)
DMR_BS_DATA_SYNC = np.array([
    1,1,0,1, 1,1,1,1, 1,1,1,1, 0,1,0,1,  # 0xDFF5
    0,1,1,1, 1,1,0,1, 0,1,1,1, 0,1,0,1,  # 0x7D75
    1,1,0,1, 1,1,1,1, 0,1,0,1, 1,1,0,1,  # 0xDF5D
], dtype=np.uint8)

# D-STAR Frame Sync: 0x552D (JARL D-STAR specification)
DSTAR_FRAME_SYNC = np.array([
    0,1,0,1, 0,1,0,1, 0,0,1,0, 1,1,0,1,  # 0x552D
    0,1,0,1, 0,1,0,1,                      # padding to 24 bits
], dtype=np.uint8)

# D-STAR data scrambling polynomial: x^9 + x^5 + 1 (LFSR seed 0x1FF)
DSTAR_SCRAMBLE_POLY = 0x021

# YSF Frame Sync (20 dibits) — Yaesu System Fusion specification
YSF_SYNC = np.array([
    3,1,1,0, 1,3,0,1, 3,0,2,1, 1,2,0,3, 1,0,3,1,
], dtype=np.uint8)

# P25 Frame Sync: 0x5575F5FF77FF (TIA-102.BAAA Section 7.1, 24 dibits)
P25_FRAME_SYNC = np.array([
    1,1, 1,1, 1,3, 1,1, 3,3, 1,1,  # 0x5575F5
    3,3, 3,3, 1,3, 1,3, 3,3, 3,3,  # 0xFF77FF
], dtype=np.uint8)

# P25 LDU1 NID status symbols
P25_LDU1_NID = np.array([
    1,1,0,1, 0,1,1,1, 1,1,0,1, 0,0,1,0,
], dtype=np.uint8)

# NXDN Frame Sync Word (FSW): RDCH and TDCH variants
NXDN_FRAME_SYNC_RDCH = np.array([
    3,0, 3,1, 3,0, 3,0, 0,0,
], dtype=np.uint8)

NXDN_FRAME_SYNC_TDCH = np.array([
    0,3, 0,2, 0,3, 0,3, 3,3,
], dtype=np.uint8)
