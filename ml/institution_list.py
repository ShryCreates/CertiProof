"""
CertiProof — Indian University Database
==========================================
50 Indian universities with roll number patterns and accreditation info.
"""

INDIAN_UNIVERSITIES = [
    {"name": "Indian Institute of Technology Bombay",       "short": "IIT Bombay",      "code": "IITB",   "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{2}[A-Z]{1,2}[0-9]{4,6}"},
    {"name": "Indian Institute of Technology Delhi",        "short": "IIT Delhi",       "code": "IITD",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{3}[0-9]{6}[A-Z]"},
    {"name": "Indian Institute of Technology Madras",       "short": "IIT Madras",      "code": "IITM",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z][0-9]{3}"},
    {"name": "Indian Institute of Technology Kanpur",       "short": "IIT Kanpur",      "code": "IITK",   "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{9}"},
    {"name": "Indian Institute of Technology Kharagpur",    "short": "IIT KGP",         "code": "IITKGP", "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{2}[A-Z]{2}[0-9]{5}"},
    {"name": "Indian Institute of Technology Roorkee",      "short": "IIT Roorkee",     "code": "IITR",   "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{8}"},
    {"name": "Indian Institute of Science Bangalore",       "short": "IISc",            "code": "IISC",   "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}"},
    {"name": "National Institute of Technology Trichy",     "short": "NIT Trichy",      "code": "NITT",   "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{6}[A-Z]{2}[0-9]{3}"},
    {"name": "National Institute of Technology Warangal",   "short": "NIT Warangal",    "code": "NITW",   "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}[A-Z]"},
    {"name": "National Institute of Technology Surathkal",  "short": "NIT Surathkal",   "code": "NITK",   "accreditation": "NAAC A+",  "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}"},
    {"name": "Delhi Technological University",              "short": "DTU",             "code": "DTU",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]"},
    {"name": "Jadavpur University",                         "short": "JU",              "code": "JU",     "accreditation": "NAAC A+",  "roll_pattern": r"[A-Z]{3}[0-9]{2}[0-9]{4}"},
    {"name": "University of Delhi",                         "short": "DU",              "code": "DU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{7}"},
    {"name": "Mumbai University",                           "short": "MU",              "code": "MU",     "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Anna University",                             "short": "AU",              "code": "AU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{12}"},
    {"name": "Osmania University",                          "short": "OU",              "code": "OU",     "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Pune University",                             "short": "SPPU",            "code": "SPPU",   "accreditation": "NAAC A",   "roll_pattern": r"[A-Z]{1}[0-9]{10}"},
    {"name": "Calcutta University",                         "short": "CU",              "code": "CU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}"},
    {"name": "Banaras Hindu University",                    "short": "BHU",             "code": "BHU",    "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{4}[A-Z]{2}[0-9]{3}"},
    {"name": "Jawaharlal Nehru University",                 "short": "JNU",             "code": "JNU",    "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{3}[A-Z][0-9]{3}"},
    {"name": "Hyderabad University",                        "short": "UoH",             "code": "UOH",    "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{2}[A-Z]{4}[0-9]{3}"},
    {"name": "Amity University",                            "short": "Amity",           "code": "AMITY",  "accreditation": "NAAC A+",  "roll_pattern": r"A[0-9]{11}"},
    {"name": "VIT University Vellore",                      "short": "VIT",             "code": "VIT",    "accreditation": "NAAC A++", "roll_pattern": r"[0-9]{2}[A-Z]{3}[0-9]{4}"},
    {"name": "SRM Institute of Science and Technology",     "short": "SRMIST",          "code": "SRM",    "accreditation": "NAAC A++", "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}"},
    {"name": "Manipal Academy of Higher Education",         "short": "MAHE",            "code": "MAHE",   "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}"},
    {"name": "Christ University Bangalore",                 "short": "Christ University","code": "CHRIST", "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{7}"},
    {"name": "Symbiosis International University",          "short": "SIU",             "code": "SIU",    "accreditation": "NAAC A",   "roll_pattern": r"SIU[0-9]{8}"},
    {"name": "BITS Pilani",                                 "short": "BITS",            "code": "BITS",   "accreditation": "NAAC A",   "roll_pattern": r"20[0-9]{2}[A-Z]{2}[0-9]{4}[A-Z]"},
    {"name": "Thapar Institute of Engineering",             "short": "Thapar",          "code": "TIET",   "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{9}"},
    {"name": "PSG College of Technology",                   "short": "PSGCT",           "code": "PSGCT",  "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{2}[A-Z][0-9]{4}"},
    {"name": "Indian Institute of Technology Hyderabad",    "short": "IIT Hyderabad",   "code": "IITH",   "accreditation": "NAAC A+",  "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{4}[0-9]{3}"},
    {"name": "Indian Institute of Technology Gandhinagar",  "short": "IIT Gandhinagar", "code": "IITGN",  "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{2}[A-Z]{2}[0-9]{5}"},
    {"name": "National Institute of Technology Calicut",    "short": "NIT Calicut",     "code": "NITC",   "accreditation": "NAAC A+",  "roll_pattern": r"[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}"},
    {"name": "National Institute of Technology Rourkela",   "short": "NIT Rourkela",    "code": "NITR",   "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}"},
    {"name": "Visvesvaraya Technological University",       "short": "VTU",             "code": "VTU",    "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{10}[A-Z]{2}[0-9]{3}"},
    {"name": "Rajasthan Technical University",              "short": "RTU",             "code": "RTU",    "accreditation": "NAAC B+",  "roll_pattern": r"[0-9]{13}"},
    {"name": "Gujarat Technological University",            "short": "GTU",             "code": "GTU",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{12}"},
    {"name": "Kurukshetra University",                      "short": "KUK",             "code": "KUK",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{11}"},
    {"name": "Panjab University",                           "short": "PU",              "code": "PU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{8}"},
    {"name": "Aligarh Muslim University",                   "short": "AMU",             "code": "AMU",    "accreditation": "NAAC A+",  "roll_pattern": r"[A-Z]{2}[0-9]{4}[A-Z]{2}"},
    {"name": "Cochin University of Science and Technology", "short": "CUSAT",           "code": "CUSAT",  "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Kerala University",                           "short": "KU",              "code": "KU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{10}"},
    {"name": "Andhra University",                           "short": "AU Vizag",        "code": "AUV",    "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{12}"},
    {"name": "Sri Venkateswara University",                 "short": "SVU",             "code": "SVU",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Madurai Kamaraj University",                  "short": "MKU",             "code": "MKU",    "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}"},
    {"name": "Bharathiar University",                       "short": "BU",              "code": "BU",     "accreditation": "NAAC A+",  "roll_pattern": r"[0-9]{9}"},
    {"name": "Savitribai Phule Pune University",            "short": "SPPU Pune",       "code": "SPPU2",  "accreditation": "NAAC A",   "roll_pattern": r"[A-Z][0-9]{10}"},
    {"name": "Shivaji University",                          "short": "SUK",             "code": "SUK",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Nagpur University",                           "short": "RTMNU",           "code": "RTMNU",  "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
    {"name": "Dibrugarh University",                        "short": "DU Assam",        "code": "DUA",    "accreditation": "NAAC A",   "roll_pattern": r"[0-9]{10}"},
]

# Quick lookup maps
BY_CODE  = {u["code"]:  u for u in INDIAN_UNIVERSITIES}
BY_SHORT = {u["short"]: u for u in INDIAN_UNIVERSITIES}
BY_NAME  = {u["name"]:  u for u in INDIAN_UNIVERSITIES}

ALL_NAMES   = [u["name"]  for u in INDIAN_UNIVERSITIES]
ALL_SHORTS  = [u["short"] for u in INDIAN_UNIVERSITIES]
ALL_CODES   = [u["code"]  for u in INDIAN_UNIVERSITIES]
