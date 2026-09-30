import re
import urllib.parse

CATEGORIES = [
    "បច្ចេកវិទ្យា",     # Technology
    "សង្គម",           # Society
    "នយោបាយ",         # Politics
    "សង្គ្រាម",         # War
    "ពត៌មានក្នុងស្រុក",  # Local
    "អន្តរជាតិ",        # International
    "កីឡា",            # Sports
]

KEYWORDS = {
    "បច្ចេកវិទ្យា": [
        "tech", "software", "hardware", "apple", "google", "microsoft", "cyber", "ai", 
        "internet", "smartphone", "digital", "ev", "data", "cloud", "chip", "semiconductor",
        "បច្ចេកវិទ្យា", "ទូរស័ព្ទ", "កុំព្យូទ័រ", "អ៊ីនធឺណិត", "ស្មាតហ្វូន", "បណ្ដាញសង្គម", 
        "ឌីជីថល", "អេឡិចត្រូនិក", "កម្មវិធី", "អវកាស", "រថយន្តអគ្គិសនី", "បញ្ញាសិប្បនិម្មិត"
    ],
    "សង្គម": [
        "society", "people", "education", "health", "culture", "community", "crime", 
        "accident", "police", "hospital", "school", "traffic", "poverty",
        "សង្គម", "ប្រជាពលរដ្ឋ", "អប់រំ", "សុខាភិបាល", "វប្បធម៌", "សាសនា", "គ្រោះថ្នាក់", 
        "ឧក្រិដ្ឋកម្ម", "ប៉ូលិស", "ឃាតកម្ម", "គ្រោះធម្មជាតិ", "ក្មេងទំនើង", "ចរាចរណ៍", "មន្ទីរពេទ្យ"
    ],
    "នយោបាយ": [
        "politics", "political", "politician", "government", "election", "minister", 
        "parliament", "senate", "law", "diplomacy", "diplomat", "diplomatic", "president", 
        "policy", "sanction", "treaty", "ambassador",
        "នយោបាយ", "រដ្ឋាភិបាល", "ការបោះឆ្នោត", "រដ្ឋមន្ត្រី", "គណបក្ស", "សភា", "ព្រឹទ្ធសភា", 
        "នាយករដ្ឋមន្ត្រី", "ច្បាប់", "សិទ្ធិមនុស្ស", "ការទូត", "ក្រសួង", "នយោបាយការបរទេស"
    ],
    "សង្គ្រាម": [
        "war", "military", "army", "armies", "weapon", "missile", "russia", "ukraine", "gaza", 
        "israel", "conflict", "terror", "terrorist", "terrorism", "troops", "attack", "invasion", "defense",
        "សង្គ្រាម", "កងទ័ព", "អាវុធ", "មីស៊ីល", "រុស្ស៊ី", "អ៊ុយក្រែន", "អ៊ីស្រាអែល", 
        "ហាម៉ាស់", "បាញ់ប្រហារ", "យោធា", "ភេរវកម្ម", "ជម្លោះ", "ការវាយប្រហារ", "ទាហាន"
    ],
    "ពត៌មានក្នុងស្រុក": [
        "cambodia", "phnom penh", "siem reap", "sihanoukville", "national", "khmer", 
        "battambang", "kampot", "kandal", "angkor",
        "ក្នុងស្រុក", "កម្ពុជា", "ភ្នំពេញ", "សៀមរាប", "ព្រះសីហនុ", "ក្រសួង", "សម្តេច", 
        "ជាតិ", "អាជ្ញាធរ", "ខេត្ត", "រាជធានី", "បាត់ដំបង", "កំពត", "កណ្តាល", 
        "ព័ត៌មានក្នុងស្រុក", "ពត៌មានក្នុងស្រុក"
    ],
    "អន្តរជាតិ": [
        "international", "global", "world", "usa", "united states", "china", "europe", "asean", "foreign", 
        "united nations", "beijing", "washington", "nato",
        "អន្តរជាតិ", "ពិភពលោក", "អាមេរិក", "ចិន", "អឺរ៉ុប", "អាស៊ាន", "បរទេស", 
        "អង្គការសហប្រជាជាតិ", "អ.ស.ប", "សហភាពអឺរ៉ុប", "មហាអំណាច"
    ],
    "កីឡា": [
        "sport", "football", "soccer", "basketball", "tennis", "volleyball", "badminton",
        "olympic", "sea games", "asean games", "fifa", "world cup", "champion", "league",
        "athlete", "tournament", "medal", "gold", "silver", "bronze", "stadium", "match",
        "score", "goal", "player", "coach", "team", "win", "defeat", "final", "semifinal",
        "boxing", "swimming", "cycling", "marathon", "athletics", "gym", "wrestling",
        "កីឡា", "បាល់ទាត់", "បាល់បោះ", "វាយកូនបាល់", "ជើងឯក", "ស៊ីហ្គេម", "អូឡាំពិក",
        "មេដាយ", "ពានរង្វាន់", "ការប្រកួត", "គ្រូបង្ហាត់", "អត្តពលិក", "ក្រុម", "ចំណាត់ថ្នាក់", 
        "ជ័យជំនះ", "ការចាញ់", "គោល", "ពហុកីឡាដ្ឋាន", "ប្រកួតជម្រុះ", "វគ្គផ្តាច់ព្រ័ត្រ", "ហ្វ៊ីហ្វា"
    ]
}

HOT_KEYWORDS = [
    'breaking', 'urgent', 'alert', 'exclusive', 'emergency', 'attack', 'blast', 
    'dead', 'killed', 'disaster', 'crash', 'explosion', 'quake',
    'បន្ទាន់', 'ក្តៅគគុក', 'ទាន់ហេតុការណ៍', 'រន្ធត់', 'ផ្ទុះ', 'ស្លាប់', 'គ្រោះមហន្តរាយ', 'បាញ់ប្រហារ'
]

def analyze_article_metadata(km_title: str = "", km_text: str = "", en_title: str = "") -> dict:
    """
    Performs comprehensive smart NLP analysis:
    - Weighted category classification (title 3x weight, body 1x weight)
    - Hot news / Breaking news score
    - Urgency level
    - Dynamic clean hashtags
    """
    # Safe string sanitation (guards against NoneType)
    km_title_clean = (km_title or "").strip()
    km_text_clean = (km_text or "").strip()
    en_title_clean = (en_title or "").strip()

    title_combined = f"{km_title_clean} {en_title_clean}".lower().strip()
    body_combined = km_text_clean.lower()
    full_text = f"{title_combined} {body_combined}".strip()
    
    # 1. Weighted Category Scoring
    category_scores = {}
    for category, keywords in KEYWORDS.items():
        score = 0.0
        for kw in keywords:
            kw_lower = kw.lower().strip()
            # English word boundary match (supports optional plural -s/-es)
            if re.match(r'^[a-z0-9\s]+$', kw_lower):
                pattern = rf"\b{re.escape(kw_lower)}(?:s|es)?\b"
                if title_combined and re.search(pattern, title_combined):
                    score += 3.0  # Title match carries 3x weight
                if body_combined and re.search(pattern, body_combined):
                    score += 1.0
            else:
                # Khmer continuous text match
                if title_combined and kw_lower in title_combined:
                    score += 3.0
                if body_combined and kw_lower in body_combined:
                    score += 1.0
        if score > 0:
            category_scores[category] = score
            
    # Select categories that have a meaningful score (sorted by highest relevance)
    sorted_categories = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    # Take top categories with score >= 2.0 (or top 1 if any score)
    matched_categories = [cat for cat, s in sorted_categories if s >= 2.0]
    if not matched_categories and sorted_categories:
        matched_categories = [sorted_categories[0][0]]
        
    # 2. Hot News / Urgency Analysis
    hot_hits = 0
    for kw in HOT_KEYWORDS:
        kw_lower = kw.lower().strip()
        if re.match(r'^[a-z0-9\s]+$', kw_lower):
            # Avoid substring collisions like 'dead' in 'deadline'
            if re.search(rf"\b{re.escape(kw_lower)}\b", full_text):
                hot_hits += 1
        else:
            if kw_lower in full_text:
                hot_hits += 1
            
    is_hot = hot_hits >= 1
    if hot_hits >= 2:
        urgency = "HIGH"
    elif hot_hits == 1:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"
        
    # 3. Dynamic Hashtags
    if matched_categories:
        hashtags = " ".join([f"#{cat.replace(' ', '_')}" for cat in matched_categories[:3]])
    else:
        hashtags = "#ព័ត៌មានទូទៅ"
        
    return {
        "categories": matched_categories,
        "is_hot": is_hot,
        "urgency": urgency,
        "hashtags": hashtags,
        "category_scores": category_scores
    }

def categorize_article(*args, **kwargs) -> list:
    """
    Backwards-compatible wrapper returning list of categories.
    Supports:
      - categorize_article(km_title, km_text, en_title)
      - categorize_article(km_text, en_title) [Legacy]
      - categorize_article(km_title="...", km_text="...", en_title="...")
    """
    km_title = kwargs.get("km_title", "")
    km_text = kwargs.get("km_text", "")
    en_title = kwargs.get("en_title", "")

    if len(args) == 1:
        km_text = km_text or args[0]
    elif len(args) == 2:
        # Legacy positional call: (km_text, en_title)
        km_text = km_text or args[0]
        en_title = en_title or args[1]
    elif len(args) >= 3:
        km_title = km_title or args[0]
        km_text = km_text or args[1]
        en_title = en_title or args[2]

    result = analyze_article_metadata(km_title=km_title, km_text=km_text, en_title=en_title)
    return result["categories"]

CONFLICT_LOCATIONS = {
    # Ukraine / Russia Conflict
    "Pokrovsk": ["pokrovsk", "ប៉ូក្រូវស្ក៍", "ប៉ូក្រូវ"],
    "Kharkiv": ["kharkiv", "ខារគីវ", "ខាគីវ"],
    "Kursk": ["kursk", "គូស្ក៍", "ឃឺស"],
    "Bakhmut": ["bakhmut", "បាក់មូត"],
    "Avdiivka": ["avdiivka", "អាវឌីវកា"],
    "Zaporizhzhia": ["zaporizhzhia", "ហ្សាផូរីហ្សា"],
    "Kherson": ["kherson", "ខឺសុន"],
    "Donetsk": ["donetsk", "ដូណេតស្ក៍", "ដូណេត"],
    "Luhansk": ["luhansk", "លូហានស្ក៍"],
    "Crimea": ["crimea", "គ្រីមៀ"],
    "Belgorod": ["belgorod", "ប៊ែលហ្គោរ៉ូដ"],
    "Odesa": ["odesa", "odessa", "អូដេសា"],
    "Kyiv": ["kyiv", "kiev", "គៀវ"],
    "Kupyansk": ["kupyansk", "គូព្យានស្ក៍"],
    "Chasiv Yar": ["chasiv yar", "ឆាស៊ីវយ៉ា"],
    "Toretsk": ["toretsk", "តូរេតស្ក៍"],
    "Vuhledar": ["vuhledar", "វូឡេដា"],
    "Sumy": ["sumy", "ស៊ូមី"],
    "Black Sea": ["black sea", "សមុទ្រខ្មៅ"],
    
    # Middle East Conflict
    "Gaza Strip": ["gaza", "ហ្កាហ្សា", "ហ្គាហ្សា"],
    "Rafah": ["rafah", "រ៉ាហ្វា"],
    "Khan Younis": ["khan younis", "ខាន់យូនីស"],
    "Beirut": ["beirut", "ប៊ែរូត"],
    "Southern Lebanon": ["lebanon", "លីបង់", "hezbollah"],
    "Tel Aviv": ["tel aviv", "តែលអាវីវ"],
    "Jerusalem": ["jerusalem", "ហ្ស៊េរុយសាឡិម"],
    "West Bank": ["west bank", "វេសប៊ែង"],
    "Tehran": ["tehran", "តេអេរ៉ង់", "iran"],
    "Damascus": ["damascus", "syria", "ស៊ីរី", "ដាម៉ាស់"],
    "Red Sea": ["red sea", "សមុទ្រក្រហម", "yemen", "houthi", "យេម៉ែន"],
    
    # Other Conflict Hotspots
    "Taiwan Strait": ["taiwan", "តៃវ៉ាន់", "ច្រកសមុទ្រតៃវ៉ាន់"],
    "South China Sea": ["south china sea", "សមុទ្រចិនខាងត្បូង"],
    "Sudan": ["sudan", "khartoum", "khartum", "ស៊ូដង់"],
    "Myanmar": ["myanmar", "burma", "មីយ៉ាន់ម៉ា"]
}

def get_conflict_map_info(text: str = "", url: str = "") -> dict:
    """
    Extracts conflict geographic location and builds an interactive map link.
    ONLY intended to be called for Category 'សង្គ្រាម'.
    """
    text_lower = (text or "").lower()
    url_lower = (url or "").lower()
    
    # 1. If source already is a map intelligence provider
    if "liveuamap.com" in url_lower:
        return {
            "label": "🗺️ ពិនិត្យលើផែនទីសង្គ្រាម (Liveuamap)",
            "url": url if url.startswith("http") else "https://liveuamap.com"
        }
    if "deepstatemap.live" in url_lower:
        return {
            "label": "🗺️ ពិនិត្យលើផែនទីយុទ្ធសាស្ត្រ (DeepState)",
            "url": url if url.startswith("http") else "https://deepstatemap.live"
        }
        
    # 2. Check for specific conflict city/hotspot
    for location, keywords in CONFLICT_LOCATIONS.items():
        for kw in keywords:
            kw_clean = kw.lower().strip()
            matched = False
            if re.match(r'^[a-z0-9\s]+$', kw_clean):
                if re.search(rf"\b{re.escape(kw_clean)}\b", text_lower):
                    matched = True
            else:
                if kw_clean in text_lower:
                    matched = True
                    
            if matched:
                encoded = urllib.parse.quote(f"{location} war map")
                return {
                    "label": f"🗺️ ពិនិត្យទីតាំងលើផែនទី ({location})",
                    "url": f"https://www.google.com/maps/search/?api=1&query={encoded}"
                }
                    
    # 3. Fallback for war articles without specific city
    if any(k in text_lower for k in ["ukraine", "russia", "អ៊ុយក្រែន", "រុស្ស៊ី", "kyiv", "moscow", "ម៉ូស្គូ"]):
        return {
            "label": "🗺️ ពិនិត្យលើផែនទីសមរភូមិ (Ukraine Conflict Map)",
            "url": "https://liveuamap.com"
        }
    if any(k in text_lower for k in [
        "israel", "gaza", "palestine", "hezbollah", "iran", "lebanon", "yemen", "syria",
        "អ៊ីស្រាអែល", "ហាម៉ាស់", "ប៉ាឡេស្ទីន", "លីបង់", "យេម៉ែន", "ស៊ីរី", "អ៊ីរ៉ង់"
    ]):
        return {
            "label": "🗺️ ពិនិត្យលើផែនទីសមរភូមិ (Middle East Conflict Map)",
            "url": "https://israelpalestine.liveuamap.com"
        }
        
    return {
        "label": "🗺️ ពិនិត្យទីតាំងជម្លោះ (Global Conflict Map)",
        "url": "https://www.cfr.org/global-conflict-tracker"
    }