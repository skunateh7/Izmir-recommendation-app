# ─────────────────────────────────────────────────────────────────────────────
# data.py — İzmir Tourist Activities Dataset (v2 — AHP/TOPSIS Ready)
#
# Interest categories (Stage 1 - Content-Based Filtering):
#   beach, history, food, nature, family, nightlife  (scored 0–5)
#
# TOPSIS criteria (Stage 3 - TOPSIS Ranking):
#   beach_score, cultural_score, price_score, festival_score,
#   tourist_density, weather_comfort  (scored 1–10, higher = better)
# ─────────────────────────────────────────────────────────────────────────────

ACTIVITIES = [
    # ── HISTORY / CULTURE ─────────────────────────────────────────────────────
    {
        "id": 1,
        "name": "Ephesus Ancient City",
        "category": "history",
        "sub_category": "archaeological site",
        "district": "Selçuk (day trip ~1hr)",
        "description": (
            "UNESCO World Heritage Site and one of the largest Roman archaeological "
            "sites in the world. Features the Library of Celsus, the Temple of "
            "Hadrian, and terraced houses with mosaics."
        ),
        "tags": ["UNESCO", "Roman", "archaeology", "ancient city", "day trip"],
        "best_season": "Spring / Fall",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 5, "food": 0, "nature": 1, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 10, "price_score": 5, "festival_score": 6, "tourist_density": 3, "weather_comfort": 7},
        "source": "TimeOut Turkey, Adventures of Alice",
    },
    {
        "id": 2,
        "name": "İzmir Agora Open Air Museum",
        "category": "history",
        "sub_category": "archaeological site",
        "district": "Konak",
        "description": (
            "Ancient Roman marketplace built in the 2nd century AD. One of the most "
            "well-preserved agoras in the Mediterranean (~40,000 sqm). Includes a "
            "temple of Zeus and a library."
        ),
        "tags": ["Roman", "agora", "open-air museum", "ancient", "Smyrna"],
        "best_season": "Year-round",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 5, "food": 0, "nature": 0, "family": 3, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 9, "price_score": 6, "festival_score": 4, "tourist_density": 6, "weather_comfort": 7},
        "source": "Adventures of Alice, Lonely Planet",
    },
    {
        "id": 3,
        "name": "Kadifekale (Velvet Castle)",
        "category": "history",
        "sub_category": "castle / fortress",
        "district": "Konak",
        "description": (
            "3rd-century BC hilltop fortress offering sweeping panoramic views of "
            "the Gulf of İzmir. Includes Roman cisterns and ancient walls."
        ),
        "tags": ["castle", "fortress", "panoramic view", "ancient", "hilltop"],
        "best_season": "Year-round",
        "price_range": "Free",
        "interest_scores": {"beach": 0, "history": 5, "food": 0, "nature": 2, "family": 3, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 8, "price_score": 10, "festival_score": 3, "tourist_density": 7, "weather_comfort": 6},
        "source": "Thrillophilia, Adventures of Alice",
    },
    {
        "id": 4,
        "name": "Kemeraltı Bazaar",
        "category": "history",
        "sub_category": "bazaar / cultural district",
        "district": "Konak",
        "description": (
            "Historic Ottoman bazaar district with over 3000 years of recorded "
            "history. Includes Kızlarağası Hanı (built 1744) and Havra Sokağı."
        ),
        "tags": ["bazaar", "Ottoman", "market", "culture", "shopping", "synagogue"],
        "best_season": "Year-round",
        "price_range": "Free (shopping extra)",
        "interest_scores": {"beach": 0, "history": 4, "food": 2, "nature": 0, "family": 4, "nightlife": 1},
        "criteria": {"beach_score": 1, "cultural_score": 9, "price_score": 9, "festival_score": 6, "tourist_density": 4, "weather_comfort": 8},
        "source": "Lonely Planet, Goats on the Road",
    },
    {
        "id": 5,
        "name": "Konak Clock Tower",
        "category": "history",
        "sub_category": "landmark",
        "district": "Konak",
        "description": (
            "Iconic clock tower built in 1901, gifted by German Emperor Wilhelm II "
            "to Ottoman Sultan Abdulhamid II. A symbol of İzmir."
        ),
        "tags": ["clock tower", "Ottoman", "landmark", "square", "icon"],
        "best_season": "Year-round",
        "price_range": "Free",
        "interest_scores": {"beach": 0, "history": 4, "food": 0, "nature": 0, "family": 3, "nightlife": 1},
        "criteria": {"beach_score": 1, "cultural_score": 7, "price_score": 10, "festival_score": 5, "tourist_density": 5, "weather_comfort": 7},
        "source": "Goats on the Road, Dominican Abroad",
    },
    {
        "id": 6,
        "name": "İzmir Archaeology & Ethnography Museum",
        "category": "history",
        "sub_category": "museum",
        "district": "Konak",
        "description": (
            "Large museum with artifacts dating back to the Bronze Age, including "
            "the Marble Statue of Androklos and statues from Aegean shipwrecks."
        ),
        "tags": ["museum", "archaeology", "Bronze Age", "artifacts", "indoor"],
        "best_season": "Year-round",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 5, "food": 0, "nature": 0, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 9, "price_score": 7, "festival_score": 5, "tourist_density": 7, "weather_comfort": 10},
        "source": "Goats on the Road",
    },
    {
        "id": 7,
        "name": "Pergamon Acropolis",
        "category": "history",
        "sub_category": "archaeological site",
        "district": "Bergama (day trip ~2hrs)",
        "description": (
            "Ancient acropolis of Pergamon, site of the famous Pergamon Altar. "
            "One of the most important Hellenistic cities of the ancient world."
        ),
        "tags": ["Pergamon", "Hellenistic", "acropolis", "ancient", "day trip"],
        "best_season": "Spring / Fall",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 5, "food": 0, "nature": 2, "family": 3, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 10, "price_score": 5, "festival_score": 5, "tourist_density": 6, "weather_comfort": 6},
        "source": "Thrillophilia, Dominican Abroad",
    },
    {
        "id": 8,
        "name": "Tarihi Asansör (Historical Elevator)",
        "category": "history",
        "sub_category": "landmark / viewpoint",
        "district": "Konak",
        "description": (
            "Historic elevator on Dario Moreno Street offering panoramic views of "
            "İzmir. Named after the famous Sephardic Jewish singer Dario Moreno."
        ),
        "tags": ["elevator", "viewpoint", "panorama", "landmark", "Sephardic"],
        "best_season": "Year-round",
        "price_range": "Small fee",
        "interest_scores": {"beach": 1, "history": 4, "food": 1, "nature": 0, "family": 3, "nightlife": 2},
        "criteria": {"beach_score": 2, "cultural_score": 7, "price_score": 9, "festival_score": 4, "tourist_density": 6, "weather_comfort": 8},
        "source": "Visit Izmir, TripAdvisor",
    },

    # ── BEACH ─────────────────────────────────────────────────────────────────
    {
        "id": 9,
        "name": "Çeşme Beach (İlıca Beach)",
        "category": "beach",
        "sub_category": "sandy beach",
        "district": "Çeşme",
        "description": (
            "Popular blue-flag sandy beach with thermal spring-warmed waters. "
            "Shallow and family-friendly, close to Çeşme town center."
        ),
        "tags": ["sandy beach", "thermal waters", "blue flag", "family", "swimming"],
        "best_season": "May – October",
        "price_range": "Free / Small fee",
        "interest_scores": {"beach": 5, "history": 0, "food": 0, "nature": 3, "family": 5, "nightlife": 0},
        "criteria": {"beach_score": 9, "cultural_score": 2, "price_score": 8, "festival_score": 5, "tourist_density": 4, "weather_comfort": 8},
        "source": "Boutique Small Hotels, Explore Kusadasi",
    },
    {
        "id": 10,
        "name": "Alaçatı Windsurfing Bay",
        "category": "beach",
        "sub_category": "water sports",
        "district": "Çeşme / Alaçatı",
        "description": (
            "World-renowned windsurfing destination with consistent thermal winds "
            "of 15–25 knots. Hosts PWA Windsurfing World Cup annually."
        ),
        "tags": ["windsurfing", "kitesurfing", "PWA World Cup", "water sports", "adventure"],
        "best_season": "May – September",
        "price_range": "Paid (rentals/lessons)",
        "interest_scores": {"beach": 5, "history": 0, "food": 0, "nature": 3, "family": 2, "nightlife": 1},
        "criteria": {"beach_score": 10, "cultural_score": 3, "price_score": 4, "festival_score": 9, "tourist_density": 5, "weather_comfort": 8},
        "source": "LikeCesme, Turkey Travel Planner",
    },
    {
        "id": 11,
        "name": "Altınkum Beach",
        "category": "beach",
        "sub_category": "sandy beach",
        "district": "Çeşme",
        "description": (
            "Quiet cove beach with clear turquoise Aegean water. "
            "Ideal for swimming and snorkeling away from crowds."
        ),
        "tags": ["quiet beach", "cove", "snorkeling", "swimming", "turquoise"],
        "best_season": "May – October",
        "price_range": "Free / Small fee",
        "interest_scores": {"beach": 4, "history": 0, "food": 0, "nature": 4, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 8, "cultural_score": 1, "price_score": 9, "festival_score": 3, "tourist_density": 8, "weather_comfort": 8},
        "source": "Explore Kusadasi, Boutique Small Hotels",
    },
    {
        "id": 12,
        "name": "Kordon Promenade (Kordonboyu)",
        "category": "beach",
        "sub_category": "waterfront promenade",
        "district": "Alsancak",
        "description": (
            "1.5km waterside promenade dating to the 1850s. Popular for walking, "
            "cycling, street food, and sunset watching along the Aegean Sea."
        ),
        "tags": ["promenade", "seafront", "sunset", "cycling", "street food"],
        "best_season": "Year-round",
        "price_range": "Free",
        "interest_scores": {"beach": 4, "history": 1, "food": 2, "nature": 2, "family": 4, "nightlife": 2},
        "criteria": {"beach_score": 7, "cultural_score": 5, "price_score": 10, "festival_score": 6, "tourist_density": 5, "weather_comfort": 8},
        "source": "Goats on the Road, Thrillophilia",
    },
    {
        "id": 13,
        "name": "Boat Trip – Çeşme Peninsula",
        "category": "beach",
        "sub_category": "boat tour",
        "district": "Çeşme",
        "description": (
            "Daily boat tours around the bays of the Çeşme Peninsula with swimming, "
            "snorkeling, and hidden coves in the Aegean Sea."
        ),
        "tags": ["boat tour", "swimming", "snorkeling", "Aegean", "coves"],
        "best_season": "May – October",
        "price_range": "Paid",
        "interest_scores": {"beach": 5, "history": 0, "food": 0, "nature": 4, "family": 5, "nightlife": 0},
        "criteria": {"beach_score": 9, "cultural_score": 2, "price_score": 5, "festival_score": 4, "tourist_density": 7, "weather_comfort": 7},
        "source": "Visit Izmir, LikeCesme",
    },
    {
        "id": 14,
        "name": "Scuba Diving – Karaburun",
        "category": "beach",
        "sub_category": "diving",
        "district": "Karaburun",
        "description": (
            "Scuba diving in pristine Aegean waters with high visibility and rich "
            "marine life. Certified diving center for all experience levels."
        ),
        "tags": ["scuba diving", "snorkeling", "Aegean", "marine life", "adventure"],
        "best_season": "May – October",
        "price_range": "Paid",
        "interest_scores": {"beach": 5, "history": 0, "food": 0, "nature": 5, "family": 2, "nightlife": 0},
        "criteria": {"beach_score": 9, "cultural_score": 1, "price_score": 4, "festival_score": 2, "tourist_density": 9, "weather_comfort": 7},
        "source": "LikeCesme",
    },
    {
        "id": 15,
        "name": "Agamemnon Thermal Baths (Balçova)",
        "category": "beach",
        "sub_category": "thermal spa",
        "district": "Balçova",
        "description": (
            "Historic thermal baths believed used by Greek king Agamemnon. "
            "Natural mineral-rich thermal springs with spa and wellness experiences."
        ),
        "tags": ["thermal baths", "spa", "wellness", "hot springs", "Agamemnon"],
        "best_season": "Year-round",
        "price_range": "Paid",
        "interest_scores": {"beach": 3, "history": 2, "food": 0, "nature": 3, "family": 3, "nightlife": 0},
        "criteria": {"beach_score": 5, "cultural_score": 6, "price_score": 5, "festival_score": 3, "tourist_density": 7, "weather_comfort": 10},
        "source": "Dominican Abroad",
    },

    # ── NATURE ────────────────────────────────────────────────────────────────
    {
        "id": 16,
        "name": "Dilek Peninsula National Park",
        "category": "nature",
        "sub_category": "national park",
        "district": "Kuşadası (day trip ~1.5hrs)",
        "description": (
            "One of Turkey's most pristine national parks with crystal-clear Aegean "
            "coves, dense forest, and rich wildlife including wild horses."
        ),
        "tags": ["national park", "hiking", "wildlife", "coves", "forest", "nature"],
        "best_season": "April – October",
        "price_range": "Paid",
        "interest_scores": {"beach": 4, "history": 0, "food": 0, "nature": 5, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 7, "cultural_score": 2, "price_score": 7, "festival_score": 3, "tourist_density": 8, "weather_comfort": 7},
        "source": "Visit Izmir, Thrillophilia",
    },
    {
        "id": 17,
        "name": "Meles River Walk",
        "category": "nature",
        "sub_category": "river walk / park",
        "district": "Bornova",
        "description": (
            "Scenic river walk along the ancient Meles River — believed to be the "
            "birthplace of Homer. Popular with locals for morning walks and picnics."
        ),
        "tags": ["river", "walk", "Homer", "nature", "park", "local"],
        "best_season": "Spring / Fall",
        "price_range": "Free",
        "interest_scores": {"beach": 0, "history": 2, "food": 0, "nature": 4, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 5, "price_score": 10, "festival_score": 3, "tourist_density": 8, "weather_comfort": 7},
        "source": "Visit Izmir, IzmirTimes",
    },
    {
        "id": 18,
        "name": "Alaçatı Herb Festival & Countryside",
        "category": "nature",
        "sub_category": "festival / countryside",
        "district": "Alaçatı (Çeşme)",
        "description": (
            "Alaçatı's famous annual herb festival celebrates the region's wild herbs "
            "and Aegean cuisine. The surrounding countryside offers cycling routes "
            "through vineyards and olive groves."
        ),
        "tags": ["herbs", "festival", "countryside", "cycling", "Aegean", "vineyard"],
        "best_season": "March – May",
        "price_range": "Free / Low",
        "interest_scores": {"beach": 1, "history": 1, "food": 3, "nature": 5, "family": 4, "nightlife": 1},
        "criteria": {"beach_score": 2, "cultural_score": 6, "price_score": 9, "festival_score": 10, "tourist_density": 6, "weather_comfort": 9},
        "source": "WRO 2024, Explore Kusadasi",
    },

    # ── FOOD ──────────────────────────────────────────────────────────────────
    {
        "id": 19,
        "name": "Boyoz & Turkish Breakfast Tour",
        "category": "food",
        "sub_category": "culinary experience",
        "district": "Konak / Kemeraltı",
        "description": (
            "Try İzmir's signature breakfast pastry boyoz, a Sephardic-origin flaky "
            "savory pastry. Traditional İzmir breakfast with cheeses, olives, "
            "tomatoes, eggs, and tea."
        ),
        "tags": ["boyoz", "Turkish breakfast", "Sephardic", "pastry", "local food"],
        "best_season": "Year-round",
        "price_range": "Low",
        "interest_scores": {"beach": 0, "history": 1, "food": 5, "nature": 0, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 7, "price_score": 9, "festival_score": 5, "tourist_density": 6, "weather_comfort": 9},
        "source": "Culinary Backstreets, Royal Turkish Schools",
    },
    {
        "id": 20,
        "name": "Culinary Backstreets Food Tour",
        "category": "food",
        "sub_category": "guided culinary tour",
        "district": "Konak / Kemeraltı",
        "description": (
            "Professionally guided 6-hour walking food tour. Samples include boyoz, "
            "sübye, pide, kokoreç, lokma, şambali, lentil soup, and Turkish coffee."
        ),
        "tags": ["food tour", "guided tour", "local cuisine", "Aegean food"],
        "best_season": "Year-round",
        "price_range": "Paid (~$125 USD)",
        "interest_scores": {"beach": 0, "history": 2, "food": 5, "nature": 0, "family": 3, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 8, "price_score": 3, "festival_score": 5, "tourist_density": 7, "weather_comfort": 8},
        "source": "Culinary Backstreets, WareOnTheGlobe",
    },
    {
        "id": 21,
        "name": "Seafood Dining – Kordon & Alsancak",
        "category": "food",
        "sub_category": "restaurant experience",
        "district": "Alsancak / Kordon",
        "description": (
            "Fresh Aegean seafood at waterfront restaurants. Grilled sea bass, "
            "calamari, octopus, and stuffed mussels."
        ),
        "tags": ["seafood", "Aegean cuisine", "waterfront dining", "fish"],
        "best_season": "Year-round",
        "price_range": "Medium – High",
        "interest_scores": {"beach": 1, "history": 0, "food": 5, "nature": 0, "family": 3, "nightlife": 1},
        "criteria": {"beach_score": 3, "cultural_score": 5, "price_score": 4, "festival_score": 4, "tourist_density": 5, "weather_comfort": 8},
        "source": "Chasing the Donkey, Wanderlog",
    },
    {
        "id": 22,
        "name": "Tavacı Recep Usta",
        "category": "food",
        "sub_category": "traditional Turkish restaurant",
        "district": "Alsancak",
        "description": (
            "Michelin Guide-listed restaurant famous for traditional Turkish grilled "
            "meats including lamb and beef specialties."
        ),
        "tags": ["Michelin", "Turkish cuisine", "grilled meat", "lamb", "traditional"],
        "best_season": "Year-round",
        "price_range": "Medium",
        "interest_scores": {"beach": 0, "history": 0, "food": 5, "nature": 0, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 6, "price_score": 6, "festival_score": 4, "tourist_density": 5, "weather_comfort": 9},
        "source": "Wanderlog, Chasing the Donkey",
    },
    {
        "id": 23,
        "name": "Meyhane Experience – Rakı & Meze",
        "category": "food",
        "sub_category": "cultural dining",
        "district": "Alsancak / Kordon",
        "description": (
            "Traditional Turkish tavern (meyhane) dining with wide variety of meze, "
            "fresh seafood, and rakı, often with live folk music."
        ),
        "tags": ["meyhane", "meze", "rakı", "traditional", "seafood", "live music"],
        "best_season": "Year-round",
        "price_range": "Medium – High",
        "interest_scores": {"beach": 0, "history": 1, "food": 5, "nature": 0, "family": 2, "nightlife": 3},
        "criteria": {"beach_score": 1, "cultural_score": 8, "price_score": 4, "festival_score": 6, "tourist_density": 5, "weather_comfort": 9},
        "source": "MyHolidays, Regency Holidays",
    },
    {
        "id": 24,
        "name": "Kemeraltı Market Street Food",
        "category": "food",
        "sub_category": "street food",
        "district": "Konak",
        "description": (
            "Street food in the historic bazaar: simit, gevrek, stuffed mussels, "
            "traditional lentil soups at casual lokantas."
        ),
        "tags": ["street food", "simit", "gevrek", "mussels", "budget", "bazaar"],
        "best_season": "Year-round",
        "price_range": "Low",
        "interest_scores": {"beach": 0, "history": 1, "food": 5, "nature": 0, "family": 4, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 7, "price_score": 10, "festival_score": 5, "tourist_density": 4, "weather_comfort": 7},
        "source": "Royal Turkish Schools, Culinary Backstreets",
    },
    {
        "id": 25,
        "name": "Alaçatı Culinary & Wine Scene",
        "category": "food",
        "sub_category": "food destination",
        "district": "Alaçatı (Çeşme)",
        "description": (
            "Alaçatı's cobblestone village is a culinary destination with modern "
            "restaurants, Aegean cuisine, wine tastings, and an annual Herb Festival."
        ),
        "tags": ["Aegean cuisine", "wine", "herbs", "cobblestone", "cafes", "gourmet"],
        "best_season": "Spring – Fall",
        "price_range": "Medium – High",
        "interest_scores": {"beach": 1, "history": 1, "food": 5, "nature": 2, "family": 3, "nightlife": 2},
        "criteria": {"beach_score": 3, "cultural_score": 7, "price_score": 4, "festival_score": 8, "tourist_density": 5, "weather_comfort": 8},
        "source": "WRO 2024, Explore Kusadasi",
    },

    # ── FAMILY ────────────────────────────────────────────────────────────────
    {
        "id": 26,
        "name": "İzmir Wildlife Park (Doğal Yaşam Parkı)",
        "category": "family",
        "sub_category": "wildlife park",
        "district": "Çiğli",
        "description": (
            "One of Turkey's largest wildlife parks with native and exotic animals "
            "in natural habitats. Popular family destination with walking trails, "
            "picnic areas, and educational programs."
        ),
        "tags": ["wildlife", "zoo", "animals", "family", "children", "park"],
        "best_season": "Year-round",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 0, "food": 0, "nature": 4, "family": 5, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 3, "price_score": 6, "festival_score": 5, "tourist_density": 5, "weather_comfort": 7},
        "source": "Visit Izmir, TripAdvisor",
    },
    {
        "id": 27,
        "name": "İzmir Science Centre (Bilim Merkezi)",
        "category": "family",
        "sub_category": "science museum",
        "district": "Konak",
        "description": (
            "Interactive science and technology museum with hands-on exhibits for "
            "all ages. Includes a planetarium and technology workshops."
        ),
        "tags": ["science", "interactive", "planetarium", "children", "museum", "STEM"],
        "best_season": "Year-round",
        "price_range": "Paid",
        "interest_scores": {"beach": 0, "history": 1, "food": 0, "nature": 1, "family": 5, "nightlife": 0},
        "criteria": {"beach_score": 1, "cultural_score": 5, "price_score": 6, "festival_score": 4, "tourist_density": 5, "weather_comfort": 10},
        "source": "Visit Izmir, IzmirTimes",
    },
    {
        "id": 28,
        "name": "Aqua Fantasy Waterpark",
        "category": "family",
        "sub_category": "waterpark",
        "district": "Kuşadası (near İzmir)",
        "description": (
            "One of Turkey's largest waterparks with slides, wave pools, and a hotel "
            "complex. Very popular with families during summer months."
        ),
        "tags": ["waterpark", "slides", "wave pool", "family", "summer", "children"],
        "best_season": "June – September",
        "price_range": "Paid",
        "interest_scores": {"beach": 3, "history": 0, "food": 0, "nature": 0, "family": 5, "nightlife": 0},
        "criteria": {"beach_score": 6, "cultural_score": 1, "price_score": 4, "festival_score": 4, "tourist_density": 3, "weather_comfort": 7},
        "source": "TripAdvisor, Visit Izmir",
    },

    # ── NIGHTLIFE ─────────────────────────────────────────────────────────────
    {
        "id": 29,
        "name": "Bios Bar – Live Music (Alsancak)",
        "category": "nightlife",
        "sub_category": "live music bar",
        "district": "Alsancak",
        "description": (
            "One of İzmir's most popular live music venues. Features famous and "
            "up-and-coming Turkish musicians. Open Wed/Fri/Sat, also hosts comedy nights."
        ),
        "tags": ["live music", "bar", "Turkish musicians", "comedy", "nightlife"],
        "best_season": "Year-round",
        "price_range": "Medium",
        "interest_scores": {"beach": 0, "history": 0, "food": 1, "nature": 0, "family": 0, "nightlife": 5},
        "criteria": {"beach_score": 1, "cultural_score": 6, "price_score": 6, "festival_score": 7, "tourist_density": 5, "weather_comfort": 9},
        "source": "Wanderlog, Arrival Guides",
    },
    {
        "id": 30,
        "name": "Gazi Kadınlar Sokağı (Bar Street)",
        "category": "nightlife",
        "sub_category": "bar-hopping street",
        "district": "Alsancak",
        "description": (
            "The busiest nightlife street in İzmir. Renovated Ottoman houses turned "
            "bars, seafood restaurants with live folk music, and rock bars."
        ),
        "tags": ["bar street", "Ottoman houses", "bar hopping", "rock bars", "folk music"],
        "best_season": "Year-round",
        "price_range": "Low – Medium",
        "interest_scores": {"beach": 0, "history": 1, "food": 1, "nature": 0, "family": 0, "nightlife": 5},
        "criteria": {"beach_score": 1, "cultural_score": 5, "price_score": 7, "festival_score": 6, "tourist_density": 4, "weather_comfort": 8},
        "source": "Holidify",
    },
    {
        "id": 31,
        "name": "Kordon Waterfront Bars",
        "category": "nightlife",
        "sub_category": "outdoor bar / lounge",
        "district": "Alsancak",
        "description": (
            "Seafront bars and open-air cafes along the Kordon promenade with Aegean "
            "Sea views. Relaxed and scenic nightlife experience."
        ),
        "tags": ["waterfront", "outdoor bar", "sea view", "cocktails", "sunset"],
        "best_season": "Year-round",
        "price_range": "Low – Medium",
        "interest_scores": {"beach": 2, "history": 0, "food": 1, "nature": 1, "family": 1, "nightlife": 4},
        "criteria": {"beach_score": 5, "cultural_score": 3, "price_score": 7, "festival_score": 5, "tourist_density": 5, "weather_comfort": 7},
        "source": "IzmirTimes, Aegean Locations",
    },
    {
        "id": 32,
        "name": "Traditional Meyhane Night with Live Music",
        "category": "nightlife",
        "sub_category": "cultural nightlife",
        "district": "Alsancak / Konak",
        "description": (
            "Evenings at a traditional meyhane with live türkü (Turkish folk music), "
            "rakı, and meze. A culturally authentic İzmir nightlife experience."
        ),
        "tags": ["meyhane", "türkü", "folk music", "rakı", "cultural", "authentic nightlife"],
        "best_season": "Year-round",
        "price_range": "Medium",
        "interest_scores": {"beach": 0, "history": 1, "food": 3, "nature": 0, "family": 1, "nightlife": 5},
        "criteria": {"beach_score": 1, "cultural_score": 9, "price_score": 5, "festival_score": 6, "tourist_density": 6, "weather_comfort": 9},
        "source": "Regency Holidays, IzmirTimes",
    },
    {
        "id": 33,
        "name": "Çeşme Summer Beach Clubs",
        "category": "nightlife",
        "sub_category": "beach club (seasonal)",
        "district": "Çeşme",
        "description": (
            "Summer beach clubs with cocktails, dancing, DJ performances, and Aegean "
            "Sea views. Entry typically includes one drink."
        ),
        "tags": ["beach club", "DJ", "cocktails", "summer", "Aegean", "dancing"],
        "best_season": "June – September",
        "price_range": "Medium – High",
        "interest_scores": {"beach": 3, "history": 0, "food": 1, "nature": 1, "family": 0, "nightlife": 5},
        "criteria": {"beach_score": 8, "cultural_score": 2, "price_score": 4, "festival_score": 7, "tourist_density": 4, "weather_comfort": 7},
        "source": "Aegean Locations, Boutique Small Hotels",
    },
    {
        "id": 34,
        "name": "Ooze Venue – Live Concerts",
        "category": "nightlife",
        "sub_category": "music venue / club",
        "district": "Central İzmir (near Bornova Metro)",
        "description": (
            "Popular concert venue hosting renowned Turkish musicians and DJs. "
            "Favorite among university students. Accessible by metro."
        ),
        "tags": ["concert venue", "live music", "DJ", "students", "metro"],
        "best_season": "Year-round",
        "price_range": "Medium",
        "interest_scores": {"beach": 0, "history": 0, "food": 0, "nature": 0, "family": 0, "nightlife": 5},
        "criteria": {"beach_score": 1, "cultural_score": 5, "price_score": 6, "festival_score": 8, "tourist_density": 5, "weather_comfort": 10},
        "source": "Wanderlog",
    },
    {
        "id": 35,
        "name": "Konak Pier Upscale Bars (Breeze Bar)",
        "category": "nightlife",
        "sub_category": "upscale rooftop bar",
        "district": "Alsancak (Konak Pier)",
        "description": (
            "Chic bars in the historic Konak Pier (Gustave Eiffel's firm). Stunning "
            "sea views, exquisite cocktails. Includes Breeze Bar at Mövenpick Hotel."
        ),
        "tags": ["rooftop bar", "cocktails", "upscale", "sea view", "Eiffel", "Mövenpick"],
        "best_season": "Year-round",
        "price_range": "High",
        "interest_scores": {"beach": 1, "history": 1, "food": 1, "nature": 0, "family": 0, "nightlife": 5},
        "criteria": {"beach_score": 4, "cultural_score": 6, "price_score": 3, "festival_score": 5, "tourist_density": 6, "weather_comfort": 9},
        "source": "Aegean Locations, Holidify",
    },
]

# ── CATEGORY METADATA ─────────────────────────────────────────────────────────
CATEGORY_META = {
    "beach":     {"label": "Beach & Sun",       "emoji": "🏖️"},
    "history":   {"label": "History & Culture", "emoji": "🏛️"},
    "food":      {"label": "Food & Cuisine",    "emoji": "🍽️"},
    "nature":    {"label": "Nature",            "emoji": "🌿"},
    "family":    {"label": "Family",            "emoji": "👨‍👩‍👧"},
    "nightlife": {"label": "Nightlife",         "emoji": "🌙"},
}

# ── TOPSIS CRITERIA METADATA ─────────────────────────────────────────────────
# All criteria are BENEFIT type (higher = better for the tourist)
# tourist_density: 10 = very uncrowded (good), 1 = very crowded (bad)
# price_score:     10 = very affordable, 1 = very expensive
CRITERIA_META = {
    "beach_score":    {"label": "Beach Quality",  "type": "benefit"},
    "cultural_score": {"label": "Cultural Value", "type": "benefit"},
    "price_score":    {"label": "Affordability",  "type": "benefit"},
    "festival_score": {"label": "Events/Festivals","type": "benefit"},
    "tourist_density":{"label": "Low Crowding",   "type": "benefit"},
    "weather_comfort":{"label": "Weather Comfort","type": "benefit"},
}

CRITERIA_KEYS = list(CRITERIA_META.keys())
