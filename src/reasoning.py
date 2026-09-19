"""
Multi-metric reasoning engine: processes structured environmental inputs,
identifies missing information, and builds context for Gemini.
"""

from dataclasses import dataclass, field


@dataclass
class EnvironmentalInput:
    """Structured environmental data from the user."""
    soil_organic_carbon: float | None = None  # percent
    soil_ph: float | None = None
    soil_moisture: float | None = None  # percent
    rainfall: float | None = None  # mm/year
    temperature: float | None = None  # Celsius
    land_use: str | None = None  # e.g., "rice monoculture", "agroforestry"
    crop_type: str | None = None
    biodiversity_index: float | None = None  # Shannon index or similar
    pollution_level: str | None = None  # qualitative or quantitative
    deforestation_rate: float | None = None  # percent per year
    region: str | None = None
    latitude: float | None = None  # bonus geo-coordinate
    longitude: float | None = None  # bonus geo-coordinate
    additional_notes: str | None = None


# Thresholds for flagging concern levels
THRESHOLDS = {
    "soil_organic_carbon": {"critical": 0.5, "low": 1.0, "moderate": 2.0, "good": 3.0},
    "soil_ph": {"strongly_acidic": 4.5, "acidic": 5.5, "optimal_low": 6.0, "optimal_high": 7.0, "alkaline": 8.5},
    "soil_moisture": {"wilting": 15, "stress": 30},
    "temperature": {"frost": 0, "cold_stress": 5, "heat_stress": 35, "severe_heat": 40},
    "rainfall": {"arid": 250, "semi_arid": 500, "sub_humid": 1000},
    "biodiversity_index": {"low": 1.0, "moderate": 2.0, "high": 3.0},
}


def parse_environmental_input(data: dict) -> EnvironmentalInput:
    """Parse a dict of environmental parameters into a structured input."""
    return EnvironmentalInput(
        soil_organic_carbon=_to_float(data.get("soil_organic_carbon")),
        soil_ph=_to_float(data.get("soil_ph")),
        soil_moisture=_to_float(data.get("soil_moisture")),
        rainfall=_to_float(data.get("rainfall")),
        temperature=_to_float(data.get("temperature")),
        land_use=data.get("land_use"),
        crop_type=data.get("crop_type"),
        biodiversity_index=_to_float(data.get("biodiversity_index")),
        pollution_level=data.get("pollution_level"),
        deforestation_rate=_to_float(data.get("deforestation_rate")),
        region=data.get("region"),
        latitude=_to_float(data.get("latitude")),
        longitude=_to_float(data.get("longitude")),
        additional_notes=data.get("additional_notes"),
    )


def _to_float(val) -> float | None:
    """Safely convert a value to float."""
    if val is None or val == "":
        return None
    try:
        if isinstance(val, str):
            val = val.rstrip('.').strip()
        return float(val)
    except (ValueError, TypeError):
        return None


def assess_environmental_conditions(env_input: EnvironmentalInput) -> dict:
    """
    Analyze the provided environmental inputs and flag concerns.

    Returns a dict with:
      - provided_variables: list of provided variable names
      - missing_variables: list of missing but useful variable names
      - concerns: list of identified environmental concerns
      - interactions: list of cross-variable interaction notes
      - clarifying_questions: questions to ask the user for missing critical info
    """
    provided = []
    missing = []
    concerns = []
    interactions = []
    clarifying_questions = []

    # Check each variable
    if env_input.soil_organic_carbon is not None:
        provided.append("soil_organic_carbon")
        soc = env_input.soil_organic_carbon
        if soc < THRESHOLDS["soil_organic_carbon"]["critical"]:
            concerns.append(f"CRITICAL: Soil organic carbon ({soc}%) is critically low (<0.5%). Soil is severely degraded.")
        elif soc < THRESHOLDS["soil_organic_carbon"]["low"]:
            concerns.append(f"WARNING: Soil organic carbon ({soc}%) is low (<1%). Soil health is compromised.")
        elif soc < THRESHOLDS["soil_organic_carbon"]["moderate"]:
            concerns.append(f"NOTE: Soil organic carbon ({soc}%) is below moderate levels. Improvement is recommended.")
    else:
        missing.append("soil_organic_carbon")

    if env_input.soil_ph is not None:
        provided.append("soil_ph")
        ph = env_input.soil_ph
        if ph < THRESHOLDS["soil_ph"]["strongly_acidic"]:
            concerns.append(f"CRITICAL: Soil pH ({ph}) is strongly acidic. Aluminum toxicity likely. Most crops will struggle.")
        elif ph < THRESHOLDS["soil_ph"]["acidic"]:
            concerns.append(f"WARNING: Soil pH ({ph}) is acidic. Nutrient availability is reduced.")
        elif ph > THRESHOLDS["soil_ph"]["alkaline"]:
            concerns.append(f"WARNING: Soil pH ({ph}) is highly alkaline. Micronutrient deficiencies likely.")
    else:
        missing.append("soil_ph")

    if env_input.soil_moisture is not None:
        provided.append("soil_moisture")
        sm = env_input.soil_moisture
        if sm < THRESHOLDS["soil_moisture"]["wilting"]:
            concerns.append(f"CRITICAL: Soil moisture ({sm}%) indicates severe water limitation/stress.")
        elif sm < THRESHOLDS["soil_moisture"]["stress"]:
            concerns.append(f"WARNING: Soil moisture ({sm}%) indicates water limitation/stress for most crops.")
    else:
        missing.append("soil_moisture")

    if env_input.rainfall is not None:
        provided.append("rainfall")
        rain = env_input.rainfall
        if rain < THRESHOLDS["rainfall"]["arid"]:
            concerns.append(f"WARNING: Rainfall ({rain} mm/year) is in the arid range. Rainfed agriculture is not viable.")
        elif rain < THRESHOLDS["rainfall"]["semi_arid"]:
            concerns.append(f"NOTE: Rainfall ({rain} mm/year) is semi-arid. Water management is critical.")
    else:
        missing.append("rainfall")

    if env_input.temperature is not None:
        provided.append("temperature")
        temp = env_input.temperature
        if temp > THRESHOLDS["temperature"]["severe_heat"]:
            concerns.append(f"CRITICAL: Temperature ({temp}°C) causes severe heat stress in most crops.")
        elif temp > THRESHOLDS["temperature"]["heat_stress"]:
            concerns.append(f"WARNING: Temperature ({temp}°C) is in the heat stress range for many crops.")
        elif temp < THRESHOLDS["temperature"]["frost"]:
            concerns.append(f"WARNING: Temperature ({temp}°C) poses frost risk.")
    else:
        missing.append("temperature")

    if env_input.land_use is not None:
        provided.append("land_use")
        if "monoculture" in env_input.land_use.lower():
            concerns.append(f"NOTE: Monoculture land use ({env_input.land_use}) is associated with reduced biodiversity and increased pest pressure.")
    else:
        missing.append("land_use")

    if env_input.crop_type is not None:
        provided.append("crop_type")
    else:
        missing.append("crop_type")

    if env_input.biodiversity_index is not None:
        provided.append("biodiversity_index")
        bi = env_input.biodiversity_index
        if bi < THRESHOLDS["biodiversity_index"]["low"]:
            concerns.append(f"WARNING: Biodiversity index ({bi}) is low. Ecosystem resilience is compromised.")
        elif bi < THRESHOLDS["biodiversity_index"]["moderate"]:
            concerns.append(f"NOTE: Biodiversity index ({bi}) is below moderate. Enhancement recommended.")
    else:
        missing.append("biodiversity_index")

    if env_input.pollution_level is not None:
        provided.append("pollution_level")
        if env_input.pollution_level.lower() in ["high", "severe", "critical"]:
            concerns.append(f"WARNING: Pollution level reported as '{env_input.pollution_level}'. Environmental and health risks present.")
    else:
        missing.append("pollution_level")

    if env_input.deforestation_rate is not None:
        provided.append("deforestation_rate")
        if env_input.deforestation_rate > 1.0:
            concerns.append(f"WARNING: Deforestation rate ({env_input.deforestation_rate}%/year) is high. Significant habitat loss occurring.")
    else:
        missing.append("deforestation_rate")

    if env_input.region is not None:
        provided.append("region")
    if env_input.latitude is not None:
        provided.append("latitude")
    if env_input.longitude is not None:
        provided.append("longitude")

    # --- Genuine 3-Variable Compound Interactions ---
    if (env_input.temperature is not None and env_input.temperature >= 30 and
        env_input.rainfall is not None and env_input.rainfall <= 500 and
        env_input.soil_moisture is not None and env_input.soil_moisture <= 30):
        interactions.append(
            "Compound 3-Variable Arid Heat-Drought Stress (Temperature + Rainfall + Soil Moisture): "
            "High temperature accelerates evapotranspiration demand while low rainfall and low soil moisture "
            "exhaust plant-available water. Under this compounding stress, transpiration demand exceeds soil hydraulic "
            "conductivity, making surface residue shading, mulch retention, and drought-tolerant rooting systems essential."
        )

    if (env_input.soil_ph is not None and env_input.soil_ph < 5.5 and
        env_input.soil_organic_carbon is not None and env_input.soil_organic_carbon < 1.0 and
        env_input.soil_moisture is not None and env_input.soil_moisture <= 30):
        interactions.append(
            "Compound 3-Variable Biochemical Limitation (Soil pH + SOC + Moisture): "
            "Strongly acidic conditions induce potential aluminum toxicity and phosphorus fixation, while depleted "
            "SOC eliminates soil water retention and microbial buffering. In desiccated acidic soils, organic buffering "
            "or liming must precede biological nitrogen-fixing interventions to restore root establishment."
        )

    if (env_input.temperature is not None and env_input.temperature >= 30 and
        env_input.soil_organic_carbon is not None and env_input.soil_organic_carbon < 1.0 and
        env_input.rainfall is not None and env_input.rainfall <= 500):
        interactions.append(
            "Compound 3-Variable Carbon Degradation (Temperature + SOC + Rainfall): "
            "Elevated temperature accelerates microbial decomposition of organic carbon (Q10 effect) while semi-arid "
            "rainfall severely restricts vegetative biomass inputs. Without continuous residue cover and compost additions, "
            "soil organic pools face rapid, compounding depletion."
        )

    if (env_input.land_use is not None and "monoculture" in env_input.land_use.lower() and
        env_input.soil_moisture is not None and env_input.soil_moisture <= 30 and
        env_input.rainfall is not None and env_input.rainfall <= 500):
        interactions.append(
            "Compound 3-Variable Monoculture Vulnerability (Land Use + Moisture + Rainfall): "
            "Continuous single-crop farming under low moisture and low rainfall depletes a single rooting horizon "
            "and increases pathogen vulnerability. Transitioning to multi-species rotations with deep-rooting drought-hardy "
            "legumes diversifies water extraction profiles and breaks host-specific disease cycles."
        )

    if (env_input.land_use is not None and "wetland" in env_input.land_use.lower() and
        env_input.pollution_level is not None and env_input.pollution_level.lower() in ["high", "severe", "critical"] and
        env_input.biodiversity_index is not None and env_input.biodiversity_index < 2.0):
        interactions.append(
            "Compound 3-Variable Wetland Eutrophication & Biodiversity Collapse (Wetland + Pollution + Biodiversity): "
            "Excessive nutrient or chemical loading in degraded wetlands overwhelms biological assimilation capacity, "
            "causing hypoxia and severe trophic collapse. Re-establishing hydrological buffer zones and native emergent "
            "macrophytes is urgent to filter contaminants and rebuild aquatic invertebrate populations."
        )

    # --- 2-Variable Pairwise Interactions ---
    if env_input.soil_organic_carbon is not None and env_input.soil_ph is not None:
        if env_input.soil_organic_carbon < 1.0 and (env_input.soil_ph < 5.5 or env_input.soil_ph > 8.0):
            interactions.append(
                "Low SOC combined with extreme pH suggests severely compromised soil biology. "
                "Address pH first, as microbial activity needed for SOC buildup is pH-dependent."
            )

    if env_input.soil_organic_carbon is not None and env_input.temperature is not None:
        if env_input.temperature > 30 and env_input.soil_organic_carbon < 2.0:
            interactions.append(
                "High temperatures accelerate SOC decomposition (Q10 effect). "
                "SOC management is especially critical in warm climates—higher organic input rates are needed."
            )

    if env_input.soil_organic_carbon is not None and env_input.soil_moisture is not None:
        if env_input.soil_organic_carbon < 1.0 and env_input.soil_moisture < 30:
            interactions.append(
                "Low SOC combined with low soil moisture compounds drought vulnerability. "
                "Depleted organic matter reduces water-holding capacity, accelerating plant moisture stress."
            )

    if env_input.rainfall is not None and env_input.soil_moisture is not None:
        if env_input.rainfall > 1000 and env_input.soil_moisture < 30:
            interactions.append(
                "High rainfall but low soil moisture suggests poor water retention or excessive drainage. "
                "Soil structure, compaction, or texture issues likely. Investigate soil physical properties."
            )
        elif env_input.rainfall < 500 and env_input.soil_moisture < 30:
            interactions.append(
                "Low rainfall combined with low soil moisture indicates severe water deficit. "
                "Moisture conservation, mulching, and drought-tolerant crops are critical priorities."
            )

    if env_input.temperature is not None and env_input.soil_moisture is not None:
        if env_input.temperature > 30 and env_input.soil_moisture < 30:
            interactions.append(
                "High temperature combined with low soil moisture induces acute evapotranspiration stress. "
                "Crop canopy shading, organic mulching, and efficient irrigation are vital to mitigate heat-drought compounding."
            )

    if env_input.rainfall is not None and env_input.deforestation_rate is not None:
        if env_input.deforestation_rate > 1.0 and env_input.rainfall is not None:
            interactions.append(
                "Deforestation disrupts local water cycles. High deforestation rates can reduce regional "
                "rainfall through loss of evapotranspiration, creating a feedback loop."
            )

    if env_input.biodiversity_index is not None and env_input.pollution_level is not None:
        if env_input.biodiversity_index < 2.0 and env_input.pollution_level.lower() in ["high", "severe"]:
            interactions.append(
                "Low biodiversity combined with high pollution indicates ecosystem degradation. "
                "Pollution reduction should be prioritized alongside habitat restoration."
            )

    if env_input.land_use is not None and env_input.biodiversity_index is not None:
        if "monoculture" in env_input.land_use.lower() and env_input.biodiversity_index < 2.0:
            interactions.append(
                "Monoculture land use is a likely driver of the low biodiversity observed. "
                "Diversifying the cropping system is the most direct intervention."
            )

    # --- Targeted Clarifying Questions for Incomplete Inputs ---
    if len(provided) == 0:
        clarifying_questions.append(
            "What are your soil organic carbon (%) and soil pH values? (Crucial to determine baseline fertility, nutrient solubility, and whether microbial life can support restoration)."
        )
        clarifying_questions.append(
            "What is your annual rainfall (mm) and current soil moisture level? (Determines whether biological amendments or cover cropping can establish without causing acute moisture deficit)."
        )
        clarifying_questions.append(
            "What is the current land use or cropping history (e.g., wheat monoculture, agroforestry, wetland, pasture)? (Identifies systemic ecological disturbance factors and pest baselines)."
        )
    elif len(provided) < 4:
        if "soil_organic_carbon" not in provided:
            clarifying_questions.append(
                "What is your soil organic carbon (SOC) level? (Essential because SOC dictates moisture retention capacity and biological nutrient cycling rate)."
            )
        if "soil_ph" not in provided:
            clarifying_questions.append(
                "What is your soil pH? (Crucial because pH below 5.5 triggers aluminum toxicity while extreme pH limits phosphorus and micronutrient availability)."
            )
        if "rainfall" not in provided and "soil_moisture" not in provided:
            clarifying_questions.append(
                "What is your annual rainfall (mm) or current soil moisture level (%)? (Determines whether crop and soil management is governed by acute water deficit)."
            )
        if "land_use" not in provided:
            clarifying_questions.append(
                "What is the current land use or cropping system? (e.g., monoculture, mixed cropping, agroforestry, pasture)."
            )

    return {
        "provided_variables": provided,
        "missing_variables": missing,
        "concerns": concerns,
        "interactions": interactions,
        "clarifying_questions": clarifying_questions,
    }


def build_environmental_context(env_input: EnvironmentalInput, assessment: dict) -> str:
    """
    Build a structured context string from environmental inputs and assessment
    to include in the Gemini prompt.
    """
    parts = []

    # Provided data summary
    if assessment["provided_variables"]:
        parts.append("## Provided Environmental Data")
        data_map = {
            "soil_organic_carbon": f"Soil Organic Carbon: {env_input.soil_organic_carbon}%",
            "soil_ph": f"Soil pH: {env_input.soil_ph}",
            "soil_moisture": f"Soil Moisture: {env_input.soil_moisture}%",
            "rainfall": f"Annual Rainfall: {env_input.rainfall} mm/year",
            "temperature": f"Temperature: {env_input.temperature}°C",
            "land_use": f"Land Use: {env_input.land_use}",
            "crop_type": f"Crop Type: {env_input.crop_type}",
            "biodiversity_index": f"Biodiversity Index (Shannon): {env_input.biodiversity_index}",
            "pollution_level": f"Pollution Level: {env_input.pollution_level}",
            "deforestation_rate": f"Deforestation Rate: {env_input.deforestation_rate}%/year",
            "region": f"Region: {env_input.region}",
            "latitude": f"Latitude: {env_input.latitude}",
            "longitude": f"Longitude: {env_input.longitude}",
        }
        for var in assessment["provided_variables"]:
            if var in data_map:
                parts.append(f"- {data_map[var]}")

    # Concerns
    if assessment["concerns"]:
        parts.append("\n## Environmental Concerns Identified")
        for concern in assessment["concerns"]:
            parts.append(f"- {concern}")

    # Cross-variable interactions
    if assessment["interactions"]:
        parts.append("\n## Cross-Variable Interactions")
        for interaction in assessment["interactions"]:
            parts.append(f"- {interaction}")

    # Missing data
    if assessment["missing_variables"] and assessment["provided_variables"]:
        parts.append(f"\n## Data Not Provided: {', '.join(assessment['missing_variables'])}")

    # Additional notes
    if env_input.additional_notes:
        parts.append(f"\n## Additional Notes from User\n{env_input.additional_notes}")

    return "\n".join(parts)


def extract_environmental_data_from_text(text: str) -> dict:
    """
    Opportunistically extract environmental variables from free-form user query text
    when structured inputs are not explicitly provided or only partially provided.
    """
    import re
    extracted = {}
    if not text:
        return extracted

    # Soil Organic Carbon (%)
    soc_m = re.search(r'(?:soil\s+organic\s+carbon|soc)\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%', text, re.I)
    if soc_m:
        extracted["soil_organic_carbon"] = float(soc_m.group(1).rstrip('.'))

    # Soil pH
    ph_m = re.search(r'(?:soil\s+)?ph\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)', text, re.I)
    if ph_m:
        extracted["soil_ph"] = float(ph_m.group(1).rstrip('.'))

    # Soil moisture (% field capacity)
    moist_m = re.search(r'(?:soil\s+)?moisture\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%', text, re.I)
    if moist_m:
        extracted["soil_moisture"] = float(moist_m.group(1).rstrip('.'))

    # Annual rainfall (mm)
    rain_m = re.search(r'(?:annual\s+)?rainfall\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*mm', text, re.I)
    if rain_m:
        extracted["rainfall"] = float(rain_m.group(1).rstrip('.'))

    # Temperature (°C)
    temp_m = re.search(r'temperature\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*(?:°|deg|degrees)?\s*c', text, re.I)
    if temp_m:
        extracted["temperature"] = float(temp_m.group(1).rstrip('.'))

    # Deforestation rate (%/year)
    defor_m = re.search(r'deforestation\s*(?:rate)?\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)\s*%', text, re.I)
    if defor_m:
        extracted["deforestation_rate"] = float(defor_m.group(1).rstrip('.'))

    # Biodiversity index
    bio_m = re.search(r'biodiversity\s*(?:index)?\s*(?:is|=|:)?\s*([0-9]+(?:\.[0-9]+)?)', text, re.I)
    if bio_m:
        extracted["biodiversity_index"] = float(bio_m.group(1).rstrip('.'))

    # Land use detection
    text_lower = text.lower()
    for lu in ["wheat monoculture", "rice monoculture", "monoculture", "agroforestry", "wetland", "mixed farming", "pasture"]:
        if lu in text_lower:
            extracted["land_use"] = lu
            if "wheat" in lu:
                extracted["crop_type"] = "wheat"
            elif "rice" in lu:
                extracted["crop_type"] = "rice"
            break

    # Region detection
    for reg in ["semi-arid", "arid", "sub-humid", "humid", "tropical", "temperate", "mediterranean"]:
        if reg in text_lower:
            extracted["region"] = reg
            break

    # Pollution level detection
    for poll in ["severe", "high", "moderate", "low"]:
        if f"{poll} pollution" in text_lower or f"pollution is {poll}" in text_lower or f"pollution level is {poll}" in text_lower or f"pollution: {poll}" in text_lower:
            extracted["pollution_level"] = poll
            break

    return extracted
