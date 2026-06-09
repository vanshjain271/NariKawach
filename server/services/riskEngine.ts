import axios from "axios";

const ML_URL = process.env.ML_SERVICE_URL;
const ML_API_KEY = process.env.ML_API_KEY;

const ML_ENABLED = Boolean(ML_URL && ML_API_KEY);

type RiskPayload = {
  user_id: string;
  crimeDensity: number;
  crowdDensity: number;
  lightingScore: number;
  hour: number;
};

export async function calculateRisk(payload: RiskPayload) {
  // ✅ Fallback mode (ML not configured)
  if (!ML_ENABLED) {
    const score =
      payload.crimeDensity * 0.4 +
      payload.crowdDensity * 0.3 +
      (10 - payload.lightingScore) * 0.3;

    let risk_level = "low";

    if (score > 7) risk_level = "high";
    else if (score > 4) risk_level = "medium";

    return {
      user_id: payload.user_id,
      risk_level,
      confidence: 0.6,
      reason: "Calculated using fallback rule engine",
      source: "fallback-rule-engine"
    };
  }

  // 🔵 ML mode (future / optional)
  const mlPayload = {
    user_id: payload.user_id,
    current_location: { lat: 0, lng: 0 },
    route: [],
    time: { hour: payload.hour, day: "Monday" },
    crime_density: payload.crimeDensity,
    crowd_density: payload.crowdDensity,
    lighting_score: payload.lightingScore,
    speed: 1.0,
    route_deviation_score: 0.3
  };

  try {
    const response = await axios.post(ML_URL!, mlPayload, {
      headers: {
        "Content-Type": "application/json",
        "x-api-key": ML_API_KEY!
      },
      timeout: 3000
    });

    return {
      ...response.data,
      source: "ml-service"
    };
  } catch (error) {
    // 🟡 ML failed → fallback automatically
    return {
      user_id: payload.user_id,
      risk_level: "medium",
      confidence: 0.5,
      reason: "ML service unavailable, using fallback",
      source: "ml-fallback"
    };
  }
}
