import axios from "axios";

const ML_URL = process.env.ML_SERVICE_URL;
const ML_API_KEY = process.env.ML_API_KEY;

if (!ML_URL || !ML_API_KEY) {
  throw new Error("❌ ML service environment variables not configured");
}

export async function calculateRisk(payload: {
  user_id: string;
  crimeDensity: number;
  crowdDensity: number;
  lightingScore: number;
  hour: number;
}) {
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

  const response = await axios.post(ML_URL, mlPayload, {
    headers: {
      "Content-Type": "application/json",
      "x-api-key": ML_API_KEY
    }
  });

  return response.data;
}
