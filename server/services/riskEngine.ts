export function calculateRisk(data: any) {
  const {
    crimeDensity,
    crowdDensity,
    lightingScore,
    hour
  } = data;

  let score = 0;

  score += crimeDensity * 0.4;
  score += (10 - crowdDensity) * 0.2;
  score += (10 - lightingScore) * 0.2;

  if (hour >= 20 || hour <= 5) score += 2;

  let risk_level = "Low";
  if (score > 6) risk_level = "High";
  else if (score > 3) risk_level = "Medium";

  return { risk_level, score };
}
