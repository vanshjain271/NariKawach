export async function sendMockAlert(guardian: any, user_id: string) {
  console.log("EMERGENCY ALERT");
  console.log(`User ID: ${user_id}`);
  console.log(`Guardian: ${guardian.name}`);
  console.log(`Phone: ${guardian.phone}`);
}

export function sendGuardianAlert(guardian, trip) {
  console.log("ALERT SENT");
  console.log(`Guardian Name: ${guardian.name}`);
  console.log(`Guardian Phone: ${guardian.phone}`);
  console.log(`Trip ID: ${trip.id}`);
  console.log("Live tracking link sent (mock)");
}
