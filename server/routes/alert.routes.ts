import express from "express";
import { supabase } from "../supabaseClient.js";
import { sendMockAlert, sendGuardianAlert } from "../services/alertService.js";

const router = express.Router();

export async function triggerAlert(user_id: string) {
  const { data: guardians } = await supabase
    .from("guardians")
    .select("*")
    .eq("user_id", user_id);

  if (!guardians) return;

  for (const guardian of guardians) {
    await sendMockAlert(guardian, user_id);
  }
}

/**
 * POST /alert/send
 */
router.post("/send", async (req, res) => {
  const { user_id, trip_id } = req.body;

  const { data: guardians, error } = await supabase
    .from("guardians")
    .select("*")
    .eq("user_id", user_id);

  if (error) return res.status(400).json(error);
  if (!guardians.length) return res.status(404).json({ message: "No guardians found" });

  guardians.forEach(g => sendGuardianAlert(g, { id: trip_id }));

  res.json({ success: true, sent_to: guardians.length });
});


export default router;
