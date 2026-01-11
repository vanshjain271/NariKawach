import express from "express";
import { supabase } from "../supabaseClient.js";
import { broadcastLocation } from "../websocket.js";

const router = express.Router();

router.post("/update", async (req, res) => {
  const { user_id, lat, lng } = req.body;

  const { error } = await supabase.from("location_logs").insert([
    { user_id, lat, lng }
  ]);

  if (error) return res.status(400).json(error);

  // send live update to guardians
  broadcastLocation({ user_id, lat, lng });

  res.json({ status: "Location updated" });
});

export default router;
