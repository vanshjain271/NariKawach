import express from "express";
import { supabase } from "../supabaseClient.js";

const router = express.Router();

// Start Trip
router.post("/start", async (req, res) => {
  const { user_id, start_location } = req.body;

  const { data, error } = await supabase.from("trips").insert([
    { user_id, start_location, status: "active" }
  ]).select();

  if (error) return res.status(400).json(error);
  res.json(data[0]);
});

// End Trip
router.post("/end", async (req, res) => {
  const { trip_id, end_location } = req.body;

  const { data, error } = await supabase
    .from("trips")
    .update({ status: "completed", end_location })
    .eq("id", trip_id);

  if (error) return res.status(400).json(error);
  res.json({ message: "Trip ended successfully" });
});

// Get Trip History
router.get("/history/:user_id", async (req, res) => {
  const { user_id } = req.params;

  const { data, error } = await supabase
    .from("trips")
    .select("*")
    .eq("user_id", user_id);

  res.json(data);
});

export default router;
