import express from "express";
import { supabase } from "../supabaseClient.js";

const router = express.Router();

/**
 * POST /guardian/add
 */
router.post("/add", async (req, res) => {
  const { user_id, name, phone } = req.body;

  const { data, error } = await supabase.from("guardians").insert([
    { user_id, name, phone }
  ]);

  if (error) return res.status(400).json(error);
  res.json(data);
});

/**
 * GET /guardian/:user_id
 */
router.get("/:user_id", async (req, res) => {
  const { user_id } = req.params;

  const { data, error } = await supabase
    .from("guardians")
    .select("*")
    .eq("user_id", user_id);

  if (error) return res.status(400).json(error);

  res.json(data);
});

export default router;
