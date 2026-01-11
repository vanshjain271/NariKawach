import express from "express";
import { supabase } from "../supabaseClient.js";
import { calculateRisk } from "../services/riskEngine.js";
import { triggerAlert } from "./alert.routes.js";
import { z } from "zod";

const router = express.Router();

const riskSchema = z.object({
    user_id: z.string().uuid(),
    crimeDensity: z.number().min(0).max(10),
    crowdDensity: z.number().min(0).max(10),
    lightingScore: z.number().min(0).max(10),
    hour: z.number().min(0).max(23)
});

router.post("/calculate", async (req, res) => {
    const validation = riskSchema.safeParse(req.body);

    if (!validation.success) {
        return res.status(400).json({
            error: "Invalid request body",
            details: validation.error.issues
        });
    }

    const { user_id, crimeDensity, crowdDensity, lightingScore, hour } = validation.data;

    const result = calculateRisk({ crimeDensity, crowdDensity, lightingScore, hour });

    await supabase.from("risk_status").upsert({
        user_id,
        risk_level: result.risk_level,
        reason: `Score: ${result.score.toFixed(2)}`
    });

    if (result.risk_level === "High") {
        await triggerAlert(user_id);
    }

    res.json(result);
});

export default router;
