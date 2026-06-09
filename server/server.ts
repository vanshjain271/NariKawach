import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import http from "http";

import locationRoutes from "./routes/location.routes.js";
import riskRoutes from "./routes/risk.routes.js";
import guardianRoutes from "./routes/guardian.routes.js";
import alertRoutes from "./routes/alert.routes.js";
import tripRoutes from "./routes/trip.routes.js";
import { initWebSocket } from "./websocket.js";

dotenv.config();

const app = express();

const allowedOrigins = [
  "https://nari-kawach.vercel.app",
  "https://nari-kawach-zeta.vercel.app",
  "http://localhost:5173",
  "http://localhost:3000"
];

app.use(cors({
  origin: (origin, callback) => {
    if (!origin) return callback(null, true);
    if (allowedOrigins.includes(origin) || origin.endsWith(".vercel.app")) {
      return callback(null, true);
    }
    return callback(new Error("Not allowed by CORS"), false);
  },
  credentials: true
}));

app.use(express.json());

app.get("/", (req, res) => {
  res.json({
    status: "Nari Kawach Backend Running",
    message: "Server is healthy"
  });
});

app.use("/location", locationRoutes);
app.use("/risk", riskRoutes);
app.use("/guardian", guardianRoutes);
app.use("/alert", alertRoutes);
app.use("/trip", tripRoutes);

const server = http.createServer(app);
initWebSocket(server);

const PORT = process.env.PORT || 5001;
server.listen(PORT, () => {
  console.log(`Nari Kawach backend running on port ${PORT}`);
});
