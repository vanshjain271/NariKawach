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
app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
  res.json({
    status: "Nari Kawach Backend Running",
    message: "Server is healthy",
    apis: {
      location: "POST /location/update",
      risk: "POST /risk/calculate",
      guardian: "POST /guardian/add | GET /guardian/:user_id",
      alert: "POST /alert/send"
    }
  });
});

app.use("/location", locationRoutes);
app.use("/risk", riskRoutes);
app.use("/guardian", guardianRoutes);
app.use("/alert", alertRoutes);
app.use("/trip", tripRoutes);

const server = http.createServer(app);
initWebSocket(server);

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => {
  console.log(`Nari Kawach backend running on port ${PORT}`);
});
