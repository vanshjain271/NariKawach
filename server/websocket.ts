import { WebSocketServer } from "ws";

let wss: WebSocketServer;

export function initWebSocket(server: any) {
  wss = new WebSocketServer({ server });

  wss.on("connection", ws => {
    console.log("Guardian dashboard connected");
  });
}

export function broadcastLocation(location: any) {
  if (!wss) return;

  wss.clients.forEach(client => {
    if (client.readyState === 1) {
      client.send(JSON.stringify(location));
    }
  });
}
