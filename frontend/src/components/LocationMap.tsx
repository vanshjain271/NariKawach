import { useEffect, useState, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { MapPin, Navigation, AlertCircle } from "lucide-react";
import { calculateRisk } from "../services/riskService";

// Fix Leaflet default marker icon issue (Vite + React)
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon-2x.png",
  iconUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-icon.png",
  shadowUrl:
    "https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/images/marker-shadow.png",
});

interface LocationMapProps {
  isTracking: boolean;
  onLocationUpdate?: (lat: number, lng: number) => void;
  userId?: string;
  showMapAlways?: boolean;
}

interface LocationState {
  lat: number;
  lng: number;
  accuracy: number;
}

const LocationMap = ({ isTracking, onLocationUpdate, userId }: LocationMapProps) => {
  const [location, setLocation] = useState<LocationState | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [permissionDenied, setPermissionDenied] = useState(false);
  const [risk, setRisk] = useState<any>(null);

  const mapRef = useRef<L.Map | null>(null);
  const mapContainerRef = useRef<HTMLDivElement | null>(null);
  const markerRef = useRef<L.Marker | null>(null);
  const circleRef = useRef<L.Circle | null>(null);
  const watchIdRef = useRef<number | null>(null);

  /* -------------------------------
     Initialize Map (only once)
  -------------------------------- */
  useEffect(() => {
    if (!mapContainerRef.current || mapRef.current) return;

    const map = L.map(mapContainerRef.current, {
      center: [20.5937, 78.9629],
      zoom: 16,
      zoomControl: false,
    });

    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap contributors",
    }).addTo(map);

    mapRef.current = map;

    setTimeout(() => {
      map.invalidateSize();
    }, 0);

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [isTracking]);

  /* -------------------------------
     Update marker & accuracy circle
  -------------------------------- */
  useEffect(() => {
    if (!mapRef.current || !location) return;

    const { lat, lng, accuracy } = location;

    if (!markerRef.current) {
      const customIcon = L.divIcon({
        className: "custom-marker",
        html: `<div style="
          width: 22px;
          height: 22px;
          background: hsl(var(--primary));
          border-radius: 50%;
          border: 3px solid white;
          box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        "></div>`,
        iconSize: [22, 22],
        iconAnchor: [11, 11],
      });

      markerRef.current = L.marker([lat, lng], { icon: customIcon }).addTo(
        mapRef.current
      );
    } else {
      markerRef.current.setLatLng([lat, lng]);
    }

    if (!circleRef.current) {
      circleRef.current = L.circle([lat, lng], {
        radius: accuracy,
        color: "hsl(var(--primary))",
        fillColor: "hsl(var(--primary))",
        fillOpacity: 0.1,
        weight: 2,
      }).addTo(mapRef.current);
    } else {
      circleRef.current.setLatLng([lat, lng]);
      circleRef.current.setRadius(accuracy);
    }

    mapRef.current.setView([lat, lng], mapRef.current.getZoom());
  }, [location]);

  /* -------------------------------
     Geolocation Tracking + AI Call
  -------------------------------- */
  useEffect(() => {
    if (!isTracking) {
      if (watchIdRef.current !== null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
        watchIdRef.current = null;
      }
      return;
    }

    if (!navigator.geolocation) {
      setError("Geolocation not supported");
      return;
    }

    watchIdRef.current = navigator.geolocation.watchPosition(
      async (pos) => {
        const newLocation = {
          lat: pos.coords.latitude,
          lng: pos.coords.longitude,
          accuracy: pos.coords.accuracy,
        };

        setLocation(newLocation);
        setError(null);
        onLocationUpdate?.(newLocation.lat, newLocation.lng);

        try {
          const payload = {
            user_id: userId,
            lat: newLocation.lat,
            lng: newLocation.lng,
            crimeDensity: Math.random() * 10,
            crowdDensity: Math.random() * 10,
            lightingScore: 10 - Math.random() * 5,
            hour: new Date().getHours(),
          };

          const riskResponse = await calculateRisk(payload);
          setRisk(riskResponse);
          console.log("AI RISK RESPONSE:", riskResponse);
        } catch (err) {
          console.error("Risk service unavailable", err);
        }
      },
      (err) => {
        if (err.code === err.PERMISSION_DENIED) {
          setPermissionDenied(true);
          setError("Permission denied");
        } else {
          setError("Unable to fetch location");
        }
      },
      {
        enableHighAccuracy: true,
        timeout: 10000,
        maximumAge: 5000,
      }
    );

    return () => {
      if (watchIdRef.current !== null) {
        navigator.geolocation.clearWatch(watchIdRef.current);
      }
    };
  }, [isTracking, onLocationUpdate]);

  /* -------------------------------
     React to HIGH risk
  -------------------------------- */
  useEffect(() => {
    if (risk?.risk_level?.toLowerCase() === "high") {
      alert("⚠️ High Risk Area Detected. Stay Alert!");
    }
  }, [risk]);

  /* -------------------------------
     UI States
  -------------------------------- */

  if (permissionDenied) {
    return (
      <div className="aspect-video bg-destructive/10 rounded-2xl border flex items-center justify-center">
        <div className="text-center px-4">
          <AlertCircle className="w-8 h-8 text-destructive mx-auto mb-2" />
          <p className="text-sm font-medium">
            Location permission required
          </p>
          <p className="text-xs text-muted-foreground">
            Enable location access in browser settings
          </p>
        </div>
      </div>
    );
  }

  if (error && !location) {
    return (
      <div className="aspect-video bg-muted/40 rounded-2xl border flex items-center justify-center">
        <div className="text-center">
          <Navigation className="w-8 h-8 text-primary mx-auto mb-2 animate-pulse" />
          <p className="text-sm text-muted-foreground">
            Acquiring location…
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="aspect-video rounded-2xl overflow-hidden border shadow-soft relative">
      {risk && (
        <div
          className={`absolute top-3 left-3 z-10 px-3 py-1 rounded-full text-xs font-medium ${
            risk.risk_level?.toLowerCase() === "high"
              ? "bg-red-600 text-white"
              : risk.risk_level?.toLowerCase() === "medium"
                ? "bg-yellow-500 text-black"
                : "bg-green-600 text-white"
          }`}
        >
          Risk: {risk.risk_level}
        </div>
      )}

      <div
        ref={mapContainerRef}
        style={{ height: "100%", width: "100%" }}
      />
    </div>
  );
};

export default LocationMap;
