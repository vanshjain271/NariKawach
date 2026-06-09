import { useState, useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import {
  Shield,
  Play,
  Square,
  AlertTriangle,
  CheckCircle,
  Eye,
  ShieldAlert,
  MapPin,
  Users,
  Phone,
  Bell
} from "lucide-react";

import { Button } from "@/components/ui/button";
import { useToast } from "@/hooks/use-toast";
import BottomNav from "@/components/BottomNav";
import ConfirmDialog from "@/components/ConfirmDialog";
import LocationMap from "@/components/LocationMap";
import DemoModePanel from "@/components/DemoModePanel";
import Logo from "@/components/Logo";
import FeatureCard from "@/components/FeatureCard";
import heroBanner from "@/assets/hero-banner.png";

import { api } from "@/lib/api";

type RiskLevel = "low" | "medium" | "high";
type SafetyStatus = "safe" | "monitoring" | "emergency";

const Home = () => {
  const [user, setUser] = useState<any>(null);
  const [isTripActive, setIsTripActive] = useState(false);
  const [currentTripId, setCurrentTripId] = useState<string | null>(null);
  const [riskLevel, setRiskLevel] = useState<RiskLevel>("low");
  const [safetyStatus, setSafetyStatus] = useState<SafetyStatus>("safe");
  const [loading, setLoading] = useState(true);

  const [endTripDialogOpen, setEndTripDialogOpen] = useState(false);
  const [panicDialogOpen, setPanicDialogOpen] = useState(false);
  const [triggeringPanic, setTriggeringPanic] = useState(false);

  const [currentLat, setCurrentLat] = useState<number | null>(null);
  const [currentLng, setCurrentLng] = useState<number | null>(null);

  const [searchParams] = useSearchParams();
  const isDemoMode = searchParams.get("demo") === "true";

  const navigate = useNavigate();
  const { toast } = useToast();

  // Load user
  useEffect(() => {
    const storedUser = JSON.parse(localStorage.getItem("nk_user") || "null");
    if (!storedUser) {
      navigate("/auth");
    } else {
      setUser(storedUser);
      checkActiveTrip(storedUser.id);
      setLoading(false);
    }
  }, [navigate]);

  // GPS
  useEffect(() => {
    if (!navigator.geolocation) return;
    navigator.geolocation.getCurrentPosition(
      (pos) => {
        setCurrentLat(pos.coords.latitude);
        setCurrentLng(pos.coords.longitude);
      },
      () => console.error("Location error"),
      { enableHighAccuracy: true }
    );
  }, []);

  const checkActiveTrip = async (userId: string) => {
    try {
      const { data: trips } = await api.get(`/trip/history/${userId}`);
      const active = trips.find((t: any) => t.status === "active");
      if (active) {
        setIsTripActive(true);
        setCurrentTripId(active.id);
        setSafetyStatus("monitoring");
      }
    } catch {
      console.error("Failed to fetch trip history");
    }
  };

  const startTrip = async () => {
    if (!currentLat || !currentLng) {
      toast({
        title: "Location not ready",
        description: "Please wait for GPS.",
        variant: "destructive"
      });
      return;
    }

    try {
      const { data } = await api.post("/trip/start", {
        user_id: user.id,
        start_location: { lat: currentLat, lng: currentLng }
      });

      setCurrentTripId(data.id);
      setIsTripActive(true);
      setSafetyStatus("monitoring");

      // Check if the user already has at least one guardian
      const { data: guardians } = await api.get(`/guardian/${user.id}`);
      if (guardians && guardians.length > 0) {
        toast({
          title: "Trip Monitoring Started",
          description: "NariKawach is keeping you safe.",
        });
      } else {
        navigate("/onboarding");
      }
    } catch {
      toast({
        title: "Trip Error",
        description: "Could not start trip.",
        variant: "destructive"
      });
    }
  };

  const endTrip = async () => {
    try {
      await api.post("/trip/end", { trip_id: currentTripId });

      setIsTripActive(false);
      setCurrentTripId(null);
      setSafetyStatus("safe");
      setEndTripDialogOpen(false);

      toast({ title: "Trip Ended" });
    } catch {
      console.error("End trip failed");
    }
  };

  const triggerPanic = async () => {
    setTriggeringPanic(true);
    try {
      if (!isTripActive) {
        const { data: trip } = await api.post("/trip/start", { user_id: user.id });
        setCurrentTripId(trip.id);
        setIsTripActive(true);
      }
      setRiskLevel("high");
      setSafetyStatus("emergency");
      navigate("/emergency");
    } catch {
      toast({
        title: "Emergency failed",
        description: "Please try again",
        variant: "destructive"
      });
    } finally {
      setTriggeringPanic(false);
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <Logo size="lg" />
      </div>
    );
  }

  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b sticky top-0 bg-background z-50">
        <div className="container px-4 py-3 flex justify-between">
          <Logo size="sm" />
          <span className="text-xs px-3 py-1 rounded-full bg-safe text-safe-foreground">
            Low Risk
          </span>
        </div>
      </header>

      <main className="flex-1 pb-28 px-4">
        {!isTripActive && (
          <img src={heroBanner} className="rounded-2xl my-4" />
        )}

        <LocationMap
          isTracking={isTripActive}
          userId={user.id}
          showMapAlways
        />

        <Button
          size="lg"
          className="w-full mt-6"
          onClick={() => (isTripActive ? setEndTripDialogOpen(true) : startTrip())}
        >
          {isTripActive ? <Square /> : <Play />}
          {isTripActive ? "End Trip" : "Start Protected Trip"}
        </Button>

        <Button
          variant="outline"
          className="w-full mt-4 border-red-500 text-red-600"
          onClick={() => setPanicDialogOpen(true)}
        >
          <ShieldAlert className="mr-2" /> Emergency SOS
        </Button>
      </main>

      {isDemoMode && (
        <DemoModePanel
          currentRisk={riskLevel}
          isTripActive={isTripActive}
          onRiskChange={setRiskLevel}
        />
      )}

      <BottomNav />

      <ConfirmDialog
        open={endTripDialogOpen}
        onOpenChange={setEndTripDialogOpen}
        title="End Trip?"
        confirmText="End Trip"
        onConfirm={endTrip}
      />

      <ConfirmDialog
        open={panicDialogOpen}
        onOpenChange={setPanicDialogOpen}
        title="Trigger Emergency?"
        confirmText={triggeringPanic ? "Triggering..." : "Trigger Emergency"}
        onConfirm={triggerPanic}
        variant="destructive"
      />
    </div>
  );
};

export default Home;
