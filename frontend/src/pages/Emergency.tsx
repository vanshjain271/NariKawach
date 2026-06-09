import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import {
  Shield,
  MapPin,
  Phone,
  CheckCircle,
  User,
  ArrowLeft,
  MessageCircle
} from "lucide-react";

import { Button } from "@/components/ui/button";
import LocationMap from "@/components/LocationMap";
import { api } from "@/lib/api";

interface Guardian {
  id: string;
  name: string;
  phone: string;
}

const Emergency = () => {
  const [guardians, setGuardians] = useState<Guardian[]>([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const user = JSON.parse(localStorage.getItem("nk_user") || "null");
    if (!user) {
      navigate("/auth");
    } else {
      fetchGuardians(user.id);
      triggerBackendAlert(user.id);
    }
  }, [navigate]);

  const triggerBackendAlert = async (userId: string) => {
    try {
      await api.post("/alert/send", { user_id: userId });
    } catch (error) {
      console.error("Failed to notify backend of emergency", error);
    }
  };

  const fetchGuardians = async (userId: string) => {
    try {
      const { data } = await api.get(`/guardian/${userId}`);
      setGuardians(data);
    } catch (error) {
      console.error("Error fetching guardians", error);
    } finally {
      setLoading(false);
    }
  };

  const handleReturnHome = () => {
    navigate("/home");
  };

  if (loading) {
    return (
      <div className="min-h-screen gradient-emergency flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-emergency/30 border-t-emergency rounded-full animate-spin" />
      </div>
    );
  }

  const userId = JSON.parse(localStorage.getItem("nk_user") || "{}")?.id;

  return (
    <div className="min-h-screen gradient-emergency flex flex-col">
      {/* Header */}
      <nav className="container mx-auto px-4 py-6">
        <div className="flex items-center gap-2">
          <div className="w-10 h-10 rounded-xl bg-emergency/10 flex items-center justify-center">
            <Shield className="w-5 h-5 text-emergency" />
          </div>
          <span className="text-xl font-semibold text-foreground">
            NariKawach
          </span>
        </div>
      </nav>

      {/* Main Content */}
      <div className="flex-1 container mx-auto px-4 py-6">
        <div className="max-w-lg mx-auto space-y-6">

          {/* Emergency Banner */}
          <div className="bg-card rounded-2xl p-6 shadow-soft border border-emergency/20">
            <div className="flex items-center gap-3 mb-4">
              <div className="w-12 h-12 rounded-xl bg-emergency/10 flex items-center justify-center">
                <Shield className="w-6 h-6 text-emergency animate-pulse" />
              </div>
              <div>
                <h1 className="text-xl font-semibold">
                  Emergency Mode Active
                </h1>
                <p className="text-sm text-muted-foreground">
                  Safety protocols initiated
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2 px-4 py-3 bg-emergency/5 rounded-xl border border-emergency/10">
              <CheckCircle className="w-5 h-5 text-emergency" />
              <span className="text-sm font-medium">
                SOS Sent — Guardians notified
              </span>
            </div>
          </div>

          {/* Location */}
          <div className="bg-card rounded-2xl overflow-hidden shadow-soft border">
            <div className="p-4 border-b">
              <h2 className="font-medium flex items-center gap-2">
                <MapPin className="w-4 h-4 text-emergency" />
                Live Location
              </h2>
            </div>
            <div className="aspect-[16/9]">
              <LocationMap
                isTracking
                userId={userId}
                showMapAlways
              />
            </div>
          </div>

          {/* Guardians */}
          <div className="bg-card rounded-2xl shadow-soft border">
            <div className="p-4 border-b">
              <h2 className="font-medium">
                Emergency Contacts
              </h2>
            </div>

            <div className="p-4 space-y-3">
              {guardians.length > 0 ? (
                guardians.map((g) => (
                  <div
                    key={g.id}
                    className="flex items-center justify-between p-3 bg-accent/30 rounded-xl"
                  >
                    <div className="flex items-center gap-3">
                      <div className="w-10 h-10 rounded-full bg-primary/10 flex items-center justify-center">
                        <User className="w-5 h-5 text-primary" />
                      </div>
                      <div>
                        <p className="font-medium">{g.name}</p>
                        <p className="text-sm text-muted-foreground">
                          {g.phone}
                        </p>
                      </div>
                    </div>

                    <div className="flex gap-2">
                      <a
                        href={`sms:${g.phone}?body=EMERGENCY! I need help. Track my live location on NariKawach: ${window.location.origin}/home`}
                        className="w-10 h-10 rounded-full bg-primary flex items-center justify-center animate-pulse"
                        title="Send SOS SMS"
                      >
                        <MessageCircle className="w-5 h-5 text-primary-foreground" />
                      </a>
                      <a
                        href={`tel:${g.phone}`}
                        className="w-10 h-10 rounded-full bg-safe flex items-center justify-center"
                        title="Call Guardian"
                      >
                        <Phone className="w-5 h-5 text-safe-foreground" />
                      </a>
                    </div>
                  </div>
                ))
              ) : (
                <p className="text-sm text-muted-foreground text-center py-4">
                  No guardians configured
                </p>
              )}
            </div>
          </div>

          {/* Back */}
          <Button
            onClick={handleReturnHome}
            variant="outline"
            className="w-full h-12"
          >
            <ArrowLeft className="w-5 h-5 mr-2" />
            Return to Dashboard
          </Button>

          <p className="text-xs text-center text-muted-foreground">
            Stay calm. Help is on the way.
          </p>
        </div>
      </div>
    </div>
  );
};

export default Emergency;
