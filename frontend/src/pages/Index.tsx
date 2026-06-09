import { useEffect } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Shield, Users, MapPin, Bell, ArrowRight, Heart } from "lucide-react";
import { Button } from "@/components/ui/button";

const Index = () => {
  const navigate = useNavigate();

  useEffect(() => {
    const user = localStorage.getItem("nk_user");
    if (user) {
      navigate("/home");
    }
  }, [navigate]);

  return (
    <div className="min-h-screen gradient-hero">
      {/* Navigation */}
      <nav className="container mx-auto px-4 py-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center">
              <Shield className="w-5 h-5 text-primary" />
            </div>
            <span className="text-xl font-semibold text-foreground">NariKawach</span>
          </div>
          <Link to="/auth">
            <Button variant="outline" className="shadow-soft">
              Login
            </Button>
          </Link>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="container mx-auto px-4 pt-16 pb-24">
        <div className="max-w-3xl mx-auto text-center animate-fade-in">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
            <Heart className="w-4 h-4" />
            Your safety matters
          </div>
          <h1 className="text-4xl md:text-5xl lg:text-6xl font-bold text-foreground mb-6 text-balance leading-tight">
            Your Silent Safety Companion
          </h1>
          <p className="text-lg md:text-xl text-muted-foreground mb-10 max-w-2xl mx-auto text-balance">
            NariKawach helps detect potentially unsafe situations early using contextual awareness — without surveillance.
          </p>
          <Link to="/auth">
            <Button size="lg" className="shadow-calm text-lg px-8 py-6 h-auto">
              Get Started
              <ArrowRight className="w-5 h-5 ml-2" />
            </Button>
          </Link>
        </div>
      </section>

      {/* How It Works Section */}
      <section className="container mx-auto px-4 pb-24">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-2xl md:text-3xl font-semibold text-center text-foreground mb-4">
            How It Works
          </h2>
          <p className="text-muted-foreground text-center mb-12 max-w-xl mx-auto">
            Simple, consent-based safety in three easy steps
          </p>

          <div className="grid md:grid-cols-3 gap-6 md:gap-8">
            {/* Step 1 */}
            <div className="relative animate-slide-up" style={{ animationDelay: "0.1s" }}>
              <div className="bg-card rounded-2xl p-6 shadow-soft h-full border border-border/50">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                  <Users className="w-6 h-6 text-primary" />
                </div>
                <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                  1
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  Setup Trusted Contacts
                </h3>
                <p className="text-muted-foreground text-sm">
                  Add your emergency guardians who will be notified when you need help.
                </p>
              </div>
            </div>

            {/* Step 2 */}
            <div className="relative animate-slide-up" style={{ animationDelay: "0.2s" }}>
              <div className="bg-card rounded-2xl p-6 shadow-soft h-full border border-border/50">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                  <MapPin className="w-6 h-6 text-primary" />
                </div>
                <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                  2
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  Enable During Travel
                </h3>
                <p className="text-muted-foreground text-sm">
                  Start safety monitoring when you're traveling or in unfamiliar areas.
                </p>
              </div>
            </div>

            {/* Step 3 */}
            <div className="relative animate-slide-up" style={{ animationDelay: "0.3s" }}>
              <div className="bg-card rounded-2xl p-6 shadow-soft h-full border border-border/50">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4">
                  <Bell className="w-6 h-6 text-primary" />
                </div>
                <div className="absolute -top-3 -left-3 w-8 h-8 rounded-full bg-primary text-primary-foreground flex items-center justify-center text-sm font-semibold">
                  3
                </div>
                <h3 className="text-lg font-semibold text-foreground mb-2">
                  Get Intelligent Assistance
                </h3>
                <p className="text-muted-foreground text-sm">
                  Receive contextual alerts and automatic escalation when needed.
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Trust Section */}
      <section className="container mx-auto px-4 pb-24">
        <div className="max-w-2xl mx-auto text-center gradient-calm rounded-3xl p-8 md:p-12 border border-primary/10">
          <Shield className="w-12 h-12 text-primary mx-auto mb-4" />
          <h2 className="text-2xl font-semibold text-foreground mb-3">
            Privacy-First Design
          </h2>
          <p className="text-muted-foreground mb-6">
            Your location and data are only accessed with your explicit consent.
            We believe safety should never come at the cost of privacy.
          </p>
          <Link to="/auth">
            <Button variant="outline" className="shadow-soft">
              Learn More
              <ArrowRight className="w-4 h-4 ml-2" />
            </Button>
          </Link>
        </div>
      </section>

      {/* Footer */}
      <footer className="container mx-auto px-4 py-8 border-t border-border">
        <div className="flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-2">
            <Shield className="w-5 h-5 text-primary" />
            <span className="font-medium text-foreground">NariKawach</span>
          </div>
          <p className="text-sm text-muted-foreground">
            © {new Date().getFullYear()} <span className="font-bold text-green-600">NariKawach</span>. Your safety, your control.
          </p>
        </div>
      </footer>
    </div>
  );
};

export default Index;
