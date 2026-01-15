# NariKawach - Your Silent Safety Companion

- NariKawach is a privacy-first, consent-driven women’s safety application designed to support users during travel through real-time risk awareness, guardian readiness, and calm emergency escalation — without intrusive surveillance.

### 🌐 Live Demo: [Visit](https://youtu.be/6u50YgdIvRw)

- This repository contains the complete frontend and backend-integrated codebase used for the hackathon submission.

# Architecture Diagram:
<img width="1430" height="736" alt="Architecture_Diagram" src="https://github.com/user-attachments/assets/3c579f4c-5369-422a-aa21-2e336da74338" />

# Flow Diagram:
<img width="1760" height="1000" alt="Flow_chart" src="https://github.com/user-attachments/assets/6873c0d7-8a7e-4c9a-a4b0-15cff5cd1c6c" />

# Sequence Diagram:
<img width="1740" height="1495" alt="Sequence Diagram" src="https://github.com/user-attachments/assets/f11f5757-2a08-422f-9100-05f41cdf7389" />

##  Key Features

- Secure Authentication
Email-based login and signup with validation and protected routes.

- Guardian Management
Add and manage trusted emergency contacts who can be notified during high-risk situations.

- Trip-Based Safety Monitoring
Safety tracking activates only during user-initiated trips.

- Live Location Map
Real-time map display using browser geolocation and Leaflet, with safe fallbacks for demo reliability.

- Risk-Level Awareness: Clear safety states — Low, Medium, High — with UI-based escalation.

- Emergency Mode: Dedicated emergency screen showing guardian contacts, live location, and SOS confirmation.

- Auto Mode: Manual risk simulation to showcase escalation workflows safely during demos.

- Privacy by Design: Location data is used only during active trips and never shared without user consent.

## Technology Stack
- Category	Technology
- Frontend	React + TypeScript
- Build Tool	Vite
- Styling	Tailwind CSS + shadcn/ui
- Routing	React Router DOM
- State Management	TanStack React Query
- Forms & Validation	React Hook Form + Zod
- Maps	Leaflet (OpenStreetMap tiles)
- Backend	Supabase (Auth, Database, Realtime)
- Notifications (UI)	Sonner

## Database Overview

- The application uses three core tables, all protected with Row Level Security (RLS):

1️⃣ guardians – trusted emergency contacts

2️⃣ risk_status – current safety level of the user

3️⃣ trips – trip lifecycle and outcomes

Each user can only read and write their own data.

**Node.js (v18+ recommended)**

```sh
npm
```

Setup
# Install dependencies
```sh
npm install
```

# Start development server
``` sh
npm run dev
```

The app will be available at:

```sh
http://localhost:8081
```

## Environment Variables

Create a .env file in the root directory:
```sh
VITE_SUPABASE_URL=your_supabase_project_url
VITE_SUPABASE_ANON_KEY=your_supabase_anon_key
```

- Never commit real environment variables to GitHub.

## Auto Mode

- For safe and reliable demonstrations, NariKawach includes a Demo Mode:

- Simulate Medium / High risk levels

- Trigger emergency flow without real danger

## Responsive Design
- Mobile-first layout
- Touch-friendly controls
- Fixed bottom navigation
- Calm animations and soft color palette for reassurance

## Design Philosophy

- NariKawach is built around three principles:

- Consent First – Monitoring starts only when the user chooses

- Calm Over Panic – No aggressive visuals or fear-driven UI

- Privacy by Default – No background tracking, no surveillance

## Future Plan

- These features are intentionally scoped out of the hackathon build to ensure stability and ethical deployment:

- SMS / WhatsApp alerts to guardians

- AI-driven risk detection

- Hardware-based panic triggers

- Route playback and historical maps

- Production-grade notification services

## Final Note

This project demonstrates a complete, extensible safety workflow with strong UX, ethical design, and clear separation of concerns between frontend, backend, and AI components.

## Team Members:

1️⃣ [@vanshjain271](https://github.com/vanshjain271)
2️⃣ [@Prince-Koladiya09](https://github.com/Prince-Koladiya09)
3️⃣ [@MeetRvyas](https://github.com/MeetRvyas)
4️⃣ [@AnshMNSoni](https://github.com/AnshMNSoni)

## Thankyou
