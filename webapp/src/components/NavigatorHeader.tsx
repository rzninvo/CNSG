import { Compass, MapPin } from "lucide-react";

export function NavigatorHeader() {
  return (
    <header className="relative safe-top px-4 pb-6 pt-4">
      {/* Background glow effects */}
      <div className="absolute inset-0 overflow-hidden">
        <div className="absolute -top-20 -left-20 w-40 h-40 bg-primary/20 rounded-full blur-3xl" />
        <div className="absolute -top-10 -right-10 w-32 h-32 bg-accent/20 rounded-full blur-3xl" />
      </div>
      
      <div className="relative flex items-center justify-center gap-3">
        <div className="relative">
          <div className="absolute inset-0 animate-radar rounded-full bg-primary/30" />
          <div className="relative flex items-center justify-center w-12 h-12 rounded-full bg-gradient-to-br from-primary to-accent">
            <Compass className="w-6 h-6 text-primary-foreground animate-spin-slow" />
          </div>
        </div>
        
        <div className="text-center">
          <h1 className="text-2xl font-bold tracking-tight">
            <span className="gradient-text">Indoor</span>
            <span className="text-foreground"> Navigator</span>
          </h1>
          <p className="text-sm text-muted-foreground flex items-center justify-center gap-1.5 mt-0.5">
            <MapPin className="w-3.5 h-3.5 text-primary" />
            Find your way instantly
          </p>
        </div>
      </div>
    </header>
  );
}
