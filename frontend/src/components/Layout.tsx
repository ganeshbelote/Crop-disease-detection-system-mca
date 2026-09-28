import { NavLink, Outlet } from "react-router-dom";
import { Leaf } from "lucide-react";

const navItems = [
  { to: "/", label: "Dashboard", end: true },
  { to: "/analyze", label: "Analyze Leaf" },
  { to: "/history", label: "History" },
  { to: "/about", label: "Methodology" },
];

export default function Layout() {
  return (
    <div className="min-h-screen flex flex-col">
      <header className="border-b border-canopy-200 bg-white/60">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <Leaf className="w-5 h-5 text-canopy-600" strokeWidth={2} aria-hidden />
            <div>
              <p className="font-serif text-lg leading-none text-ink">Crop Disease Detection</p>
              <p className="text-xs text-canopy-700 font-mono mt-0.5">MCA Deep Learning Project</p>
            </div>
          </div>
          <nav aria-label="Primary" className="flex gap-1">
            {navItems.map((item) => (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.end}
                className={({ isActive }) =>
                  `px-3 py-1.5 text-sm rounded-sm transition-colors ${
                    isActive
                      ? "bg-canopy-600 text-white"
                      : "text-ink/80 hover:bg-canopy-100"
                  }`
                }
              >
                {item.label}
              </NavLink>
            ))}
          </nav>
        </div>
      </header>

      <main className="flex-1 max-w-5xl w-full mx-auto px-6 py-10">
        <Outlet />
      </main>

      <footer className="border-t border-canopy-200 mt-auto">
        <div className="max-w-5xl mx-auto px-6 py-5 text-xs text-ink/60 leading-relaxed">
          This system provides an informational, model-based prediction only. It
          is not a substitute for diagnosis by a qualified agricultural
          extension officer or plant pathologist.
        </div>
      </footer>
    </div>
  );
}
