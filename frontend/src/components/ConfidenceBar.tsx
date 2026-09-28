interface ConfidenceBarProps {
  label: string;
  confidence: number; // 0-1
  emphasized?: boolean;
}

export default function ConfidenceBar({ label, confidence, emphasized = false }: ConfidenceBarProps) {
  const pct = Math.round(confidence * 1000) / 10; // one decimal place
  return (
    <div className="mb-3 last:mb-0">
      <div className="flex justify-between text-sm mb-1">
        <span className={emphasized ? "font-medium text-ink" : "text-ink/80"}>{label}</span>
        <span className="font-mono text-ink/70">{pct.toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-canopy-100 rounded-sm overflow-hidden">
        <div
          className={`h-full ${emphasized ? "bg-canopy-600" : "bg-canopy-300"}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}
