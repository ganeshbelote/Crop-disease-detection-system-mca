import type { ReactNode } from "react";

interface SectionProps {
  title?: string;
  children: ReactNode;
  className?: string;
}

export default function Section({ title, children, className = "" }: SectionProps) {
  return (
    <section className={`border border-canopy-200 bg-white rounded-sm p-6 ${className}`}>
      {title && (
        <h2 className="font-serif text-xl text-ink mb-3 pb-3 border-b border-canopy-100">
          {title}
        </h2>
      )}
      {children}
    </section>
  );
}
