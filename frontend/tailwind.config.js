/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        // Agricultural / scientific palette — deliberately not the generic
        // warm-cream-and-terracotta AI-generated default.
        canopy: {
          50: "#f4f7f2",
          100: "#e4ebe0",
          200: "#c7d6bd",
          300: "#a3bd93",
          400: "#7ba066",
          500: "#547f42",
          600: "#3d6432",
          700: "#324f2a",
          800: "#2a4024",
          900: "#243620",
        },
        soil: {
          50: "#f8f5f0",
          100: "#ece3d6",
          200: "#d7c4a6",
          300: "#bd9d74",
          400: "#a67c4e",
          500: "#8a6138",
          600: "#6f4b2c",
          700: "#5a3c25",
          800: "#4a3220",
          900: "#3d2a1c",
        },
        paper: "#f6f5f0",
        ink: "#1f261d",
        rust: "#a6491f",
      },
      fontFamily: {
        serif: ["\"Lora\"", "Georgia", "serif"],
        sans: ["\"IBM Plex Sans\"", "system-ui", "sans-serif"],
        mono: ["\"IBM Plex Mono\"", "monospace"],
      },
    },
  },
  plugins: [],
};
