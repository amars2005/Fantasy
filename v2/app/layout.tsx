import type { Metadata } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Draft Board",
  description:
    "A projection engine and draft optimiser: market-anchored projections, " +
    "value over next available, and tiers derived from where the data cannot " +
    "separate players.",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
