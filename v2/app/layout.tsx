import type { Metadata, Viewport } from "next";

import "./globals.css";

export const metadata: Metadata = {
  title: "Draft Board",
  description:
    "A projection engine and draft optimiser: market-anchored projections, " +
    "value over next available, and tiers derived from where the data cannot " +
    "separate players.",
};

/**
 * Without this a phone renders the page at a 980px desktop width and scales it
 * down, which is why the board arrived unreadable however the CSS was written.
 * `maximumScale` is deliberately left alone: pinching a column of numbers is a
 * reasonable thing to want to do mid-draft.
 */
export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#0f1216",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
