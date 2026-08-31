/**
 * The "your leagues" list, kept in this browser.
 *
 * There are no accounts, so a league URL is the only way back into a league.
 * That is fine for your own bookmark and less fine as something other people
 * rely on, so the device at least remembers what it created. Losing this list
 * loses nothing that the URL would not recover.
 */

const KEY = "fantasy.v2.leagues";

export interface RememberedLeague {
  id: string;
  name: string;
  createdAt: string;
}

function read(): RememberedLeague[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = window.localStorage.getItem(KEY);
    return raw ? (JSON.parse(raw) as RememberedLeague[]) : [];
  } catch {
    // Private windows, cleared site data, browsers set to block storage.
    return [];
  }
}

function write(leagues: RememberedLeague[]): void {
  try {
    window.localStorage.setItem(KEY, JSON.stringify(leagues));
  } catch {
    // Not being able to remember is not worth breaking the page over.
  }
}

export function listLeagues(): RememberedLeague[] {
  return read().sort((a, b) => (a.createdAt < b.createdAt ? 1 : -1));
}

export function rememberLeague(league: RememberedLeague): void {
  const existing = read().filter((l) => l.id !== league.id);
  write([league, ...existing].slice(0, 50));
}

export function forgetLeague(id: string): void {
  write(read().filter((l) => l.id !== id));
}
