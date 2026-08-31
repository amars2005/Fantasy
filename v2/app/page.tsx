"use client";

import { useEffect, useState } from "react";

import LeagueForm from "../components/LeagueForm";
import { listLeagues, forgetLeague, rememberLeague, type RememberedLeague } from "../lib/localLeagues";
import type { LeagueConfig } from "../lib/types";

type Tab = "manual" | "sleeper" | "espn";

export default function Home() {
  const [leagues, setLeagues] = useState<RememberedLeague[]>([]);
  const [tab, setTab] = useState<Tab>("manual");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [imported, setImported] = useState<{
    name: string;
    config: LeagueConfig;
    unmapped?: string[];
    ignored?: string[];
    notes?: string[];
    caution?: string;
  } | null>(null);

  useEffect(() => setLeagues(listLeagues()), []);

  async function createLeague(name: string, config: LeagueConfig) {
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/leagues", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ name, config }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Could not create the league");

      rememberLeague({ id: data.id, name: data.name, createdAt: new Date().toISOString() });
      window.location.href = `/league/${data.id}`;
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setSubmitting(false);
    }
  }

  return (
    <div className="page">
      <h1>Draft board</h1>
      <p>
        Market-anchored projections under your own scoring, value over what actually
        survives to your next pick, and tiers taken from where the data cannot separate
        players.
      </p>

      {leagues.length > 0 && (
        <div className="panel">
          <h2>Your leagues on this device</h2>
          <div className="body">
            <ul className="leaguelist">
              {leagues.map((l) => (
                <li key={l.id}>
                  <a href={`/league/${l.id}`}>{l.name}</a>
                  <button
                    type="button"
                    onClick={() => {
                      forgetLeague(l.id);
                      setLeagues(listLeagues());
                    }}
                  >
                    Forget
                  </button>
                </li>
              ))}
            </ul>
            <p className="hint">
              There are no accounts, so a league&apos;s link is the only way back into it.
              This list lives in this browser only — keep the link somewhere safe.
            </p>
          </div>
        </div>
      )}

      <h2 style={{ fontSize: 15, marginTop: 28 }}>New league</h2>
      <div className="tabs">
        <button
          type="button"
          className={tab === "manual" ? "active" : ""}
          onClick={() => {
            setTab("manual");
            setImported(null);
          }}
        >
          Enter settings
        </button>
        <button
          type="button"
          className={tab === "sleeper" ? "active" : ""}
          onClick={() => setTab("sleeper")}
        >
          Import from Sleeper
        </button>
        <button
          type="button"
          className={tab === "espn" ? "active" : ""}
          onClick={() => setTab("espn")}
        >
          Import from ESPN
        </button>
      </div>

      {tab !== "manual" && !imported && (
        <ImportPanel
          source={tab}
          onImported={(result) => {
            setImported(result);
            setTab("manual");
          }}
        />
      )}

      {(tab === "manual" || imported) && (
        <LeagueForm
          initial={imported ?? undefined}
          notice={imported?.caution ?? null}
          unmapped={imported?.unmapped}
          ignored={imported?.ignored}
          notes={imported?.notes}
          onSubmit={createLeague}
          submitting={submitting}
          error={error}
        />
      )}
    </div>
  );
}

function ImportPanel({
  source,
  onImported,
}: {
  source: "sleeper" | "espn";
  onImported: (r: {
    name: string;
    config: LeagueConfig;
    unmapped?: string[];
    ignored?: string[];
    notes?: string[];
    caution?: string;
  }) => void;
}) {
  const [leagueId, setLeagueId] = useState("");
  const [espnS2, setEspnS2] = useState("");
  const [swid, setSwid] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true);
    setError(null);
    try {
      const res = await fetch("/api/import", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ source, leagueId, espnS2, swid }),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error ?? "Import failed");
      onImported(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="panel">
      <h2>{source === "sleeper" ? "Sleeper" : "ESPN"}</h2>
      <div className="body">
        {source === "espn" && (
          <ol className="steps">
            <li>
              Open your league on <code>fantasy.espn.com</code> while signed in.
            </li>
            <li>
              In the address bar, find <code>leagueId=</code> and copy the number
              straight after it. From{" "}
              <code>…/league?leagueId=1234567&amp;seasonId=2026</code> that is{" "}
              <strong>1234567</strong>. Pasting the whole link works too.
            </li>
            <li>
              <strong>Public league?</strong> Leave the two cookie boxes empty and
              press Import.
            </li>
            <li>
              <strong>Private league?</strong> Both cookies are required — one on
              its own will not work. Open DevTools (F12) →{" "}
              <em>Application</em> → <em>Cookies</em> →{" "}
              <code>https://fantasy.espn.com</code>, then copy the two values
              below exactly as shown, with no <code>espn_s2=</code> prefix and no
              trailing semicolon. <code>SWID</code> keeps its curly braces.
            </li>
          </ol>
        )}

        <div className="field">
          <label htmlFor="importId">League id</label>
          <input
            id="importId"
            type="text"
            value={leagueId}
            onChange={(e) => setLeagueId(e.target.value)}
            placeholder={
              source === "sleeper" ? "the long number in your league URL" : "1234567"
            }
          />
        </div>

        {source === "espn" && (
          <>
            <div className="grid2">
              <div className="field">
                <label htmlFor="s2">espn_s2 cookie</label>
                <input
                  id="s2"
                  type="text"
                  value={espnS2}
                  onChange={(e) => setEspnS2(e.target.value)}
                  placeholder="AEBxyz%2F… (long, private leagues only)"
                />
              </div>
              <div className="field">
                <label htmlFor="swid">SWID cookie</label>
                <input
                  id="swid"
                  type="text"
                  value={swid}
                  onChange={(e) => setSwid(e.target.value)}
                  placeholder="{1A2B3C4D-…}"
                />
              </div>
            </div>
            <p className="hint">
              <strong>After importing:</strong> ESPN&apos;s stat ids are
              undocumented and shift between seasons, so nothing is saved yet.
              The next screen is your settings filled in — check the scoring, the
              lineup, and the kicker and defence values, then press Create
              league. Anything the importer could not place is listed at the top.
            </p>
          </>
        )}

        <button
          type="button"
          className="primary"
          onClick={() => void run()}
          disabled={busy || !leagueId.trim()}
        >
          {busy ? "Fetching…" : "Import settings"}
        </button>
        {error && <div className="err">{error}</div>}
      </div>
    </div>
  );
}
