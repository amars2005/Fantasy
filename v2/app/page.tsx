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
  onImported: (r: { name: string; config: LeagueConfig; unmapped?: string[]; caution?: string }) => void;
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
        <div className="field">
          <label htmlFor="importId">League id</label>
          <input
            id="importId"
            type="text"
            value={leagueId}
            onChange={(e) => setLeagueId(e.target.value)}
            placeholder={source === "sleeper" ? "the long number in your league URL" : "leagueId=…"}
          />
        </div>

        {source === "espn" && (
          <>
            <div className="grid2">
              <div className="field">
                <label htmlFor="s2">espn_s2 cookie</label>
                <input id="s2" type="text" value={espnS2} onChange={(e) => setEspnS2(e.target.value)} />
              </div>
              <div className="field">
                <label htmlFor="swid">SWID cookie</label>
                <input id="swid" type="text" value={swid} onChange={(e) => setSwid(e.target.value)} />
              </div>
            </div>
            <p className="hint">
              Only needed for private leagues. ESPN&apos;s stat ids are undocumented and
              shift between seasons, so whatever comes back lands in the form for you to
              check — nothing is saved until you say so.
            </p>
          </>
        )}

        <button type="button" className="primary" onClick={() => void run()} disabled={busy || !leagueId}>
          {busy ? "Fetching…" : "Import settings"}
        </button>
        {error && <div className="err">{error}</div>}
      </div>
    </div>
  );
}
