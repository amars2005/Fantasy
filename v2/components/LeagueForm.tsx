"use client";

import { useMemo, useState } from "react";

import { anchorWarnings, nearestFfcFormat, REFERENCE_LEAGUE } from "../lib/config";
import type { LeagueConfig, Position, Slot } from "../lib/types";

const SLOTS: Slot[] = ["QB", "RB", "WR", "TE", "FLEX", "SUPERFLEX", "K", "DST"];

/** The handful of rules most leagues actually differ on. */
const HEADLINE_SCORING: { key: string; label: string }[] = [
  { key: "receptions", label: "Per reception" },
  { key: "passing_tds", label: "Passing TD" },
  { key: "passing_yards", label: "Per passing yard" },
  { key: "rushing_tds", label: "Rushing TD" },
  { key: "receiving_tds", label: "Receiving TD" },
  { key: "passing_interceptions", label: "Interception" },
];

export interface LeagueFormProps {
  initial?: { name: string; config: LeagueConfig };
  notice?: string | null;
  unmapped?: string[];
  onSubmit: (name: string, config: LeagueConfig) => Promise<void>;
  submitting: boolean;
  error: string | null;
}

export default function LeagueForm({
  initial,
  notice,
  unmapped,
  onSubmit,
  submitting,
  error,
}: LeagueFormProps) {
  const [name, setName] = useState(initial?.name ?? "My league");
  const [config, setConfig] = useState<LeagueConfig>(
    initial?.config ?? structuredClone(REFERENCE_LEAGUE),
  );
  const [showAdvanced, setShowAdvanced] = useState(false);

  const format = useMemo(() => nearestFfcFormat(config), [config]);
  const warnings = useMemo(() => anchorWarnings(config), [config]);

  const patch = (next: Partial<LeagueConfig>) => setConfig((c) => ({ ...c, ...next }));

  const setStarter = (slot: Slot, value: number) =>
    patch({ starters: { ...config.starters, [slot]: value } });

  const setScoring = (key: string, value: number) =>
    patch({ scoring: { ...config.scoring, [key]: value } });

  const starterCount = Object.values(config.starters).reduce((a, b) => a + (b ?? 0), 0);

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void onSubmit(name, config);
      }}
    >
      {notice && <div className="banner">{notice}</div>}
      {unmapped && unmapped.length > 0 && (
        <div className="banner">
          <strong>Not imported, so these kept their defaults:</strong>{" "}
          {unmapped.join(", ")}. Check them below before saving.
        </div>
      )}

      <div className="panel">
        <h2>League</h2>
        <div className="body">
          <div className="field">
            <label htmlFor="lname">Name</label>
            <input
              id="lname"
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
            />
          </div>
          <div className="grid2">
            <div className="field">
              <label htmlFor="teams">Teams</label>
              <input
                id="teams"
                type="number"
                min={2}
                max={32}
                value={config.teams}
                onChange={(e) => patch({ teams: Number(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="rounds">Rounds</label>
              <input
                id="rounds"
                type="number"
                min={1}
                max={40}
                value={config.rounds}
                onChange={(e) => patch({ rounds: Number(e.target.value) })}
              />
            </div>
            <div className="field">
              <label htmlFor="playoffTeams">Playoff teams</label>
              <input
                id="playoffTeams"
                type="number"
                min={2}
                value={config.schedule.playoffTeams}
                onChange={(e) =>
                  patch({
                    schedule: { ...config.schedule, playoffTeams: Number(e.target.value) },
                  })
                }
              />
            </div>
            <div className="field">
              <label htmlFor="regWeeks">Regular season weeks</label>
              <input
                id="regWeeks"
                type="number"
                min={1}
                max={17}
                value={config.schedule.regularSeasonWeeks}
                onChange={(e) => {
                  const w = Number(e.target.value);
                  patch({
                    schedule: {
                      ...config.schedule,
                      regularSeasonWeeks: w,
                      // Playoff weeks follow the regular season unless edited.
                      playoffWeeks: [w + 1, w + 2, w + 3],
                    },
                  });
                }}
              />
            </div>
          </div>
          <p className="hint">
            Playoff weeks {config.schedule.playoffWeeks.join(", ")}. These are used to
            price schedule strength in the weeks that actually decide the title, which
            ADP cannot see.
          </p>
        </div>
      </div>

      <div className="panel">
        <h2>Starting lineup</h2>
        <div className="body">
          <div className="grid2">
            {SLOTS.map((slot) => (
              <div className="field" key={slot}>
                <label htmlFor={`slot-${slot}`}>{slot}</label>
                <input
                  id={`slot-${slot}`}
                  type="number"
                  min={0}
                  max={6}
                  value={config.starters[slot] ?? 0}
                  onChange={(e) => setStarter(slot, Number(e.target.value))}
                />
              </div>
            ))}
          </div>
          <p className="hint">
            {starterCount} starters over {config.rounds} rounds.{" "}
            {starterCount > config.rounds && (
              <span className="scarce">
                That lineup cannot be drafted in this many rounds.
              </span>
            )}
          </p>
        </div>
      </div>

      <div className="panel">
        <h2>Scoring</h2>
        <div className="body">
          <div className="grid2">
            {HEADLINE_SCORING.map(({ key, label }) => (
              <div className="field" key={key}>
                <label htmlFor={`sc-${key}`}>{label}</label>
                <input
                  id={`sc-${key}`}
                  type="number"
                  step="0.01"
                  value={config.scoring[key] ?? 0}
                  onChange={(e) => setScoring(key, Number(e.target.value))}
                />
              </div>
            ))}
          </div>

          <button type="button" onClick={() => setShowAdvanced((v) => !v)}>
            {showAdvanced ? "Hide" : "Show"} every scoring rule
          </button>

          {showAdvanced && (
            <div className="grid2" style={{ marginTop: 12 }}>
              {Object.keys(config.scoring)
                .filter((k) => !HEADLINE_SCORING.some((h) => h.key === k))
                .map((key) => (
                  <div className="field" key={key}>
                    <label htmlFor={`adv-${key}`}>{key.replace(/_/g, " ")}</label>
                    <input
                      id={`adv-${key}`}
                      type="number"
                      step="0.01"
                      value={config.scoring[key]}
                      onChange={(e) => setScoring(key, Number(e.target.value))}
                    />
                  </div>
                ))}
            </div>
          )}

          <p className="hint">
            ADP will be anchored to the <strong>{format}</strong> board.
          </p>
          {warnings.length > 0 && (
            <div className="banner" style={{ marginTop: 10 }}>
              {warnings.map((w) => (
                <div key={w}>{w}</div>
              ))}
            </div>
          )}
        </div>
      </div>

      <button className="primary" type="submit" disabled={submitting}>
        {submitting ? "Building the board…" : "Create league"}
      </button>
      {error && <div className="err">{error}</div>}
    </form>
  );
}
