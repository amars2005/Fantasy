"use client";

import { useMemo, useState } from "react";

import {
  ALL_SCORING_KEYS,
  anchorWarnings,
  DST_PBP_EVENTS,
  nearestFfcFormat,
  REFERENCE_LEAGUE,
  ruleLabel,
} from "../lib/config";
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
  /** Rules worth points that the model could not express. */
  unmapped?: string[];
  /** Rules the platform returned at zero, so they change nothing. */
  ignored?: string[];
  /** Judgement calls the importer made that are worth reviewing. */
  notes?: string[];
  onSubmit: (name: string, config: LeagueConfig) => Promise<void>;
  submitting: boolean;
  error: string | null;
}

export default function LeagueForm({
  initial,
  notice,
  unmapped,
  ignored,
  notes,
  onSubmit,
  submitting,
  error,
}: LeagueFormProps) {
  const [name, setName] = useState(initial?.name ?? "My league");
  const [config, setConfig] = useState<LeagueConfig>(
    initial?.config ?? structuredClone(REFERENCE_LEAGUE),
  );
  const [showAdvanced, setShowAdvanced] = useState(false);
  // Open by default after an import: those are exactly the values that need a
  // second pair of eyes, and a collapsed panel is a panel nobody reads.
  const [showKdst, setShowKdst] = useState(Boolean(initial));

  const format = useMemo(() => nearestFfcFormat(config), [config]);
  const warnings = useMemo(() => anchorWarnings(config), [config]);

  const patch = (next: Partial<LeagueConfig>) => setConfig((c) => ({ ...c, ...next }));

  const setStarter = (slot: Slot, value: number) =>
    patch({ starters: { ...config.starters, [slot]: value } });

  const setScoring = (key: string, value: number) =>
    patch({ scoring: { ...config.scoring, [key]: value } });

  const setKicker = (key: string, value: number) =>
    patch({ kickerScoring: { ...config.kickerScoring, [key]: value } });

  const setDstEvent = (key: string, value: number) =>
    patch({ dst: { ...config.dst, events: { ...config.dst.events, [key]: value } } });

  const setBand = (
    which: "pointsAllowedBands" | "yardsAllowedBands",
    index: number,
    points: number,
  ) =>
    patch({
      dst: {
        ...config.dst,
        [which]: config.dst[which].map((b, i) => (i === index ? { ...b, points } : b)),
      },
    });

  // Every rule the board can score, not merely the ones this league already
  // has a number for. A league entered by hand starts from the reference
  // league's rules, which do not include the long-touchdown bonuses -- without
  // this there is no way to type one in.
  const advancedScoringKeys = useMemo(() => {
    const keys = [...new Set([...Object.keys(config.scoring), ...ALL_SCORING_KEYS])];
    return keys.filter((k) => !HEADLINE_SCORING.some((h) => h.key === k));
  }, [config.scoring]);

  const dstEventKeys = useMemo(
    () => [...new Set([...Object.keys(config.dst.events), ...DST_PBP_EVENTS])],
    [config.dst.events],
  );

  const starterCount = Object.values(config.starters).reduce((a, b) => a + (b ?? 0), 0);

  // An all-zero ladder is legitimate, but it is also what a drifted id table
  // would produce, so it is called out rather than left to be discovered later.
  const kdstAllZero = useMemo(
    () =>
      [...config.dst.pointsAllowedBands, ...config.dst.yardsAllowedBands].every(
        (b) => b.points === 0,
      ),
    [config.dst.pointsAllowedBands, config.dst.yardsAllowedBands],
  );

  return (
    <form
      onSubmit={(e) => {
        e.preventDefault();
        void onSubmit(name, config);
      }}
    >
      {notice && <div className="banner">{notice}</div>}
      {unmapped && unmapped.length > 0 && (
        <div className="banner offline">
          <strong>Worth points here, but the board cannot score them:</strong>
          <ul style={{ margin: "6px 0 0", paddingLeft: 18 }}>
            {unmapped.map((u) => (
              <li key={u}>{u}</li>
            ))}
          </ul>
          <p className="hint" style={{ marginTop: 6 }}>
            These are almost always small per-play bonuses. Projections will run
            slightly low for the players who earn them.
          </p>
        </div>
      )}
      {notes && notes.length > 0 && (
        <div className="banner">
          {notes.map((n) => (
            <div key={n}>{n}</div>
          ))}
        </div>
      )}
      {ignored && ignored.length > 0 && (
        <details style={{ marginBottom: 12 }}>
          <summary className="hint" style={{ cursor: "pointer" }}>
            {ignored.length === 1
              ? "1 more rule your league leaves at zero — it changes nothing, but you can check it."
              : `${ignored.length} more rules your league leaves at zero — they change nothing, but you can check the list.`}
          </summary>
          <p className="hint">{ignored.join(" · ")}</p>
        </details>
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
              {advancedScoringKeys.map((key) => (
                <div className="field" key={key}>
                  <label htmlFor={`adv-${key}`}>{ruleLabel(key)}</label>
                  <input
                    id={`adv-${key}`}
                    type="number"
                    step="0.01"
                    value={config.scoring[key] ?? 0}
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

      <div className="panel">
        <h2>Kicker and defence</h2>
        <div className="body">
          <p className="hint" style={{ marginTop: 0 }}>
            These decide where kickers and defences land on the board. An import
            fills them in, so they are worth a glance even though they move
            nothing until the last two rounds.
          </p>

          <button type="button" onClick={() => setShowKdst((v) => !v)}>
            {showKdst ? "Hide" : "Show"} kicker and defence scoring
          </button>

          {showKdst && (
            <>
              <h3 style={{ fontSize: 13, margin: "16px 0 8px" }}>Field goals</h3>
              <div className="grid2">
                {Object.keys(config.kickerScoring).map((key) => (
                  <div className="field" key={key}>
                    <label htmlFor={`k-${key}`}>{ruleLabel(key)}</label>
                    <input
                      id={`k-${key}`}
                      type="number"
                      step="0.01"
                      value={config.kickerScoring[key]}
                      onChange={(e) => setKicker(key, Number(e.target.value))}
                    />
                  </div>
                ))}
              </div>

              <h3 style={{ fontSize: 13, margin: "16px 0 8px" }}>Defensive events</h3>
              <div className="grid2">
                {dstEventKeys.map((key) => (
                  <div className="field" key={key}>
                    <label htmlFor={`d-${key}`}>{ruleLabel(key)}</label>
                    <input
                      id={`d-${key}`}
                      type="number"
                      step="0.01"
                      value={config.dst.events[key] ?? 0}
                      onChange={(e) => setDstEvent(key, Number(e.target.value))}
                    />
                  </div>
                ))}
              </div>

              <h3 style={{ fontSize: 13, margin: "16px 0 8px" }}>Points allowed</h3>
              <div className="grid2">
                {config.dst.pointsAllowedBands.map((b, i) => (
                  <div className="field" key={`${b.low}-${b.high}`}>
                    <label htmlFor={`pa-${i}`}>
                      {b.high >= 999 ? `${b.low}+` : `${b.low}-${b.high}`}
                    </label>
                    <input
                      id={`pa-${i}`}
                      type="number"
                      step="0.5"
                      value={b.points}
                      onChange={(e) => setBand("pointsAllowedBands", i, Number(e.target.value))}
                    />
                  </div>
                ))}
              </div>

              <h3 style={{ fontSize: 13, margin: "16px 0 8px" }}>Yards allowed</h3>
              <div className="grid2">
                {config.dst.yardsAllowedBands.map((b, i) => (
                  <div className="field" key={`${b.low}-${b.high}`}>
                    <label htmlFor={`ya-${i}`}>
                      {b.high >= 99999 ? `${b.low}+` : `${b.low}-${b.high}`}
                    </label>
                    <input
                      id={`ya-${i}`}
                      type="number"
                      step="0.5"
                      value={b.points}
                      onChange={(e) => setBand("yardsAllowedBands", i, Number(e.target.value))}
                    />
                  </div>
                ))}
              </div>

              {kdstAllZero && (
                <p className="hint scarce" style={{ marginTop: 10 }}>
                  Every points- and yards-allowed band is zero, so defences will
                  be scored on events alone. That is a real setting in some
                  leagues — but if yours does score a shutout, fill these in.
                </p>
              )}
            </>
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
