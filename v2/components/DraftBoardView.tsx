"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { DraftBoard, type Recommendation } from "../lib/board";
import { applyNews, isEmptyTag, type NewsMap, type NewsTag } from "../lib/news";
import type { LeagueConfig, Player } from "../lib/types";
import NewsPanel, { NewsBadge } from "./NewsPanel";

interface Pick {
  seq: number;
  playerId: string | null;
  takenBy: string;
  voidedAt: string | null;
}

interface LeaguePayload {
  id: string;
  name: string;
  config: LeagueConfig;
  board: Player[];
  adpAsOf: string | null;
  refreshAvailable: boolean;
  warnings: string[];
  picks: Pick[];
  /** Team nickname lookup, so defences answer to what the room calls them. */
  teamNicknames: Record<string, string>;
}

const cacheKey = (id: string) => `fantasy.v2.league.${id}`;
const slotKey = (id: string) => `fantasy.v2.slot.${id}`;
const newsKey = (id: string) => `fantasy.v2.news.${id}`;

/** Mirror the league locally so a dropped connection does not empty the board. */
function readCache(id: string): LeaguePayload | null {
  try {
    const raw = window.localStorage.getItem(cacheKey(id));
    return raw ? (JSON.parse(raw) as LeaguePayload) : null;
  } catch {
    return null;
  }
}

function writeCache(id: string, payload: LeaguePayload): void {
  try {
    window.localStorage.setItem(cacheKey(id), JSON.stringify(payload));
  } catch {
    // Over quota or storage blocked. The board still works from memory.
  }
}

export default function DraftBoardView({ leagueId }: { leagueId: string }) {
  const [league, setLeague] = useState<LeaguePayload | null>(null);
  const [picks, setPicks] = useState<Pick[]>([]);
  const [slot, setSlot] = useState(1);
  const [query, setQuery] = useState("");
  /** What we know that the ADP snapshot does not. Local to this browser. */
  const [news, setNews] = useState<NewsMap>({});
  const [editingNews, setEditingNews] = useState<string | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [offline, setOffline] = useState(false);
  const [fromCache, setFromCache] = useState(false);
  const searchRef = useRef<HTMLInputElement>(null);

  /** Picks written optimistically but not yet acknowledged by the server. */
  const pending = useRef<Pick[]>([]);

  // --- loading --------------------------------------------------------------

  const load = useCallback(async () => {
    try {
      const res = await fetch(`/api/leagues/${leagueId}`);
      if (!res.ok) throw new Error((await res.json()).error ?? "Could not load the league");
      const data = (await res.json()) as LeaguePayload;
      setLeague(data);
      setPicks(data.picks);
      writeCache(leagueId, data);
      setOffline(false);
      setFromCache(false);
    } catch (err) {
      const cached = readCache(leagueId);
      if (cached) {
        setLeague(cached);
        setPicks(cached.picks);
        setFromCache(true);
        setOffline(true);
      } else {
        setLoadError(err instanceof Error ? err.message : String(err));
      }
    }
  }, [leagueId]);

  useEffect(() => {
    void load();
    try {
      const saved = window.localStorage.getItem(slotKey(leagueId));
      if (saved) setSlot(Number(saved));
      const savedNews = window.localStorage.getItem(newsKey(leagueId));
      if (savedNews) setNews(JSON.parse(savedNews) as NewsMap);
    } catch {
      /* storage blocked, or a tag written by an older version */
    }
  }, [leagueId, load]);

  // Reconcile on focus rather than polling. One person runs a league, so the
  // server is durability, not coordination -- but a second tab or a phone
  // still needs to catch up when it comes forward.
  useEffect(() => {
    const onFocus = () => void load();
    window.addEventListener("focus", onFocus);
    return () => window.removeEventListener("focus", onFocus);
  }, [load]);

  useEffect(() => {
    try {
      window.localStorage.setItem(slotKey(leagueId), String(slot));
    } catch {
      /* storage blocked */
    }
  }, [leagueId, slot]);

  useEffect(() => {
    try {
      window.localStorage.setItem(newsKey(leagueId), JSON.stringify(news));
    } catch {
      /* storage blocked. The tags still hold for this session. */
    }
  }, [leagueId, news]);

  const setTag = useCallback((playerId: string, tag: NewsTag | null) => {
    setNews((prev) => {
      const next = { ...prev };
      // A tag that says nothing is a tag not worth keeping -- it would sit in
      // the panel implying the board had been adjusted when it had not.
      if (tag === null || isEmptyTag(tag)) delete next[playerId];
      else next[playerId] = tag;
      return next;
    });
    if (tag === null) setEditingNews((cur) => (cur === playerId ? null : cur));
  }, []);

  // --- board ----------------------------------------------------------------

  const active = useMemo(
    () => picks.filter((p) => p.voidedAt === null).sort((a, b) => a.seq - b.seq),
    [picks],
  );

  /** The board as re-priced by what we know. Untagged players pass through. */
  const adjusted = useMemo(
    () => (league ? applyNews(league.board, news) : []),
    [league, news],
  );

  const board = useMemo(() => {
    if (!league) return null;
    const b = new DraftBoard(adjusted, league.config, slot, league.teamNicknames ?? {});
    active.forEach((p, i) => {
      // A pick with no player is one we could not identify; the clock still
      // has to move or every "picks until my next turn" number drifts.
      b.draft(p.playerId ?? `__unknown_${i}`, p.takenBy === "me" ? "me" : "other");
    });
    return b;
  }, [league, adjusted, slot, active]);

  const recommendation: Recommendation | null = useMemo(
    () => (board ? board.recommend(14) : null),
    [board],
  );

  const results = useMemo(
    () => (board && query ? board.find(query, 8) : []),
    [board, query],
  );

  // --- writing --------------------------------------------------------------

  const send = useCallback(
    async (pick: Pick) => {
      try {
        const res = await fetch(`/api/leagues/${leagueId}/picks`, {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            seq: pick.seq,
            playerId: pick.playerId,
            takenBy: pick.takenBy,
          }),
        });
        if (!res.ok) throw new Error(String(res.status));
        pending.current = pending.current.filter((p) => p.seq !== pick.seq);
        setOffline(false);
      } catch {
        // Keep it queued. `seq` makes the retry idempotent, so re-sending a
        // pick that actually landed is a no-op rather than a double-count.
        setOffline(true);
      }
    },
    [leagueId],
  );

  // Flush anything queued whenever we come back online.
  useEffect(() => {
    if (offline) return;
    const queued = [...pending.current];
    for (const p of queued) void send(p);
  }, [offline, send]);

  const mark = useCallback(
    (playerId: string | null, takenBy: "me" | "other") => {
      const seq = Math.max(0, ...picks.map((p) => p.seq)) + 1;
      const pick: Pick = { seq, playerId, takenBy, voidedAt: null };
      setPicks((prev) => [...prev, pick]);
      pending.current.push(pick);
      void send(pick);
      setQuery("");
      searchRef.current?.focus();
    },
    [picks, send],
  );

  const undoLast = useCallback(() => {
    const last = active[active.length - 1];
    if (!last) return;
    setPicks((prev) =>
      prev.map((p) => (p.seq === last.seq ? { ...p, voidedAt: new Date().toISOString() } : p)),
    );
    void fetch(`/api/leagues/${leagueId}/picks?seq=${last.seq}`, { method: "DELETE" }).catch(
      () => setOffline(true),
    );
  }, [active, leagueId]);

  // --- render ---------------------------------------------------------------

  if (loadError) {
    return (
      <div className="page">
        <h1>Not found</h1>
        <p>{loadError}</p>
        <p>
          <a href="/">Back to your leagues</a>
        </p>
      </div>
    );
  }

  if (!league || !board || !recommendation) {
    return (
      <div className="page">
        <p className="muted">Loading the board…</p>
      </div>
    );
  }

  const mine = recommendation.my_next_pick === recommendation.on_the_clock;

  return (
    <>
      <header className="app">
        <h1>{league.name}</h1>
        <div className="stat">
          <b className={mine ? "onclock" : ""}>{recommendation.on_the_clock}</b>
          <span>on the clock</span>
        </div>
        <div className="stat">
          <b>{recommendation.my_next_pick ?? "-"}</b>
          <span>my next pick</span>
        </div>
        <div className="stat">
          <b>{recommendation.picks_until_next}</b>
          <span>picks between mine</span>
        </div>
        <div className="stat">
          <b>{recommendation.roster.length}</b>
          <span>on my roster</span>
        </div>
        <div style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <label htmlFor="slot" style={{ margin: 0 }}>
            Slot
          </label>
          <select
            id="slot"
            value={slot}
            onChange={(e) => setSlot(Number(e.target.value))}
            style={{ width: 70 }}
          >
            {Array.from({ length: league.config.teams }, (_, i) => i + 1).map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
          <a href={`/api/leagues/${leagueId}/cheatsheet?slot=${slot}`}>
            <button type="button">Cheatsheet</button>
          </a>
        </div>
      </header>

      <main className="board">
        <div>
          <div className="panel">
            <h2>Mark a pick</h2>
            <div className="body">
              <input
                ref={searchRef}
                type="search"
                value={query}
                placeholder="Type a name…"
                autoFocus
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" || !results.length) return;
                  e.preventDefault();
                  mark(results[0].player_id, e.shiftKey ? "me" : "other");
                }}
              />
              <div className="hint">
                <kbd>Enter</kbd> taken by someone else &middot;{" "}
                <kbd>Shift</kbd>+<kbd>Enter</kbd> taken by me
              </div>

              {results.map((p) => (
                <div className="slot" key={p.player_id}>
                  <span>
                    <span className={`pos ${p.pos}`}>{p.pos}</span>{" "}
                    <span className="name">{p.name}</span>{" "}
                    <span className="muted">{p.tm}</span> <NewsBadge player={p} />
                  </span>
                  <span style={{ display: "flex", gap: 4 }}>
                    <button type="button" onClick={() => setEditingNews(p.player_id)}>
                      News
                    </button>
                    <button type="button" onClick={() => mark(p.player_id, "other")}>
                      Gone
                    </button>
                    <button type="button" className="mine" onClick={() => mark(p.player_id, "me")}>
                      Mine
                    </button>
                  </span>
                </div>
              ))}

              <div style={{ display: "flex", gap: 6, marginTop: 12 }}>
                <button type="button" onClick={() => mark(null, "other")}>
                  Someone took a player not on the board
                </button>
                <button type="button" onClick={undoLast} disabled={!active.length}>
                  Undo
                </button>
              </div>
            </div>
          </div>

          <NewsPanel
            players={adjusted}
            news={news}
            editing={editingNews}
            onEdit={setEditingNews}
            onChange={setTag}
          />

          <div className="panel">
            <h2>Recent picks</h2>
            <div className="body">
              {active.length === 0 && <div className="empty">Nothing yet.</div>}
              {[...active]
                .reverse()
                .slice(0, 12)
                .map((p) => {
                  const player = league.board.find((b) => b.player_id === p.playerId);
                  return (
                    <div className="slot" key={p.seq}>
                      <span className="lbl">{p.seq}</span>
                      <span style={{ flex: 1, textAlign: "left", paddingLeft: 8 }}>
                        {player ? (
                          <>
                            <span className={`pos ${player.pos}`}>{player.pos}</span>{" "}
                            {player.name}
                          </>
                        ) : (
                          <span className="empty">unidentified pick</span>
                        )}
                      </span>
                      {p.takenBy === "me" && <span className="vona">mine</span>}
                    </div>
                  );
                })}
            </div>
          </div>
        </div>

        <div>
          {offline && (
            <div className="banner offline">
              Offline. Picks are being kept locally and will sync when the connection
              returns — marking the same pick twice is safe.
            </div>
          )}
          {fromCache && !offline && (
            <div className="banner">Showing a locally cached board.</div>
          )}
          {league.refreshAvailable && (
            <div className="banner">
              Newer ADP is available, but this draft has already started so the board has
              been left alone. Refreshing would move every projection, tier and survival
              number underneath a draft in progress.{" "}
              <button
                type="button"
                onClick={() => {
                  void fetch(`/api/leagues/${leagueId}/refresh`, { method: "POST" }).then(() =>
                    load(),
                  );
                }}
              >
                Refresh anyway
              </button>
            </div>
          )}
          {league.warnings.map((w) => (
            <div className="banner" key={w}>
              {w}
            </div>
          ))}

          <div className={`verdict ${mine ? "live" : ""}`}>
            <div>
              <div className="lbl">{mine ? "You are on the clock" : "Best available"}</div>
              <div className="big">
                {recommendation.recommendations[0]?.name ?? "Board exhausted"}
              </div>
            </div>
            {recommendation.dropoff[0] && (
              <div>
                <div className="lbl">Most urgent position</div>
                <div className="big">
                  {recommendation.dropoff[0].pos}{" "}
                  <span className="muted">
                    {recommendation.dropoff[0].dropoff.toFixed(1)} lost by waiting
                  </span>
                </div>
              </div>
            )}
          </div>

          <div className="panel">
            <h2>Ranked by value over next available</h2>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Pos</th>
                    <th className="num">Tier</th>
                    <th className="num">Left</th>
                    <th className="num">ADP</th>
                    <th className="num">Proj</th>
                    <th className="num">VONA</th>
                    <th className="num">Survives</th>
                    <th className="num">Bye</th>
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {recommendation.recommendations.map((p) => {
                    const left = (p as Player & { tier_left?: number }).tier_left ?? 0;
                    const conflicts =
                      (p as Player & { bye_conflicts?: number }).bye_conflicts ?? 0;
                    return (
                      <tr key={p.player_id}>
                        <td className="name">
                          {p.name} <NewsBadge player={p} />
                        </td>
                        <td>
                          <span className={`pos ${p.pos}`}>{p.pos}</span>
                        </td>
                        <td>{p.tier}</td>
                        <td className={left <= 2 ? "scarce" : ""}>{left}</td>
                        <td>{p.adp.toFixed(1)}</td>
                        <td>{p.proj_points.toFixed(0)}</td>
                        <td className="vona">{(p.vona ?? 0).toFixed(1)}</td>
                        <td>{((p.p_survives ?? 0) * 100).toFixed(0)}%</td>
                        <td className={conflicts ? "scarce" : ""}>{p.bye ?? "-"}</td>
                        <td>
                          <span style={{ display: "flex", gap: 4, justifyContent: "flex-end" }}>
                            <button
                              type="button"
                              title="Price in news the ADP snapshot has not caught"
                              onClick={() => setEditingNews(p.player_id)}
                            >
                              News
                            </button>
                            <button type="button" onClick={() => mark(p.player_id, "other")}>
                              Gone
                            </button>
                            <button
                              type="button"
                              className="mine"
                              onClick={() => mark(p.player_id, "me")}
                            >
                              Mine
                            </button>
                          </span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>

        <div className="rail-right">
          <div className="panel">
            <h2>Position urgency</h2>
            <div className="body urg">
              {recommendation.dropoff.map((d) => {
                const max = Math.max(...recommendation.dropoff.map((x) => x.dropoff), 1);
                const pct = Math.max(0, (d.dropoff / max) * 100);
                return (
                  <div className="urgrow" key={d.pos}>
                    <span className={`pos ${d.pos}`}>{d.pos}</span>
                    <span className="bar">
                      <i
                        style={{
                          width: `${pct}%`,
                          background: d.dropoff > 0 ? "var(--good)" : "var(--line)",
                        }}
                      />
                    </span>
                    <span className="muted">{d.dropoff.toFixed(1)}</span>
                  </div>
                );
              })}
              <div className="hint">
                What you lose by waiting one round. Zero means taking that position now is
                pure waste.
              </div>
            </div>
          </div>

          <div className="panel">
            <h2>My roster</h2>
            <div className="body">
              {recommendation.roster.length === 0 && <div className="empty">Empty.</div>}
              {recommendation.roster.map((r, i) => (
                <div className="slot" key={`${r.pos}-${i}`}>
                  <span className="lbl">{r.pos}</span>
                  <span>{r.proj_points.toFixed(0)}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="panel">
            <h2>Replacement level</h2>
            <div className="body">
              {Object.entries(recommendation.replacement)
                .sort(([a], [b]) => (a < b ? -1 : 1))
                .map(([pos, level]) => (
                  <div className="slot" key={pos}>
                    <span className="lbl">{pos}</span>
                    <span>{level.toFixed(1)}</span>
                  </div>
                ))}
              <div className="hint">
                ADP as of {league.adpAsOf ?? "unknown"}. Nothing from the last day of
                injury news is in here — tag a player under News &amp; risk to price it
                in yourself.
              </div>
            </div>
          </div>
        </div>
      </main>
    </>
  );
}
