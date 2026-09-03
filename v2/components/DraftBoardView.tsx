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
const lockKey = (id: string) => `fantasy.v2.lock.${id}`;

/**
 * One number about a player, named once.
 *
 * The wide table and the phone card show the same seven quantities in the same
 * order; deriving both from this list is what stops them drifting apart.
 */
interface StatCell {
  key: string;
  label: string;
  value: string;
  className?: string;
}

const tierLeft = (p: Player) => (p as Player & { tier_left?: number }).tier_left ?? 0;
const byeConflicts = (p: Player) =>
  (p as Player & { bye_conflicts?: number }).bye_conflicts ?? 0;

/**
 * Whether this player's position has tiers worth printing.
 *
 * Kickers and defences each land in a tier of their own -- the fit genuinely
 * cannot pool them -- so the number is a rank in disguise and "1 left" reads as
 * scarcity where there is none. `untieredPositions` decides; here we just do
 * not print what it says is meaningless.
 */
const isTiered = (p: Player) => (p as Player & { tiered?: boolean }).tiered !== false;
const NO_TIER = "—";

const STAT_COLUMNS: {
  key: string;
  label: string;
  value: (p: Player) => string;
  className?: (p: Player) => string;
}[] = [
  { key: "tier", label: "Tier", value: (p) => (isTiered(p) ? String(p.tier ?? "-") : NO_TIER) },
  {
    key: "left",
    label: "Left",
    value: (p) => (isTiered(p) ? String(tierLeft(p)) : NO_TIER),
    className: (p) => (isTiered(p) && tierLeft(p) <= 2 ? "scarce" : ""),
  },
  { key: "adp", label: "ADP", value: (p) => p.adp.toFixed(1) },
  { key: "proj", label: "Proj", value: (p) => p.proj_points.toFixed(0) },
  // The measured spread of what players at this rank actually went on to
  // score. It is large -- around 100 points on the first running back -- and
  // printing the projection without it invites reading three significant
  // figures off a number that is an average over a decade of seasons.
  { key: "sd", label: "\u00b1", value: (p) => (p.sd ?? 0).toFixed(0), className: () => "spread" },
  { key: "vona", label: "VONA", value: (p) => (p.vona ?? 0).toFixed(1), className: () => "vona" },
  {
    key: "survives",
    label: "Survives",
    value: (p) => `${((p.p_survives ?? 0) * 100).toFixed(0)}%`,
  },
  {
    key: "bye",
    label: "Bye",
    value: (p) => String(p.bye ?? "-"),
    className: (p) => (byeConflicts(p) ? "scarce" : ""),
  },
];

const statsFor = (p: Player): StatCell[] =>
  STAT_COLUMNS.map((c) => ({
    key: c.key,
    label: c.label,
    value: c.value(p),
    className: c.className?.(p),
  }));

/**
 * Whether each call is available, and why not.
 *
 * The room cannot take a player on my pick and I cannot take one on theirs, so
 * with the order locked only one of the two buttons is live at a time.
 */
interface PickGate {
  gone: boolean;
  mine: boolean;
  reason: string | null;
}

function pickGate(locked: boolean, onMyPick: boolean, myNextPick: number | null): PickGate {
  if (!locked) return { gone: true, mine: true, reason: null };
  return {
    gone: !onMyPick,
    mine: onMyPick,
    reason: onMyPick
      ? "It is your pick. Untick “one call per pick” below to mark it for the room."
      : `Not your pick — you are next at ${myNextPick ?? "-"}. Untick “one call ` +
        "per pick” below to take him anyway.",
  };
}

/** The three buttons that follow a player everywhere he is offered. */
function PickButtons({
  playerId,
  gate,
  onNews,
  onMark,
}: {
  playerId: string;
  gate: PickGate;
  onNews: (id: string) => void;
  onMark: (id: string, takenBy: "me" | "other") => void;
}) {
  return (
    <>
      <button
        type="button"
        title="Price in news the ADP snapshot has not caught"
        onClick={() => onNews(playerId)}
      >
        News
      </button>
      <button
        type="button"
        disabled={!gate.gone}
        title={gate.gone ? undefined : (gate.reason ?? undefined)}
        onClick={() => onMark(playerId, "other")}
      >
        Gone
      </button>
      <button
        type="button"
        className="mine"
        disabled={!gate.mine}
        title={gate.mine ? undefined : (gate.reason ?? undefined)}
        onClick={() => onMark(playerId, "me")}
      >
        Mine
      </button>
    </>
  );
}

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
  /**
   * Which cards have their numbers open, on a screen too narrow for the table.
   * A phone shows the name and the two buttons that act on it; everything else
   * is one tap away rather than eight columns of horizontal scroll.
   */
  const [openStats, setOpenStats] = useState<Record<string, boolean>>({});
  const [allStats, setAllStats] = useState(false);
  /**
   * Only offer the call that the clock allows: "gone" when the room is
   * picking, "mine" when I am. Kept switchable, because the clock is derived
   * from a pick count and a slot -- if the room traded picks, or the app missed
   * one, the lock would be enforcing an order the room is not in.
   */
  const [lockOrder, setLockOrder] = useState(true);
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
      const savedLock = window.localStorage.getItem(lockKey(leagueId));
      if (savedLock !== null) setLockOrder(savedLock === "1");
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
      window.localStorage.setItem(lockKey(leagueId), lockOrder ? "1" : "0");
    } catch {
      /* storage blocked */
    }
  }, [leagueId, lockOrder]);

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

  const toggleStats = useCallback((playerId: string) => {
    setOpenStats((prev) => ({ ...prev, [playerId]: !prev[playerId] }));
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
      // Every path in -- keyboard, table, card, search result -- lands here, so
      // the lock is enforced once rather than at each button.
      if (lockOrder && recommendation) {
        const onMyPick = recommendation.my_next_pick === recommendation.on_the_clock;
        if ((takenBy === "me") !== onMyPick) return;
      }
      // `seq` is a storage key, not a pick number: it keeps climbing past an
      // undone pick so a retry of that pick still lands on the same row.
      const seq = Math.max(0, ...picks.map((p) => p.seq)) + 1;
      const pick: Pick = { seq, playerId, takenBy, voidedAt: null };
      setPicks((prev) => [...prev, pick]);
      pending.current.push(pick);
      void send(pick);
      setQuery("");
      // Refocusing pulls the on-screen keyboard back up on a phone, which
      // covers the board the moment a pick lands. Only do it where a physical
      // keyboard is what marked the pick.
      if (document.activeElement === searchRef.current) searchRef.current?.focus();
    },
    [picks, send, lockOrder, recommendation],
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
  const gate = pickGate(lockOrder, mine, recommendation.my_next_pick);

  return (
    <>
      <header className="app">
        <h1>{league.name}</h1>
        <div className="stats">
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
        </div>
        <div className="appctl">
          <label htmlFor="slot">Slot</label>
          <select id="slot" value={slot} onChange={(e) => setSlot(Number(e.target.value))}>
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
        <div className="col-search">
          <div className="panel">
            <h2>Mark a pick</h2>
            <div className="body">
              <input
                ref={searchRef}
                type="search"
                value={query}
                placeholder="Type a name…"
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key !== "Enter" || !results.length) return;
                  e.preventDefault();
                  mark(results[0].player_id, e.shiftKey ? "me" : "other");
                }}
              />
              <div className="hint kbdhint">
                <kbd>Enter</kbd> taken by someone else &middot;{" "}
                <kbd>Shift</kbd>+<kbd>Enter</kbd> taken by me
              </div>

              {results.map((p) => (
                <div className="slot result" key={p.player_id}>
                  <span>
                    <span className={`pos ${p.pos}`}>{p.pos}</span>{" "}
                    <span className="name">{p.name}</span>{" "}
                    <span className="muted">{p.tm}</span> <NewsBadge player={p} />
                  </span>
                  <span className="acts">
                    <PickButtons
                      playerId={p.player_id}
                      gate={gate}
                      onNews={setEditingNews}
                      onMark={mark}
                    />
                  </span>
                </div>
              ))}

              <div className="markrow">
                <button
                  type="button"
                  disabled={!gate.gone}
                  title={gate.gone ? undefined : (gate.reason ?? undefined)}
                  onClick={() => mark(null, "other")}
                >
                  Someone took a player not on the board
                </button>
                <button type="button" onClick={undoLast} disabled={!active.length}>
                  Undo
                </button>
              </div>

              <label className="checkline lock" htmlFor="lockorder">
                <input
                  id="lockorder"
                  type="checkbox"
                  checked={lockOrder}
                  onChange={(e) => setLockOrder(e.target.checked)}
                />
                One call per pick —{" "}
                {mine ? "yours, so only Mine" : "the room's, so only Gone"}
              </label>
            </div>
          </div>
        </div>

        <div className="col-main">
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
            <h2 className="panelhead">
              <span>Ranked by value over next available</span>
              <button
                type="button"
                className="statstoggle"
                aria-pressed={allStats}
                onClick={() => {
                  setAllStats((v) => !v);
                  setOpenStats({});
                }}
              >
                {allStats ? "Hide stats" : "All stats"}
              </button>
            </h2>
            <div className="tablewrap">
              <table>
                <thead>
                  <tr>
                    <th>Player</th>
                    <th>Pos</th>
                    {STAT_COLUMNS.map((c) => (
                      <th className="num" key={c.key}>
                        {c.label}
                      </th>
                    ))}
                    <th />
                  </tr>
                </thead>
                <tbody>
                  {recommendation.recommendations.map((p) => (
                    <tr key={p.player_id}>
                      <td className="name">
                        {p.name} <NewsBadge player={p} />
                      </td>
                      <td>
                        <span className={`pos ${p.pos}`}>{p.pos}</span>
                      </td>
                      {statsFor(p).map((s) => (
                        <td className={s.className} key={s.key}>
                          {s.value}
                        </td>
                      ))}
                      <td>
                        <span className="acts end">
                          <PickButtons
                            playerId={p.player_id}
                            gate={gate}
                            onNews={setEditingNews}
                            onMark={mark}
                          />
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* The same ranking for a phone: the name, the two calls on it, and
                the numbers behind a tap. Hidden wherever the table fits. */}
            <ul className="cards">
              {recommendation.recommendations.map((p, i) => {
                const open = allStats || Boolean(openStats[p.player_id]);
                const stats = statsFor(p);
                const summary = stats.filter((s) =>
                  ["vona", "tier", "left", "adp"].includes(s.key),
                );
                return (
                  <li className="card" key={p.player_id}>
                    <div className="cardhead">
                      <span className="rank">{i + 1}</span>
                      <span className={`pos ${p.pos}`}>{p.pos}</span>
                      <span className="cardname">
                        <span className="cardtitle">
                          <span className="name">{p.name}</span>{" "}
                          <span className="muted">{p.tm}</span> <NewsBadge player={p} />
                        </span>
                        <span className="cardmeta">
                          {summary.map((s, n) => (
                            <span key={s.key}>
                              {n > 0 && " · "}
                              {s.label} <b className={s.className}>{s.value}</b>
                            </span>
                          ))}
                        </span>
                      </span>
                    </div>

                    <div className="cardacts">
                      <button
                        type="button"
                        disabled={!gate.gone}
                        title={gate.gone ? undefined : (gate.reason ?? undefined)}
                        onClick={() => mark(p.player_id, "other")}
                      >
                        Gone
                      </button>
                      <button
                        type="button"
                        className="mine"
                        disabled={!gate.mine}
                        title={gate.mine ? undefined : (gate.reason ?? undefined)}
                        onClick={() => mark(p.player_id, "me")}
                      >
                        Mine
                      </button>
                      <button
                        type="button"
                        className="ghost"
                        aria-expanded={open}
                        onClick={() => toggleStats(p.player_id)}
                      >
                        {open ? "Less" : "Stats"}
                      </button>
                    </div>

                    {open && (
                      <div className="cardstats">
                        {stats.map((s) => (
                          <div className="cardstat" key={s.key}>
                            <span className="lbl">{s.label}</span>
                            <b className={s.className}>{s.value}</b>
                          </div>
                        ))}
                        <button
                          type="button"
                          className="cardnews"
                          onClick={() => setEditingNews(p.player_id)}
                        >
                          Price in news &amp; risk
                        </button>
                      </div>
                    )}
                  </li>
                );
              })}
            </ul>

            <div className="hint boardnote">
              Proj is read off a fitted curve of positional draft rank against
              what players at that rank have actually scored, so players the
              data cannot separate share a number — that is what a tier is.
              <b> &plusmn;</b> is the measured spread around it, and it is
              wide: treat the projection as the middle of a range, not a
              forecast.
            </div>
          </div>
        </div>

        <div className="rail-left">
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
              {active
                // The pick's number is its place in the draft, which is its
                // position in the live list. `seq` is a storage key and keeps
                // climbing past an undone pick, so showing it labelled the pick
                // after an undo with a number the draft had never reached.
                .map((p, i) => ({ pick: p, number: i + 1 }))
                .reverse()
                .slice(0, 12)
                .map(({ pick: p, number }) => {
                  const player = league.board.find((b) => b.player_id === p.playerId);
                  return (
                    <div className="slot" key={p.seq}>
                      <span className="lbl">{number}</span>
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
