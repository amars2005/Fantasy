"use client";

import { PRESETS, type NewsMap, type NewsTag, type PresetName } from "../lib/news";
import type { Player } from "../lib/types";

/** How much of his projection a tag has taken off, as a percentage. */
export function haircut(p: Player): number {
  const before = p.proj_before_news;
  if (!before || before <= 0) return 0;
  return Math.round((1 - p.proj_points / before) * 100);
}

/** The badge shown beside a tagged player's name, wherever he appears. */
export function NewsBadge({ player }: { player: Player }) {
  if (!player.news) return null;
  const cut = haircut(player);
  const title = [
    player.news.preset ? PRESETS[player.news.preset].label : "Adjusted",
    player.news.note,
    `${player.news.gamesMissed} games missed, slides ${player.news.adpShift} picks`,
  ]
    .filter(Boolean)
    .join(" — ");

  return (
    <span className={`newsbadge ${player.news.avoid ? "avoid" : ""}`} title={title}>
      {player.news.avoid ? "avoid" : `-${cut}%`}
    </span>
  );
}

interface Props {
  /** The board after adjustment, for names and the live readout. */
  players: Player[];
  news: NewsMap;
  editing: string | null;
  onEdit: (playerId: string | null) => void;
  /** A null tag clears the player. */
  onChange: (playerId: string, tag: NewsTag | null) => void;
}

export default function NewsPanel({ players, news, editing, onEdit, onChange }: Props) {
  const byId = (id: string) => players.find((p) => p.player_id === id);
  const tagged = Object.keys(news)
    .map((id) => byId(id))
    .filter((p): p is Player => Boolean(p));

  const target = editing ? byId(editing) : null;
  const tag = editing ? news[editing] : undefined;

  const set = (patch: Partial<NewsTag>) => {
    if (!editing) return;
    const base: NewsTag = tag ?? { gamesMissed: 0, adpShift: 0 };
    // Any hand edit drops the preset label: the numbers are yours now.
    const next = { ...base, ...patch };
    if (!("preset" in patch)) delete next.preset;
    onChange(editing, next);
  };

  const applyPreset = (name: PresetName) => {
    if (!editing) return;
    const { label: _label, rationale: _rationale, ...tagFields } = PRESETS[name];
    onChange(editing, { ...tagFields, preset: name, note: tag?.note });
  };

  return (
    <div className="panel">
      <h2>News &amp; risk</h2>
      <div className="body">
        {target && (
          <div className="newsedit">
            <div className="newshead">
              <span>
                <span className={`pos ${target.pos}`}>{target.pos}</span>{" "}
                <span className="name">{target.name}</span> <NewsBadge player={target} />
              </span>
              <button type="button" onClick={() => onEdit(null)}>
                Done
              </button>
            </div>

            <div className="presets">
              {(Object.keys(PRESETS) as PresetName[]).map((name) => (
                <button
                  type="button"
                  key={name}
                  className={tag?.preset === name ? "active" : ""}
                  title={PRESETS[name].rationale}
                  onClick={() => applyPreset(name)}
                >
                  {PRESETS[name].label}
                </button>
              ))}
            </div>

            <div className="grid2" style={{ marginTop: 12 }}>
              <div className="field">
                <label htmlFor="gamesMissed">Games missed</label>
                <input
                  id="gamesMissed"
                  type="number"
                  min={0}
                  max={17}
                  step={0.5}
                  value={tag?.gamesMissed ?? 0}
                  onChange={(e) => set({ gamesMissed: Number(e.target.value) })}
                />
              </div>
              <div className="field">
                <label htmlFor="adpShift">Slides (picks)</label>
                <input
                  id="adpShift"
                  type="number"
                  min={0}
                  max={200}
                  step={1}
                  value={tag?.adpShift ?? 0}
                  onChange={(e) => set({ adpShift: Number(e.target.value) })}
                />
              </div>
            </div>

            <div className="field">
              <label htmlFor="newsnote">Note</label>
              <input
                id="newsnote"
                type="text"
                placeholder="why, for three rounds from now"
                value={tag?.note ?? ""}
                onChange={(e) => set({ note: e.target.value })}
              />
            </div>

            <label className="checkline" htmlFor="avoid">
              <input
                id="avoid"
                type="checkbox"
                checked={Boolean(tag?.avoid)}
                onChange={(e) => set({ avoid: e.target.checked })}
              />
              Do not draft at any price
            </label>

            <div className="newsread">
              {tag?.avoid ? (
                <>
                  Valued at zero, still in the room&rsquo;s simulation at pick{" "}
                  <b>{Math.round(target.adp + (tag?.adpShift ?? 0))}</b>.
                </>
              ) : (
                <>
                  <b>{haircut(target)}%</b> off his projection —{" "}
                  {target.proj_before_news?.toFixed(0) ?? target.proj_points.toFixed(0)} to{" "}
                  <b>{target.proj_points.toFixed(0)}</b>. Expected to be taken around pick{" "}
                  <b>{Math.round(target.adp + (tag?.adpShift ?? 0))}</b> rather than{" "}
                  {target.adp.toFixed(0)}.
                </>
              )}
            </div>

            <button type="button" onClick={() => onChange(editing!, null)}>
              Clear this player
            </button>
          </div>
        )}

        {tagged.length === 0 && !target && (
          <div className="empty">
            Nothing tagged. Search a player and press News when the room knows something
            the ADP snapshot does not.
          </div>
        )}

        {tagged
          .filter((p) => p.player_id !== editing)
          .map((p) => (
            <div className="slot" key={p.player_id}>
              <span style={{ flex: 1, textAlign: "left" }}>
                <span className={`pos ${p.pos}`}>{p.pos}</span>{" "}
                <span className="name">{p.name}</span> <NewsBadge player={p} />
              </span>
              <span style={{ display: "flex", gap: 4 }}>
                <button type="button" onClick={() => onEdit(p.player_id)}>
                  Edit
                </button>
                <button type="button" onClick={() => onChange(p.player_id, null)}>
                  Clear
                </button>
              </span>
            </div>
          ))}

        <div className="hint">
          A tagged player is re-priced, never hidden. He stays in the survival
          simulation so the room still spends a pick on him — dropping him would make
          everyone behind him look scarcer than he is.
        </div>
      </div>
    </div>
  );
}
