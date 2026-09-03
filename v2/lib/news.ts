/**
 * Pricing the news that ADP has not caught up to yet.
 *
 * The board is anchored to a consensus ADP snapshot, and a snapshot is always
 * at least a day stale. When a headline lands the morning of a draft it breaks
 * the board in two independent places, and fixing only one of them leaves you
 * worse off than fixing neither:
 *
 *   *Value.* `projectPlayers` prices a player off a full-season curve fitted at
 *   his positional ADP rank. The curve knows what the third running back off
 *   the board historically scored; it does not know this one may not be on the
 *   field for six of those weeks.
 *
 *   *The market.* `vonaTable` simulates who survives to your next pick from
 *   `adp_mu`. Everyone else in the room read the same headline you did, so the
 *   pick he actually goes at has moved and the calibrated mean has not. Left
 *   alone, the simulation believes he is a now-or-never pick and that the
 *   players behind him will be picked cleaner than they will be.
 *
 * The two are corrected separately because they fail separately. A player can
 * be worth less without sliding (everyone already knew) or slide without being
 * worth less (the room is overreacting -- which is when you want him).
 *
 * What this deliberately does NOT do is remove anyone from the board. Dropping
 * a player would take his picks away from the room inside the survival
 * simulation, hand them to the players behind him, and make every one of those
 * look scarcer than he really is. One bad headline would quietly corrupt a
 * whole position. He stays on the board, priced at what you think he is worth.
 */

import { roundTo } from "./bundle";
import type { Player } from "./types";

/** Nobody plays more than 17, and it is what we assume when we cannot tell. */
const FULL_SEASON = 17;

/** What you know about a player that the ADP snapshot does not. */
export interface NewsTag {
  /**
   * Expected games missed. Fractional on purpose: this is an expectation, not
   * a forecast. A coin-flip on a six-game suspension is 3, not 6.
   */
  gamesMissed: number;
  /**
   * How many picks later than his calibrated mean the room will actually take
   * him. Your estimate of how hard the news has moved everyone else.
   */
  adpShift: number;
  /** Refuse him at any price. Zeroes the value; the slide still applies. */
  avoid?: boolean;
  /** Why, for your own benefit three rounds later. */
  note?: string;
  /** Which preset this came from, if any. */
  preset?: PresetName;
}

/** player_id -> what you know about him. */
export type NewsMap = Record<string, NewsTag>;

export type PresetName = "dtd" | "risk" | "out" | "avoid";

export interface Preset extends NewsTag {
  label: string;
  /** What the preset is assuming, shown so you can disagree with it. */
  rationale: string;
}

/**
 * Starting points, not verdicts.
 *
 * Every one of these is a guess about a specific player dressed up as a
 * category, which is why all of them are editable. The slides are deliberately
 * larger than the games lost: draft rooms overreact to news far more than the
 * expected games justify, and that overreaction is an opportunity, not a
 * correction to copy.
 */
export const PRESETS: Record<PresetName, Preset> = {
  dtd: {
    label: "Day-to-day",
    gamesMissed: 1,
    adpShift: 6,
    rationale: "A knock that costs about a game. The room shrugs and moves half a round.",
  },
  risk: {
    label: "Suspension risk",
    gamesMissed: 6,
    adpShift: 18,
    rationale:
      "Discipline is pending and its length is unknown. Weight this yourself: " +
      "a charge is not a suspension, and the games here should be your probability " +
      "times the ban you expect, not the ban itself.",
  },
  out: {
    label: "Out",
    gamesMissed: 10,
    adpShift: 24,
    rationale: "A known absence measured in months rather than weeks.",
  },
  avoid: {
    label: "Do not draft",
    gamesMissed: 0,
    adpShift: 36,
    avoid: true,
    rationale:
      "Off your board at any price. He stays in the simulation so the room " +
      "still spends a pick on him, which keeps everyone else's numbers honest.",
  },
};

/** A tag carrying nothing but its defaults -- i.e. one worth dropping. */
export function isEmptyTag(tag: NewsTag | undefined): boolean {
  if (!tag) return true;
  return !tag.avoid && tag.gamesMissed === 0 && tag.adpShift === 0 && !tag.note;
}

/**
 * The share of his projected season a tagged player is still expected to play.
 *
 * Denominated in his *own* fitted games rather than a flat 17, so missing four
 * costs a workhorse less than it costs someone already priced to split carries.
 */
function availability(player: Player, tag: NewsTag): number {
  if (tag.avoid) return 0;
  const season = player.games > 0 ? player.games : FULL_SEASON;
  const missed = Math.max(0, tag.gamesMissed || 0);
  return Math.min(1, Math.max(0, (season - missed) / season));
}

/** One rung of a position's tier ladder: the tier, and the points it is worth. */
interface TierStep {
  tier: number;
  points: number;
}

/**
 * The tier ladder per position, as the market had it.
 *
 * Built from the untagged board on purpose. Tiers are plateaus in the fitted
 * curve, so one tagged player's re-priced number must not become a rung that
 * another tagged player can be placed on.
 */
function tierLadders(players: Player[]): Map<string, TierStep[]> {
  const ladders = new Map<string, TierStep[]>();
  for (const p of players) {
    if (p.tier === undefined || p.tier_points === undefined) continue;
    let ladder = ladders.get(p.pos);
    if (!ladder) {
      ladder = [];
      ladders.set(p.pos, ladder);
    }
    if (!ladder.some((step) => step.tier === p.tier)) {
      ladder.push({ tier: p.tier, points: p.tier_points });
    }
  }
  for (const ladder of ladders.values()) ladder.sort((a, b) => a.tier - b.tier);
  return ladders;
}

/**
 * Re-price a board against what you know.
 *
 * Pure, and cheap enough to run on every keystroke: untagged players are passed
 * through by reference, and an empty map returns the board it was handed.
 */
export function applyNews(players: Player[], news: NewsMap): Player[] {
  if (!news || Object.keys(news).length === 0) return players;

  const ladders = tierLadders(players);

  const repriced = players.map((p) => {
    const tag = news[p.player_id];
    if (!tag || isEmptyTag(tag)) return p;

    const share = availability(p, tag);
    const projRaw = (p.proj_raw ?? p.proj_points) * share;
    const points = roundTo(projRaw, 1);
    const gamesRaw = Math.max(0, (p.games_raw ?? p.games) - (tag.avoid ? Infinity : tag.gamesMissed || 0));

    // Only the latent mean moves. `adp` stays as the market reported it: the
    // slide is our estimate of where he will go, not a number anyone published,
    // and conflating the two would hide which is which on the board.
    const mu = Number.isFinite(p.adp_mu) ? p.adp_mu : p.adp;

    // Re-price him into the tier his new projection earns.
    //
    // A tier is the answer to "is he the last of his kind", and leaving it
    // alone made that answer a lie in the one case it is most costly: tag a
    // receiver as out half the season and he kept his tier 1 badge, with the
    // board still reporting one left in the tier -- reach now -- about a player
    // you had just written down.
    //
    // The rung is looked up rather than recomputed. Re-running the tiering on
    // adjusted points would insert a new distinct value into the position and
    // renumber every tier below it, so tagging one receiver would silently
    // shift the tier printed against all the others.
    const ladder = ladders.get(p.pos);
    const step = ladder?.find((rung) => rung.points <= points) ?? ladder?.[ladder.length - 1];

    return {
      ...p,
      proj_points: points,
      proj_raw: projRaw,
      games: roundTo(gamesRaw, 1),
      games_raw: gamesRaw,
      adp_mu: mu + (tag.adpShift || 0),
      tier: step?.tier ?? p.tier,
      tier_points: step?.points ?? p.tier_points,
      news: tag,
      proj_before_news: p.proj_points,
    };
  });

  // `tier_size` is the count of a tier's members, and players have just moved
  // between tiers.
  const sizes = new Map<string, number>();
  for (const p of repriced) {
    const key = `${p.pos}|${p.tier}`;
    sizes.set(key, (sizes.get(key) ?? 0) + 1);
  }
  return repriced.map((p) => {
    const size = sizes.get(`${p.pos}|${p.tier}`);
    return size === p.tier_size ? p : { ...p, tier_size: size };
  });
}
