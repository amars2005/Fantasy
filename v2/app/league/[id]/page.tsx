import DraftBoardView from "../../../components/DraftBoardView";

export default async function LeaguePage({
  params,
}: {
  params: Promise<{ id: string }>;
}) {
  const { id } = await params;
  // The board is fetched client-side rather than server-rendered: it is the
  // same payload the offline cache holds, and the draft has to keep working
  // when the network does not.
  return <DraftBoardView leagueId={id} />;
}
