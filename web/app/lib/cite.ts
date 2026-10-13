import type { AnswerView } from "./api";
import { pageOf, quoted, readableDate } from "./format";

/** An answer as plain text to paste into a note or report: each claim with its
 * source numbers, then each source with its quote, place, dates and link. */
export function citationText(answer: AnswerView): string {
  const lines = answer.claims.map(
    (claim) => `${claim.text} ${claim.sources.map((number) => `[${number}]`).join("")}`,
  );
  lines.push("", "Sources");
  for (const source of answer.sources) {
    const page = pageOf(source.link);
    const place = [
      source.heading ? `${source.title} › ${source.heading}` : source.title,
      page === null ? "" : `p. ${page}`,
      source.published ? `published ${readableDate(source.published)}` : "",
      `retrieved ${readableDate(source.captured)}`,
    ].filter((part) => part !== "");
    lines.push(
      `[${source.number}] ${quoted(source.quote)} ${place.join(", ")}. ${source.link}`,
    );
  }
  lines.push("", "From PetroKnowledge, independent of NNPC Limited.");
  return lines.join("\n");
}
