// How dates and document places read on the page: in the reader's words, not in the
// data's formats (Nielsen 2, match with the real world).

// A fixed table, not Intl: browsers' locale data differ ("Sep" or "Sept"), and a
// date must read the same everywhere it is quoted.
const MONTHS = [
  "Jan",
  "Feb",
  "Mar",
  "Apr",
  "May",
  "Jun",
  "Jul",
  "Aug",
  "Sep",
  "Oct",
  "Nov",
  "Dec",
];

/** "2026-09-29" as "29 Sep 2026"; anything that is not a YYYY-MM-DD date as it is. */
export function readableDate(iso: string): string {
  const parts = /^(\d{4})-(\d{2})-(\d{2})$/.exec(iso);
  if (parts === null) return iso;
  const [, year, month, day] = parts;
  return `${Number(day)} ${MONTHS[Number(month) - 1]} ${year}`;
}

/** Whether a quote opens with its own quotation mark. */
export function hasOwnMark(quote: string): boolean {
  return /^[“"‘']/.test(quote);
}

/** A quote in quotation marks, adding none to a quote that brings its own. */
export function quoted(quote: string): string {
  return hasOwnMark(quote) ? quote : `“${quote}”`;
}

/** The PDF page a link opens at (`#page=12`), or null for any other link. */
export function pageOf(link: string): number | null {
  const found = /#page=(\d+)/.exec(link);
  return found ? Number(found[1]) : null;
}
