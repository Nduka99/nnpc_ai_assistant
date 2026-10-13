// Source links go only to the hosts the sources come from: every company registry's
// `allowed_hosts` (companies/*/sources.json), subdomains included.
const ALLOWED_HOSTS = ["nnpcgroup.com", "en.wikipedia.org", "nuprc.gov.ng", "eia.gov"];

/** The link to show for a source address, or null when it is not on an allowed
 * host over HTTP(S), in which case the title is shown as plain text. */
export function safeLink(address: string): string | null {
  let url: URL;
  try {
    url = new URL(address);
  } catch {
    return null;
  }
  const web = url.protocol === "https:" || url.protocol === "http:";
  const allowed = ALLOWED_HOSTS.some(
    (host) => url.hostname === host || url.hostname.endsWith(`.${host}`),
  );
  return web && allowed ? url.href : null;
}
