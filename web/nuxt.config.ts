// A static single-page app (ADR 0010): `npm run build` writes .output/public, which
// `petroknowledge serve` serves beside the API. `npm run dev` serves it on 3000 and
// passes /api to a running `petroknowledge serve` (API_URL, default its port 8091).
import { defineNuxtConfig } from "nuxt/config";

export default defineNuxtConfig({
  compatibilityDate: "2026-08-05",
  ssr: false,
  devtools: { enabled: false },
  telemetry: false,
  css: ["~/assets/styles.css"],
  app: {
    head: {
      htmlAttrs: { lang: "en-GB" },
      title: "PetroKnowledge",
      meta: [
        { name: "color-scheme", content: "light dark" },
        {
          name: "description",
          content:
            "An independent assistant over public NNPC information, with the quote behind every statement.",
        },
      ],
      link: [{ rel: "icon", href: "/favicon.svg", type: "image/svg+xml" }],
    },
  },
  vite: {
    server: { proxy: { "/api": process.env.API_URL ?? "http://127.0.0.1:8091" } },
  },
});
