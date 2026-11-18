// Proxy HLS pentru TV România Live — Cloudflare Worker (gratuit)
// Rezolvă: CORS, stream-uri http pe pagini https, Referer / User-Agent cerute de unele canale.
//
// IMPORTANT: pune aici adresa site-ului tău, ca proxy-ul să nu poată fi folosit de alții.
// "null" = fișierul deschis local de pe calculator (file://).
const ALLOWED_ORIGINS = [
  "https://v3ng3ance.github.io",
  "null",
];

const CORS = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "GET, HEAD, OPTIONS",
  "Access-Control-Expose-Headers": "*",
};

const DEFAULT_UA =
  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36";

export default {
  async fetch(req) {
    if (req.method === "OPTIONS") return new Response(null, { headers: CORS });

    const origin = req.headers.get("Origin");
    if (origin && !ALLOWED_ORIGINS.includes(origin)) {
      return new Response("Origin nepermis", { status: 403, headers: CORS });
    }

    const self = new URL(req.url);
    const target = self.searchParams.get("url");
    if (!target) return new Response("Proxy TV funcționează", { headers: CORS });

    let t;
    try { t = new URL(target); } catch { return new Response("URL invalid", { status: 400, headers: CORS }); }
    if (t.protocol !== "http:" && t.protocol !== "https:") {
      return new Response("Protocol nepermis", { status: 400, headers: CORS });
    }

    const ref = self.searchParams.get("ref");
    const ua = self.searchParams.get("ua");
    const h = new Headers({ "User-Agent": ua || DEFAULT_UA, "Accept": "*/*" });
    if (ref) {
      h.set("Referer", ref);
      try { h.set("Origin", new URL(ref).origin); } catch {}
    }
    const range = req.headers.get("Range");
    if (range) h.set("Range", range);

    let r;
    try {
      r = await fetch(t.href, { headers: h, redirect: "follow" });
    } catch (e) {
      return new Response("Sursa nu răspunde: " + e.message, { status: 502, headers: CORS });
    }

    const ct = r.headers.get("content-type") || "";
    const path = new URL(r.url).pathname;
    const isMedia = /^(video|audio|image)\//i.test(ct) || /\.(ts|aac|mp4|m4s|m4a|m4v|vtt|key)$/i.test(path);
    const len = +(r.headers.get("content-length") || 0);
    // Unele canale (ex. PRO TV) au linkuri fără .m3u8 și content-type greșit:
    // tot ce nu e clar media și e mic îl verificăm după conținut (#EXTM3U).
    const looksPlaylist = /mpegurl/i.test(ct) || /\.m3u8?$/i.test(path) || (!isMedia && (!len || len < 2_000_000));

    if (looksPlaylist && r.ok) {
      const bytes = await r.arrayBuffer();
      const text = new TextDecoder().decode(bytes);
      if (text.replace(/^﻿/, "").trimStart().startsWith("#EXTM3U")) {
        const base = r.url;
        const wrap = (s) => {
          let p = self.origin + "/?url=" + encodeURIComponent(new URL(s, base).href);
          if (ref) p += "&ref=" + encodeURIComponent(ref);
          if (ua) p += "&ua=" + encodeURIComponent(ua);
          return p;
        };
        const out = text.split(/\r?\n/).map((line) => {
          const l = line.trim();
          if (!l) return line;
          if (l.startsWith("#")) return line.replace(/URI="([^"]+)"/g, (_, s) => `URI="${wrap(s)}"`);
          return wrap(l);
        }).join("\n");
        return new Response(out, {
          status: r.status,
          headers: { ...CORS, "Content-Type": "application/vnd.apple.mpegurl", "Cache-Control": "no-cache" },
        });
      }
      // nu e playlist: trimitem octeții originali, neatinși
      const out = new Headers(r.headers);
      for (const [k, v] of Object.entries(CORS)) out.set(k, v);
      out.delete("set-cookie"); out.delete("content-encoding"); out.delete("content-length");
      return new Response(bytes, { status: r.status, headers: out });
    }

    const out = new Headers(r.headers);
    for (const [k, v] of Object.entries(CORS)) out.set(k, v);
    out.delete("set-cookie");
    return new Response(r.body, { status: r.status, headers: out });
  },
};
