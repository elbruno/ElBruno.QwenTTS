// Extension: qwen-tts-studio
//
// Canvas for testing Qwen3-TTS audio generation: type text, pick a voice, play
// the result in a standard audio player, and download the WAV.
//
// The heavy lifting runs in a .NET sidecar (`tts-host.cs`) that keeps one
// TtsPipeline warm in memory, so the ~5.5 GB ONNX model is loaded once per
// extension lifetime instead of once per generation. This file only wires the
// canvas: it boots the sidecar, serves `ui.html`, and proxies `/api/*` to it.

import { spawn } from "node:child_process";
import { once } from "node:events";
import { mkdir, readFile } from "node:fs/promises";
import { createServer } from "node:http";
import { createServer as createProbeServer } from "node:net";
import { homedir } from "node:os";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { joinSession, createCanvas, CanvasError } from "@github/copilot-sdk/extension";

const EXTENSION_DIR = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(EXTENSION_DIR, "..", "..", "..");
const HOST_PROJECT = path.join(EXTENSION_DIR, "tts-host.csproj");

// Generated WAVs outlive any single panel or session, so they go in the
// extension's own durable artifact directory rather than in the repo.
const COPILOT_HOME = process.env.COPILOT_HOME || path.join(homedir(), ".copilot");
const ARTIFACTS_DIR = path.join(COPILOT_HOME, "extensions", "qwen-tts-studio", "artifacts");

/** @type {import("@github/copilot-sdk/extension").CopilotSession | undefined} */
let session;

/** @type {{ port: number, child: import("node:child_process").ChildProcess } | null} */
let sidecar = null;
/** @type {Promise<{ port: number }> | null} */
let sidecarBoot = null;
let hostOutputTail = "";

/** One loopback renderer server per open canvas instance. */
const servers = new Map();

async function findFreePort() {
    const probe = createProbeServer();
    probe.listen(0, "127.0.0.1");
    await once(probe, "listening");
    const { port } = probe.address();
    await new Promise((resolve) => probe.close(resolve));
    return port;
}

async function waitForSidecar(port, child, timeoutMs = 600_000) {
    const deadline = Date.now() + timeoutMs;
    while (Date.now() < deadline) {
        if (child.exitCode !== null) {
            throw new Error(`TTS host exited with code ${child.exitCode}. ${hostOutputTail}`.trim());
        }
        try {
            const res = await fetch(`http://127.0.0.1:${port}/api/state`);
            if (res.ok) return;
        } catch {
            // Not listening yet — the first launch also restores and builds the sidecar.
        }
        await new Promise((resolve) => setTimeout(resolve, 500));
    }
    throw new Error("Timed out waiting for the TTS host to start.");
}

/** Boots the .NET sidecar once and reuses it for the extension's lifetime. */
function ensureSidecar() {
    sidecarBoot ??= (async () => {
        await mkdir(ARTIFACTS_DIR, { recursive: true });
        const port = await findFreePort();

        session?.log("Starting the Qwen TTS host (the first run restores and builds the sidecar)...", {
            level: "info",
            ephemeral: true,
        });

        const child = spawn(
            "dotnet",
            ["run", "--project", HOST_PROJECT, "--no-launch-profile", "--", "--port", String(port), "--artifacts", ARTIFACTS_DIR, "--parent-pid", String(process.pid)],
            {
                cwd: REPO_ROOT,
                stdio: ["ignore", "pipe", "pipe"],
                env: { ...process.env, DOTNET_NOLOGO: "1" },
                windowsHide: true,
            },
        );

        // Never forward child output to this process's stdout — it is reserved
        // for JSON-RPC. Keep only a tail, for error reporting.
        const capture = (chunk) => {
            hostOutputTail = `${hostOutputTail}${chunk}`.slice(-2000);
        };
        child.stdout.on("data", capture);
        child.stderr.on("data", capture);
        child.on("exit", () => {
            sidecar = null;
            sidecarBoot = null;
        });

        try {
            await waitForSidecar(port, child);
        } catch (error) {
            child.kill();
            sidecar = null;
            sidecarBoot = null;
            throw error;
        }

        sidecar = { port, child };
        return { port };
    })();

    return sidecarBoot;
}

async function readBody(req) {
    const chunks = [];
    for await (const chunk of req) chunks.push(chunk);
    return Buffer.concat(chunks);
}

async function proxyToSidecar(req, res) {
    if (!sidecar) {
        res.writeHead(503, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: "The TTS host is still starting." }));
        return;
    }

    const hasBody = req.method !== "GET" && req.method !== "HEAD";

    try {
        const upstream = await fetch(`http://127.0.0.1:${sidecar.port}${req.url}`, {
            method: req.method,
            headers: { "Content-Type": req.headers["content-type"] || "application/json" },
            body: hasBody ? await readBody(req) : undefined,
        });

        const payload = Buffer.from(await upstream.arrayBuffer());
        res.writeHead(upstream.status, {
            "Content-Type": upstream.headers.get("content-type") || "application/json",
            "Content-Length": payload.length,
            "Cache-Control": "no-store",
        });
        res.end(payload);
    } catch (error) {
        res.writeHead(502, { "Content-Type": "application/json" });
        res.end(JSON.stringify({ error: `TTS host unreachable: ${error.message}` }));
    }
}

async function startRendererServer() {
    const uiPath = path.join(EXTENSION_DIR, "ui.html");

    const server = createServer((req, res) => {
        if (req.url?.startsWith("/api/")) {
            void proxyToSidecar(req, res);
            return;
        }
        readFile(uiPath).then(
            (html) => {
                res.writeHead(200, { "Content-Type": "text/html; charset=utf-8", "Cache-Control": "no-store" });
                res.end(html);
            },
            (error) => {
                res.writeHead(500, { "Content-Type": "text/plain; charset=utf-8" });
                res.end(`Unable to read ui.html: ${error.message}`);
            },
        );
    });

    server.listen(0, "127.0.0.1");
    await once(server, "listening");
    return { server, url: `http://127.0.0.1:${server.address().port}/` };
}

/** Calls the sidecar API on behalf of an agent action. */
async function callHost(pathname, init) {
    try {
        await ensureSidecar();
    } catch (error) {
        throw new CanvasError("tts_host_unavailable", error.message);
    }

    const res = await fetch(`http://127.0.0.1:${sidecar.port}${pathname}`, {
        headers: { "Content-Type": "application/json" },
        ...init,
    });
    const body = await res.json().catch(() => ({}));
    if (!res.ok) throw new CanvasError("tts_host_error", body.error || `Request failed (${res.status})`);
    return body;
}

const canvas = createCanvas({
    id: "qwen-tts-studio",
    displayName: "Qwen TTS Studio",
    description:
        "Type text, pick a Qwen3-TTS voice, generate speech locally, play it in an audio player, and download the WAV.",
    inputSchema: {
        type: "object",
        properties: {
            text: { type: "string", description: "Text to pre-fill the canvas with." },
            speaker: { type: "string", description: "Voice to pre-select, e.g. ryan or serena." },
            language: { type: "string", description: "Language to pre-select, e.g. english or spanish." },
        },
        additionalProperties: false,
    },
    actions: [
        {
            name: "get_status",
            description: "Report whether the local TTS model is loaded, and which voices and languages are available.",
            handler: async () => callHost("/api/state"),
        },
        {
            name: "generate_speech",
            description:
                "Start a speech generation on the canvas. Returns immediately with a job id; synthesis runs in the background.",
            inputSchema: {
                type: "object",
                properties: {
                    text: { type: "string", description: "Text to synthesize (max 10,000 characters)." },
                    speaker: { type: "string", description: "Voice name, e.g. ryan." },
                    language: { type: "string", description: "Language, e.g. english. Defaults to auto." },
                },
                required: ["text"],
                additionalProperties: false,
            },
            handler: async (ctx) =>
                callHost("/api/generate", { method: "POST", body: JSON.stringify(ctx.input ?? {}) }),
        },
        {
            name: "get_generation",
            description: "Check the status, progress and result of a generation job started on this canvas.",
            inputSchema: {
                type: "object",
                properties: { jobId: { type: "string", description: "Job id returned by generate_speech." } },
                required: ["jobId"],
                additionalProperties: false,
            },
            handler: async (ctx) => callHost(`/api/jobs/${encodeURIComponent(ctx.input.jobId)}`),
        },
        {
            name: "list_generations",
            description: "List recent generations produced through this canvas, including their saved WAV file names.",
            handler: async () => ({ artifactsDir: ARTIFACTS_DIR, generations: await callHost("/api/history") }),
        },
    ],
    open: async (ctx) => {
        // Idempotent: the same instanceId can arrive again after a host re-open
        // or an extensions reload, in which case we reuse the running server.
        let entry = servers.get(ctx.instanceId);
        if (!entry) {
            entry = await startRendererServer();
            servers.set(ctx.instanceId, entry);
        }

        // Boot the model host in the background so `open` returns immediately.
        // The UI polls /api/state and reports loading progress itself.
        ensureSidecar().catch((error) => {
            session?.log(`Qwen TTS host failed to start: ${error.message}`, { level: "error" });
        });

        const prefill = new URLSearchParams();
        for (const key of ["text", "speaker", "language"]) {
            if (ctx.input?.[key]) prefill.set(key, ctx.input[key]);
        }
        const query = prefill.toString();

        return {
            title: "Qwen TTS Studio",
            status: sidecar ? "Model host running" : "Starting model host…",
            url: query ? `${entry.url}?${query}` : entry.url,
        };
    },
    onClose: async (ctx) => {
        const entry = servers.get(ctx.instanceId);
        if (!entry) return;
        servers.delete(ctx.instanceId);
        await new Promise((resolve) => entry.server.close(() => resolve()));
        // The sidecar deliberately stays alive so the warm model survives a
        // close/re-open cycle; it is torn down when the extension exits.
    },
});

session = await joinSession({ canvases: [canvas] });

const shutdown = () => {
    sidecar?.child.kill();
    sidecar = null;
};
process.on("exit", shutdown);
process.on("SIGTERM", shutdown);
process.on("SIGINT", shutdown);
