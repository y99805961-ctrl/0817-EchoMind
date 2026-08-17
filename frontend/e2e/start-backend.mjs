/* global process */
import { spawn } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { resolve } from "node:path";

const repo = resolve(process.cwd(), "..");
const windowsPython = resolve(repo, ".venv", "Scripts", "python.exe");
const unixPython = resolve(repo, ".venv", "bin", "python");
const python = process.env.E2E_PYTHON || (existsSync(windowsPython) ? windowsPython : unixPython);
const live = process.env.E2E_LIVE === "1";
const envFile = resolve(repo, ".env");
const envText = existsSync(envFile) ? readFileSync(envFile, "utf8") : "";
const redisFromEnv = envText.match(/^REDIS_PASSWORD=(.*)$/m)?.[1]?.trim() || "echomind123";
const redisPassword = process.env.E2E_REDIS_PASSWORD || redisFromEnv;
const env = {
  ...process.env,
  PYTHONIOENCODING: "utf-8",
  ...(live ? (process.env.E2E_REDIS_URL ? { REDIS_URL: process.env.E2E_REDIS_URL } : {}) : {
    E2E_TEST_MODE: "1",
    REDIS_URL: process.env.E2E_REDIS_URL || `redis://:${encodeURIComponent(redisPassword)}@localhost:6379/15`,
    CHROMA_PERSIST_DIRECTORY: "data/eval/e2e/runtime/chroma",
    RAG_REWRITE_MAX_CONCURRENCY: "2",
  }),
};

const child = spawn(python, ["-m", "uvicorn", "api.main:app", "--host", "127.0.0.1", "--port", "8000"], {
  cwd: repo,
  env,
  stdio: "inherit",
});

const stop = () => {
  if (!child.killed) child.kill("SIGTERM");
};
process.on("SIGINT", stop);
process.on("SIGTERM", stop);
child.on("exit", (code, signal) => process.exit(code ?? (signal ? 1 : 0)));
