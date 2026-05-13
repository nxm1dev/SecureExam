import { ChildProcess, spawn, spawnSync } from "child_process";
import { app } from "electron";
import * as fs from "fs";
import * as path from "path";
import axios from "axios";

import { AI_SERVICE_PORT, BACKEND_PORT, SERVICE_URLS } from "./service-urls";

type PythonLauncher = {
  command: string;
  args: string[];
  description: string;
};

type ManagedService = {
  name: "backend" | "ai-service";
  process: ChildProcess;
};

const HEALTH_TIMEOUT_MS = 60_000;
const HEALTH_POLL_MS = 1_000;
const MAX_LOG_LINES = 200;

const managedServices: ManagedService[] = [];
const serviceLogs = new Map<string, string[]>();

function pushServiceLog(serviceName: string, chunk: Buffer | string) {
  const lines = serviceLogs.get(serviceName) ?? [];
  const text = chunk.toString().trim();
  if (!text) {
    return;
  }

  lines.push(text);
  if (lines.length > MAX_LOG_LINES) {
    lines.splice(0, lines.length - MAX_LOG_LINES);
  }
  serviceLogs.set(serviceName, lines);
}

function getServiceLogTail(serviceName: string) {
  return (serviceLogs.get(serviceName) ?? []).slice(-20).join("\n");
}

function getPythonCandidates(): PythonLauncher[] {
  const configuredPython = process.env.EXAMAC_PYTHON;
  const candidates: PythonLauncher[] = [];

  if (configuredPython) {
    candidates.push({
      command: configuredPython,
      args: [],
      description: configuredPython,
    });
  }

  candidates.push(
    { command: "py", args: ["-3.12"], description: "py -3.12" },
    { command: "py", args: ["-3.11"], description: "py -3.11" },
    { command: "py", args: ["-3"], description: "py -3" },
    { command: "python", args: [], description: "python" },
    { command: "python3", args: [], description: "python3" }
  );

  return candidates;
}

function resolvePythonLauncher(): PythonLauncher {
  const versionProbe =
    "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')";

  for (const candidate of getPythonCandidates()) {
    try {
      const result = spawnSync(
        candidate.command,
        [...candidate.args, "-c", versionProbe],
        {
          encoding: "utf8",
          windowsHide: true,
        }
      );

      if (result.status !== 0) {
        continue;
      }

      const version = (result.stdout || "").trim();
      const [majorText, minorText] = version.split(".");
      const major = Number(majorText);
      const minor = Number(minorText);
      if (major > 3 || (major === 3 && minor >= 11)) {
        return candidate;
      }
    } catch {
      // Try the next launcher candidate.
    }
  }

  throw new Error(
    "Khong tim thay Python 3.11+ tren may. Cai Python 3.11 hoac 3.12 va dat vao PATH."
  );
}

function getRuntimeRoot() {
  return path.join(process.resourcesPath, "runtime");
}

function ensureDirectory(dirPath: string) {
  fs.mkdirSync(dirPath, { recursive: true });
}

function toSqliteUrl(filePath: string) {
  return `sqlite+aiosqlite:///${filePath.replace(/\\/g, "/")}`;
}

async function waitForHealth(
  serviceName: string,
  healthUrl: string,
  child?: ChildProcess
) {
  const start = Date.now();

  while (Date.now() - start < HEALTH_TIMEOUT_MS) {
    if (child && child.exitCode !== null) {
      const tail = getServiceLogTail(serviceName);
      throw new Error(
        `${serviceName} da dung som voi ma ${child.exitCode}.\n${tail}`
      );
    }

    try {
      const response = await axios.get(healthUrl, { timeout: 1_500 });
      if (response.status === 200) {
        return;
      }
    } catch {
      // Keep polling.
    }

    await new Promise((resolve) => setTimeout(resolve, HEALTH_POLL_MS));
  }

  throw new Error(
    `Het thoi gian doi ${serviceName} khoi dong.\n${getServiceLogTail(serviceName)}`
  );
}

async function isHealthy(url: string) {
  try {
    const response = await axios.get(url, { timeout: 1_000 });
    return response.status === 200;
  } catch {
    return false;
  }
}

function spawnService(
  serviceName: "backend" | "ai-service",
  python: PythonLauncher,
  cwd: string,
  port: number,
  extraEnv: Record<string, string>
) {
  const child = spawn(
    python.command,
    [...python.args, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", String(port)],
    {
      cwd,
      env: {
        ...process.env,
        PYTHONUNBUFFERED: "1",
        ...extraEnv,
      },
      windowsHide: true,
      stdio: ["ignore", "pipe", "pipe"],
    }
  );

  child.stdout?.on("data", (chunk) => {
    pushServiceLog(serviceName, chunk);
    console.log(`[${serviceName}] ${chunk.toString().trimEnd()}`);
  });

  child.stderr?.on("data", (chunk) => {
    pushServiceLog(serviceName, chunk);
    console.error(`[${serviceName}] ${chunk.toString().trimEnd()}`);
  });

  managedServices.push({ name: serviceName, process: child });
  return child;
}

export async function ensurePackagedServicesReady() {
  if (!app.isPackaged) {
    return;
  }

  const backendHealthy = await isHealthy(`${SERVICE_URLS.backend}/health`);
  const aiHealthy = await isHealthy(`${SERVICE_URLS.ai}/health`);

  if (backendHealthy && aiHealthy) {
    return;
  }

  const python = resolvePythonLauncher();
  const runtimeRoot = getRuntimeRoot();
  const backendDir = path.join(runtimeRoot, "backend");
  const aiServiceDir = path.join(runtimeRoot, "ai-service");
  const configDir = path.join(runtimeRoot, "config");
  const dataRoot = path.join(app.getPath("userData"), "runtime");
  const dbPath = path.join(dataRoot, "examac.db");
  const modelCacheDir = path.join(dataRoot, "ai-model-cache");

  ensureDirectory(dataRoot);
  ensureDirectory(modelCacheDir);

  if (!fs.existsSync(backendDir) || !fs.existsSync(aiServiceDir) || !fs.existsSync(configDir)) {
    throw new Error(
      "Khong tim thay backend, ai-service hoac config trong goi cai dat."
    );
  }

  if (!backendHealthy) {
    const backendProcess = spawnService("backend", python, backendDir, BACKEND_PORT, {
      DATABASE_URL: toSqliteUrl(dbPath),
      AI_SERVICE_URL: SERVICE_URLS.ai,
      EXAMAC_CONFIG_DIR: configDir,
    });
    await waitForHealth("backend", `${SERVICE_URLS.backend}/health`, backendProcess);
  }

  if (!aiHealthy) {
    const aiProcess = spawnService("ai-service", python, aiServiceDir, AI_SERVICE_PORT, {
      MODEL_CACHE_DIR: modelCacheDir,
      EXAMAC_CONFIG_DIR: configDir,
    });
    await waitForHealth("ai-service", `${SERVICE_URLS.ai}/health`, aiProcess);
  }
}

export function stopManagedServices() {
  for (const service of managedServices.splice(0)) {
    try {
      if (service.process.pid) {
        spawnSync("taskkill", ["/PID", String(service.process.pid), "/T", "/F"], {
          windowsHide: true,
          stdio: "ignore",
        });
      }
    } catch {
      // Ignore shutdown failures.
    }
  }
}
