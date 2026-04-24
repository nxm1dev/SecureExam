const { execSync } = require("child_process");

const port = process.argv[2];

if (!port) {
  process.exit(0);
}

function killWindowsPort(targetPort) {
  try {
    const output = execSync(`netstat -ano | findstr :${targetPort}`, {
      stdio: ["ignore", "pipe", "ignore"],
      encoding: "utf8",
    });

    const pids = new Set();
    for (const line of output.split(/\r?\n/)) {
      const trimmed = line.trim();
      if (!trimmed || !trimmed.includes(`:${targetPort}`)) {
        continue;
      }

      const parts = trimmed.split(/\s+/);
      const pid = parts[parts.length - 1];
      if (pid && /^\d+$/.test(pid)) {
        pids.add(pid);
      }
    }

    for (const pid of pids) {
      try {
        execSync(`taskkill /PID ${pid} /F`, {
          stdio: ["ignore", "ignore", "ignore"],
        });
      } catch {
        // Ignore already-exited or inaccessible processes.
      }
    }
  } catch {
    // No process is listening on the port.
  }
}

killWindowsPort(port);
