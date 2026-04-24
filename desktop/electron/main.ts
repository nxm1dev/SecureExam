import { app, BrowserView, BrowserWindow, dialog, screen, session } from "electron";
import axios from "axios";
import * as path from "path";

import { attachBrowserGuard } from "./browser-guard";
import { registerIpcHandlers } from "./ipc-handlers";
import { ensurePackagedServicesReady, stopManagedServices } from "./service-manager";
import { SERVICE_URLS } from "./service-urls";

if (!app.requestSingleInstanceLock()) {
  app.quit();
}

let mainWindow: BrowserWindow | null = null;
let examView: BrowserView | null = null;
let currentSession: { sessionId: string; userId: string } | null = null;

function createExamView(examUrl: string): void {
  if (!mainWindow) {
    return;
  }

  if (examView) {
    mainWindow.removeBrowserView(examView);
    examView = null;
  }

  examView = new BrowserView({
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });

  mainWindow.addBrowserView(examView);

  const [width, height] = mainWindow.getContentSize();
  examView.setBounds({ x: 0, y: 64, width, height: height - 64 });
  examView.setAutoResize({ width: true, height: true });

  attachBrowserGuard(examView, mainWindow, {
    onViolation: (type, metadata) => {
      mainWindow?.webContents.send("violation:detected", { type, metadata });
      if (currentSession) {
        void logViolationToBackend({
          event_type: type,
          severity: "high",
          metadata,
        });
      }
    },
  });

  void examView.webContents.loadURL(examUrl);
}

function startExamSession(payload: {
  examUrl: string;
  sessionId: string;
  userId: string;
}) {
  currentSession = { sessionId: payload.sessionId, userId: payload.userId };
  createExamView(payload.examUrl);
  mainWindow?.setFullScreen(true);
}

function stopExamSession() {
  currentSession = null;
  if (examView && mainWindow) {
    mainWindow.removeBrowserView(examView);
    examView = null;
  }
  mainWindow?.setFullScreen(false);
}

function attachDevDiagnostics(window: BrowserWindow) {
  window.webContents.on(
    "did-fail-load",
    (_event, errorCode, errorDescription, validatedURL) => {
      console.error("[Renderer] did-fail-load", {
        errorCode,
        errorDescription,
        validatedURL,
      });
    }
  );

  window.webContents.on(
    "console-message",
    (_event, level, message, line, sourceId) => {
      console.log("[Renderer console]", {
        level,
        message,
        line,
        sourceId,
      });
    }
  );

  window.webContents.on("render-process-gone", (_event, details) => {
    console.error("[Renderer] render-process-gone", details);
  });

  window.webContents.on("did-finish-load", () => {
    console.log("[Renderer] did-finish-load", window.webContents.getURL());
  });
}

function createWindow(): void {
  const { width, height } = screen.getPrimaryDisplay().workAreaSize;

  mainWindow = new BrowserWindow({
    width,
    height,
    fullscreen: false,
    titleBarStyle: process.platform === "darwin" ? "hiddenInset" : "default",
    backgroundColor: "#0f1117",
    webPreferences: {
      preload: path.join(__dirname, "preload.js"),
      contextIsolation: true,
      nodeIntegration: false,
      // sandbox must be false so the renderer can call getUserMedia
      sandbox: false,
      webSecurity: true,
    },
  });

  const isDev = !app.isPackaged;
  if (isDev) {
    void mainWindow.loadURL("http://localhost:3000");
    attachDevDiagnostics(mainWindow);
  } else {
    void mainWindow.loadFile(path.join(__dirname, "../renderer/index.html"));
  }

  mainWindow.on("blur", () => {
    if (currentSession) {
      mainWindow?.webContents.send("window:focuslost");
      void logViolationToBackend({
        event_type: "app_focus_lost",
        severity: "medium",
        metadata: {},
      });
    }
  });

  mainWindow.on("leave-full-screen", () => {
    mainWindow?.webContents.send("window:fullscreen", false);
    if (currentSession) {
      void logViolationToBackend({
        event_type: "fullscreen_exit",
        severity: "high",
        metadata: {},
      });
    }
  });

  mainWindow.on("enter-full-screen", () => {
    mainWindow?.webContents.send("window:fullscreen", true);
  });

  mainWindow.on("close", (event) => {
    if (currentSession) {
      event.preventDefault();
      void logViolationToBackend({
        event_type: "app_close_attempt",
        severity: "high",
        metadata: {},
      });
    }
  });
}

async function logViolationToBackend(partial: {
  event_type: string;
  severity: string;
  metadata: Record<string, unknown>;
}) {
  if (!currentSession) {
    return;
  }

  try {
    await axios.post(`${SERVICE_URLS.backend}/violations/`, {
      session_id: currentSession.sessionId,
      user_id: currentSession.userId,
      ...partial,
    });
  } catch (error) {
    console.error("[Main] Failed to log violation to backend:", error);
  }
}

async function bootstrapApp() {
  try {
    await ensurePackagedServicesReady();
  } catch (error: any) {
    dialog.showErrorBox(
      "SecureExam startup failed",
      error?.message || "Khong the khoi dong backend va AI service."
    );
    app.quit();
    return;
  }

  registerIpcHandlers({
    backendUrl: SERVICE_URLS.backend,
    aiUrl: SERVICE_URLS.ai,
    onExamStarted: startExamSession,
    onExamEnded: stopExamSession,
  });

  createWindow();

  app.on("activate", () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
}

app.whenReady().then(() => {
  // Grant camera and microphone permissions automatically.
  // Without this, Electron silently blocks getUserMedia in the renderer.
  session.defaultSession.setPermissionRequestHandler(
    (_webContents, permission, callback) => {
      const allowed = ["media", "camera", "microphone", "audioCapture", "videoCapture"];
      callback(allowed.includes(permission));
    }
  );

  // Also handle permission checks (Electron 20+)
  session.defaultSession.setPermissionCheckHandler(
    (_webContents, permission) => {
      const allowed = ["media", "camera", "microphone", "audioCapture", "videoCapture"];
      return allowed.includes(permission);
    }
  );

  void bootstrapApp();
});

app.on("before-quit", () => {
  stopExamSession();
  stopManagedServices();
});

app.on("window-all-closed", () => {
  if (process.platform !== "darwin") {
    app.quit();
  }
});
