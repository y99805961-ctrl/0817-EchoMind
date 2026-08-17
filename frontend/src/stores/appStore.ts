import { create } from "zustand";
import { persist } from "zustand/middleware";
import { getHealth } from "../api/chat";
import type { HealthResponse } from "../types/api";

function createUserId(): string {
  if (typeof crypto !== "undefined" && "randomUUID" in crypto) return crypto.randomUUID();
  return `demo-user-${Date.now()}`;
}

interface AppState {
  userId: string;
  health: HealthResponse | null;
  healthLoading: boolean;
  healthError: string | null;
  mobileDrawer: "conversations" | "diagnostics" | null;
  setMobileDrawer: (drawer: AppState["mobileDrawer"]) => void;
  checkHealth: () => Promise<void>;
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      userId: createUserId(),
      health: null,
      healthLoading: false,
      healthError: null,
      mobileDrawer: null,
      setMobileDrawer: (mobileDrawer) => set({ mobileDrawer }),
      checkHealth: async () => {
        set({ healthLoading: true, healthError: null });
        try {
          const health = await getHealth();
          set({ health, healthLoading: false });
        } catch (error) {
          set({ health: null, healthLoading: false, healthError: error instanceof Error ? error.message : "服务不可用" });
        }
      },
    }),
    {
      name: "echomind-app-store",
      partialize: (state) => ({ userId: state.userId }),
    },
  ),
);
