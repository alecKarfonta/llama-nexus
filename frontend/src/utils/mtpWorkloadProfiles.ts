/** Experiment-backed MTP workload presets (mirrors backend/modules/mtp_deploy.py). */

export type MtpWorkloadProfile = 'chat' | 'agent' | 'throughput' | 'custom';

export const MTP_WORKLOAD_PROFILES: MtpWorkloadProfile[] = [
  'chat',
  'agent',
  'throughput',
  'custom',
];

export const DEFAULT_MTP_WORKLOAD_PROFILE: MtpWorkloadProfile = 'chat';

export interface MtpWorkloadProfileHint {
  title: string;
  detail: string;
}

export const MTP_WORKLOAD_PROFILE_HINTS: Record<
  Exclude<MtpWorkloadProfile, 'custom'>,
  MtpWorkloadProfileHint
> = {
  chat: {
    title: 'Chat',
    detail: 'MTP n2 + preserve_thinking — ~49 tok/s on 2×16GB (free chat benchmark).',
  },
  agent: {
    title: 'Agent / tools',
    detail:
      'MTP n8 + thinking off + cache_reuse — ~87 tok/s on tool path (~3× grammar-only).',
  },
  throughput: {
    title: 'Throughput',
    detail: 'MTP off, parallel_slots=4 — batch / multi-user; lower per-stream latency.',
  },
};

type ProfileServerPatch = {
  jinja?: boolean;
  chat_template_kwargs?: Record<string, unknown> | null;
  cache_reuse?: number | null;
};

type ProfilePatch = {
  mtp: {
    enabled: boolean;
    draft_n_max: number;
    draft_n_min: number;
    draft_p_min: number;
  };
  performance: { parallel_slots: number };
  server: ProfileServerPatch;
};

const PROFILE_PATCHES: Record<Exclude<MtpWorkloadProfile, 'custom'>, ProfilePatch> = {
  chat: {
    mtp: { enabled: true, draft_n_max: 2, draft_n_min: 0, draft_p_min: 0.75 },
    performance: { parallel_slots: 1 },
    server: {
      jinja: true,
      chat_template_kwargs: { preserve_thinking: true },
      cache_reuse: null,
    },
  },
  agent: {
    mtp: { enabled: true, draft_n_max: 8, draft_n_min: 0, draft_p_min: 0.75 },
    performance: { parallel_slots: 1 },
    server: {
      jinja: true,
      chat_template_kwargs: { enable_thinking: false },
      cache_reuse: 1024,
    },
  },
  throughput: {
    mtp: { enabled: false, draft_n_max: 2, draft_n_min: 0, draft_p_min: 0.75 },
    performance: { parallel_slots: 4 },
    server: { jinja: true, chat_template_kwargs: null, cache_reuse: null },
  },
};

/** Sections touched when applying a workload profile from Deploy. */
export interface MtpWorkloadApplyResult<T> {
  mtp: T;
  performance: { parallel_slots: number };
  server: {
    jinja?: boolean;
    chat_template_kwargs?: Record<string, unknown> | null;
    cache_reuse?: number | null;
  };
}

export function applyMtpWorkloadProfile<T extends Record<string, unknown>>(
  profile: MtpWorkloadProfile,
  current: {
    mtp?: T;
    performance?: { parallel_slots?: number };
    server?: Record<string, unknown>;
  }
): MtpWorkloadApplyResult<T> {
  const mtp = { ...(current.mtp ?? {}), workload_profile: profile } as T & {
    workload_profile: MtpWorkloadProfile;
  };

  if (profile === 'custom') {
    return {
      mtp,
      performance: { parallel_slots: current.performance?.parallel_slots ?? 1 },
      server: {
        jinja: current.server?.jinja as boolean | undefined,
        chat_template_kwargs: current.server?.chat_template_kwargs as
          | Record<string, unknown>
          | null
          | undefined,
        cache_reuse: current.server?.cache_reuse as number | null | undefined,
      },
    };
  }

  const patch = PROFILE_PATCHES[profile];
  const server: MtpWorkloadApplyResult<T>['server'] = { jinja: patch.server.jinja };
  if (patch.server.chat_template_kwargs != null) {
    server.chat_template_kwargs = patch.server.chat_template_kwargs;
  }
  if (patch.server.cache_reuse != null) {
    server.cache_reuse = patch.server.cache_reuse;
  }

  return {
    mtp: { ...mtp, ...patch.mtp },
    performance: { ...patch.performance },
    server,
  };
}

/** Force MTP off when the selected GGUF has no prediction heads. */
export function clampMtpForModelCapability<C extends { mtp?: { enabled?: boolean } }>(
  cfg: C,
  modelMtpCapable: boolean
): C {
  if (modelMtpCapable || !cfg.mtp?.enabled) {
    return cfg;
  }
  const next = JSON.parse(JSON.stringify(cfg)) as C;
  next.mtp = { ...(next.mtp ?? {}), enabled: false };
  return next;
}
