/** Per-model MTP deploy settings (localStorage), keyed by name:variant. */

import mtpRecommendedDefaults from '@/config/mtpRecommendedDefaults.json';

import type { MtpWorkloadProfile } from '@/utils/mtpWorkloadProfiles';

export interface MtpModelSettings {
  enabled: boolean;
  workload_profile: MtpWorkloadProfile;
  draft_n_max: number;
  draft_n_min: number;
  draft_p_min: number;
}

const STORAGE_KEY = 'llama-nexus-mtp-by-model';

const DEFAULTS: MtpModelSettings = {
  enabled: false,
  workload_profile: 'chat',
  draft_n_max: 3,
  draft_n_min: 0,
  draft_p_min: 0.75,
};

export function modelMtpKey(name: string, variant: string): string {
  return `${name}:${variant || 'unknown'}`;
}

function readAll(): Record<string, MtpModelSettings> {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return typeof parsed === 'object' && parsed !== null ? parsed : {};
  } catch {
    return {};
  }
}

/** Match scripts/mtp_bench_lib.infer_model_family for benchmark → UI alignment. */
export function inferMtpFamily(modelName: string): string {
  const n = modelName.toLowerCase();
  if (n.includes('qwen3') || n.includes('qwen-3') || n.includes('qwen3.6')) {
    if (n.includes('a3b') || n.includes('moe') || (n.includes('35b') && n.includes('a3b'))) {
      return 'qwen3.6-moe';
    }
    return 'qwen3.6';
  }
  if (n.includes('glm')) return 'glm';
  return 'default';
}

type RecommendedFile = Record<
  string,
  Partial<MtpModelSettings> & { note?: string; _comment?: string }
>;

export function recommendedMtpForFamily(family: string): MtpModelSettings {
  const table = mtpRecommendedDefaults as RecommendedFile;
  const rec = table[family] ?? table.default ?? {};
  const profile = rec.workload_profile as MtpWorkloadProfile | undefined;
  return {
    enabled: Boolean(rec.enabled ?? DEFAULTS.enabled),
    workload_profile:
      profile && ['chat', 'agent', 'throughput', 'custom'].includes(profile)
        ? profile
        : DEFAULTS.workload_profile,
    draft_n_max: Number(rec.draft_n_max ?? DEFAULTS.draft_n_max),
    draft_n_min: Number(rec.draft_n_min ?? DEFAULTS.draft_n_min),
    draft_p_min: Number(rec.draft_p_min ?? DEFAULTS.draft_p_min),
  };
}

export function loadMtpForModel(name: string, variant: string): MtpModelSettings | null {
  if (!name) return null;
  const entry = readAll()[modelMtpKey(name, variant)];
  if (!entry || typeof entry !== 'object') return null;
  const profile = entry.workload_profile as MtpWorkloadProfile | undefined;
  return {
    enabled: Boolean(entry.enabled),
    workload_profile:
      profile && ['chat', 'agent', 'throughput', 'custom'].includes(profile)
        ? profile
        : DEFAULTS.workload_profile,
    draft_n_max: Number(entry.draft_n_max ?? DEFAULTS.draft_n_max),
    draft_n_min: Number(entry.draft_n_min ?? DEFAULTS.draft_n_min),
    draft_p_min: Number(entry.draft_p_min ?? DEFAULTS.draft_p_min),
  };
}

/** Per-model localStorage override, else benchmark-derived family defaults. */
export function resolveMtpForModel(name: string, variant: string): MtpModelSettings {
  return loadMtpForModel(name, variant) ?? recommendedMtpForFamily(inferMtpFamily(name));
}

export function saveMtpForModel(name: string, variant: string, mtp: Partial<MtpModelSettings>): void {
  if (!name) return;
  const key = modelMtpKey(name, variant);
  const prev = readAll();
  const merged: MtpModelSettings = {
    ...DEFAULTS,
    ...prev[key],
    ...mtp,
  };
  prev[key] = merged;
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(prev));
  } catch {
    // ignore quota errors
  }
}

export function defaultMtpSettings(): MtpModelSettings {
  return { ...DEFAULTS };
}
